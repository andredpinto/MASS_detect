"""Models, ways of pooling LNDb and LIDC IDRI, decision rules and the nested cross validation of notebook 06.

Every model gives a probability for each rating from 1 to 5, and a decision rule turns those probabilities into one
rating. The nested cross validation is grouped by patient: the outer folds measure the performance, and the inner folds
only choose the hyperparameters and the decision rule, so the outer folds never take part in any choice
(Cawley and Talbot, 2010).
"""
from itertools import product
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score, mean_absolute_error
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .models import OrdinalLogisticRegression, ReferenceBatchComBat

RULES = ("argmax", "median", "mean")


# Decision rules

def decode(proba, classes, rule):
    """Turns each row of probabilities into one rating."""
    classes = np.asarray(classes)
    if rule == "argmax":
        # the most probable rating
        return classes[proba.argmax(axis=1)]
    if rule == "median":
        # the first rating whose cumulative probability reaches 0.5
        cumulative = proba.cumsum(axis=1)
        return classes[(cumulative >= 0.5).argmax(axis=1)]
    if rule == "mean":
        # the expected rating, rounded half up like the labels
        expected = proba @ classes
        rounded = np.floor(expected + 0.5)
        return np.clip(rounded, classes.min(), classes.max()).astype(int)
    raise ValueError(rule)


# Ordinal wrapper for classifiers that ignore the order of the ratings

class FrankHallOrdinal(BaseEstimator, ClassifierMixin):
    """Frank and Hall (2001): one binary model per question "is the rating above k?".

    The probability of each rating is the difference between two consecutive answers.
    """

    def __init__(self, estimator=None):
        self.estimator = estimator

    def fit(self, X, y, sample_weight=None):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        extra = {} if sample_weight is None else {"sample_weight": sample_weight}
        self.models_ = []
        for k in self.classes_[:-1]:
            above_k = (y > k).astype(int)
            self.models_.append(clone(self.estimator).fit(X, above_k, **extra))
        return self

    def predict_proba(self, X):
        # column k holds P(rating > k), which cannot grow with k
        above = np.column_stack([model.predict_proba(X)[:, 1] for model in self.models_])
        above = np.minimum.accumulate(above, axis=1)

        first = 1 - above[:, 0]
        middle = above[:, :-1] - above[:, 1:]
        last = above[:, -1]
        proba = np.column_stack([first, middle, last])
        proba = np.clip(proba, 0, None)
        return proba / proba.sum(axis=1, keepdims=True)

    def predict(self, X):
        return decode(self.predict_proba(X), self.classes_, "argmax")


# The models and their hyperparameter grids

GRIDS = {
    "ordinal": {"alpha": [0.1, 1.0, 10.0, 100.0]},
    "random_forest": {"min_samples_leaf": [5, 20], "max_features": ["sqrt", 0.3]},
    "gradient_boosting": {"learning_rate": [0.05, 0.1], "max_leaf_nodes": [15, 31]},
    "svm": {"C": [0.5, 2.0]},
}


def make_model(name, params):
    """A new, unfitted model: median imputation, scaling and the classifier."""
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    if name == "ordinal":
        classifier = OrdinalLogisticRegression(alpha=params["alpha"])
    elif name == "random_forest":
        forest = RandomForestClassifier(n_estimators=300, min_samples_leaf=params["min_samples_leaf"],
                                        max_features=params["max_features"], n_jobs=1, random_state=0)
        classifier = FrankHallOrdinal(forest)
    elif name == "gradient_boosting":
        boosting = HistGradientBoostingClassifier(learning_rate=params["learning_rate"],
                                                  max_leaf_nodes=params["max_leaf_nodes"],
                                                  max_iter=200, l2_regularization=1.0, random_state=0)
        classifier = FrankHallOrdinal(boosting)
    elif name == "svm":
        svm = SVC(C=params["C"], gamma="scale", probability=True, random_state=0)
        classifier = FrankHallOrdinal(svm)
    else:
        raise ValueError(name)

    return make_pipeline(imputer, scaler, classifier)


def grid_points(name):
    """Every combination of the grid of one model, as a list of dictionaries."""
    grid = GRIDS[name]
    return [dict(zip(grid, values)) for values in product(*grid.values())]


# Ways of pooling the two datasets

class PooledModel(BaseEstimator, ClassifierMixin):
    """One way of using LNDb and LIDC IDRI together, around any model of make_model.

    pooled       both datasets as they are
    indicator    one more column that says whether the nodule comes from LIDC IDRI
    combat       the LNDb features moved onto the scale of LIDC IDRI (reference batch), log diameter as covariate
    per_dataset  one model per dataset, and the average of their probabilities
    """

    def __init__(self, name="ordinal", params=None, strategy="pooled", features=()):
        self.name = name
        self.params = params
        self.strategy = strategy
        self.features = features

    def _matrix(self, frame):
        # the feature table the model sees, after the indicator column or ComBat
        X = frame[list(self.features)]
        if self.strategy == "indicator":
            X = X.assign(is_lidc=(frame["dataset"] == "LIDC").astype(float))
        if self.strategy == "combat":
            X = self.combat_.transform(X, frame["dataset"].to_numpy(), self._covariate(frame))
        return X

    def _covariate(self, frame):
        # log diameter of the consensus region; the median fills the few regions without features
        diameter = frame["roi_diameter_mm"].fillna(self.diameter_median_)
        return np.log(diameter).to_numpy()

    def fit(self, frame, y, sample_weight=None):
        y = np.asarray(y)
        weight = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
        self.classes_ = np.unique(y)
        self.diameter_median_ = frame["roi_diameter_mm"].median()

        if self.strategy == "combat":
            self.combat_ = ReferenceBatchComBat(list(self.features), reference="LIDC")
            self.combat_.fit(frame[list(self.features)], frame["dataset"].to_numpy(), self._covariate(frame))

        if self.strategy == "per_dataset":
            self.models_ = {}
            for dataset in frame["dataset"].unique():
                rows = (frame["dataset"] == dataset).to_numpy()
                X = frame.loc[rows, list(self.features)]
                rows_weight = None if weight is None else weight[rows]
                self.models_[dataset] = _fit_pipeline(make_model(self.name, self.params), X, y[rows], rows_weight)
        else:
            self.model_ = _fit_pipeline(make_model(self.name, self.params), self._matrix(frame), y, weight)
        return self

    def predict_proba(self, frame):
        if self.strategy == "per_dataset":
            X = frame[list(self.features)]
            each = [_full(model, model.predict_proba(X), self.classes_) for model in self.models_.values()]
            return np.mean(each, axis=0)
        return _full(self.model_, self.model_.predict_proba(self._matrix(frame)), self.classes_)


def _fit_pipeline(model, X, y, weight):
    """Fits a pipeline; the sample weights go to its last step, the classifier."""
    if weight is None:
        return model.fit(X, y)
    last_step = model.steps[-1][0]
    return model.fit(X, y, **{f"{last_step}__sample_weight": weight})


def _full(model, proba, classes):
    """Probabilities on every rating, even when a model never saw one of them in training."""
    seen = model.classes_ if hasattr(model, "classes_") else model[-1].classes_
    out = np.zeros((len(proba), len(classes)))
    out[:, np.searchsorted(classes, seen)] = proba
    return out


# Metrics

def metrics(y, prediction):
    """The four metrics of one set of predictions."""
    return {
        "qwk": cohen_kappa_score(y, prediction, weights="quadratic"),
        "balanced_accuracy": balanced_accuracy_score(y, prediction),
        "macro_f1": f1_score(y, prediction, average="macro"),
        "mae": mean_absolute_error(y, prediction),
    }


def mean_qwk(y, prediction, dataset):
    """The primary metric: the QWK in each dataset, averaged over the datasets present."""
    scores = []
    for d in np.unique(dataset):
        rows = dataset == d
        scores.append(cohen_kappa_score(y[rows], prediction[rows], weights="quadratic"))
    return float(np.mean(scores))


# Nested cross validation

def _train_label(training, label):
    """The column to learn from: the reader's own rating when training on readings, the nodule label otherwise."""
    if "reader_rating" in training and label == "rating":
        return "reader_rating"
    return label


def _fit(name, params, strategy, features, train, label, weight):
    """Fits one PooledModel on the rows of 'train'."""
    model = PooledModel(name, params, strategy, features)
    sample_weight = None if weight is None else train[weight]
    return model.fit(train, train[label], sample_weight)


def _outer_job(job):
    """One outer fold: the inner folds choose the hyperparameters and the rule, then the model is refitted and predicts."""
    name, strategy, features, frame, training, label, weight, train_patients, test_rows, seed, rules, inner_folds = job
    classes = np.sort(frame[label].unique())
    train_label = _train_label(training, label)
    grid = grid_points(name)

    # Inner loop: out of fold probabilities of every grid point on the training part of this outer fold
    inner_frame = frame[frame["patient_id"].isin(train_patients)]
    strata = inner_frame["dataset"] + "_" + inner_frame[label].astype(str)
    splitter = StratifiedGroupKFold(inner_folds, shuffle=True, random_state=seed)
    probas = {i: np.zeros((len(inner_frame), len(classes))) for i in range(len(grid))}
    for inner_train, inner_test in splitter.split(inner_frame, strata, inner_frame["patient_id"]):
        inner_patients = inner_frame["patient_id"].iloc[inner_train]
        fit_rows = training[training["patient_id"].isin(inner_patients)]
        for i, params in enumerate(grid):
            model = _fit(name, params, strategy, features, fit_rows, train_label, weight)
            probas[i][inner_test] = model.predict_proba(inner_frame.iloc[inner_test])

    # Score every pair (grid point, decision rule) and keep the best one
    y_inner = inner_frame[label].to_numpy()
    dataset_inner = inner_frame["dataset"].to_numpy()
    scores = {}
    for (i, proba), rule in product(probas.items(), rules):
        scores[(i, rule)] = mean_qwk(y_inner, decode(proba, classes, rule), dataset_inner)
    best, rule = max(scores, key=scores.get)
    params = grid[best]

    # Refit on the whole training part and predict the held out fold
    fit_rows = training[training["patient_id"].isin(train_patients)]
    model = _fit(name, params, strategy, features, fit_rows, train_label, weight)
    proba = model.predict_proba(frame.iloc[test_rows])
    return seed, test_rows, proba, params, rule, best


def nested_cv(name, strategy, frame, features, label="rating", seeds=(0, 1, 2), folds=5, inner_folds=3, rules=RULES,
              training=None, weight=None, processes=10):
    """Nested cross validation grouped by patient, with the outer folds stratified by dataset and rating.

    'frame' holds the nodules that are evaluated. 'training' is the table the models learn from: the same nodules by
    default, or the readings. Returns the out of fold probabilities (seeds x nodules x ratings), the rating decoded with
    the rule chosen for each outer fold, and the hyperparameters and rule chosen in each outer fold.
    """
    frame = frame.reset_index(drop=True)
    if training is None:
        training = frame
    classes = np.sort(frame[label].unique())
    strata = frame["dataset"] + "_" + frame[label].astype(str)

    # One job per outer fold and seed, run in parallel
    jobs = []
    for seed in seeds:
        splitter = StratifiedGroupKFold(folds, shuffle=True, random_state=seed)
        for train, test in splitter.split(frame, strata, frame["patient_id"]):
            train_patients = frame["patient_id"].iloc[train].unique()
            jobs.append((name, strategy, features, frame, training, label, weight, train_patients, test, seed, rules,
                         inner_folds))
    with Pool(min(processes, len(jobs))) as pool:
        results = pool.map(_outer_job, jobs)

    # Put every fold back in place
    proba = np.zeros((len(seeds), len(frame), len(classes)))
    decoded = np.zeros((len(seeds), len(frame)), dtype=int)
    choices = []
    for seed, test, fold_proba, params, rule, grid_index in results:
        s = list(seeds).index(seed)
        proba[s, test] = fold_proba
        decoded[s, test] = decode(fold_proba, classes, rule)
        choices.append({"seed": seed, "fold_size": len(test), "rule": rule, "grid_index": grid_index, **params})
    return {"proba": proba, "decoded": decoded, "choices": pd.DataFrame(choices), "classes": classes}


# Bootstrap over patients

def patient_bootstrap(groups, n_boot=1000, seed=0):
    """Yields resampled row indices, drawing whole patients with replacement (Field and Welsh, 2007)."""
    rows_of_patient = pd.Series(np.arange(len(groups))).groupby(np.asarray(groups)).apply(np.array).to_numpy()
    n_patients = len(rows_of_patient)
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        drawn = rng.integers(0, n_patients, n_patients)
        yield np.concatenate(rows_of_patient[drawn])


def _mean_over_seeds(y, decoded, dataset, rows):
    """Mean QWK of the given rows, averaged over the seeds (one row of 'decoded' per seed)."""
    return np.mean([mean_qwk(y[rows], prediction[rows], dataset[rows]) for prediction in decoded])


def summary(y, decoded, dataset, groups, n_boot=1000):
    """Every metric per dataset (mean over the seeds), and the mean QWK with its 95% interval."""
    table = {}
    for d in np.unique(dataset):
        rows = dataset == d
        table[d] = pd.DataFrame([metrics(y[rows], prediction[rows]) for prediction in decoded]).mean()
    table = pd.DataFrame(table).T

    everyone = np.arange(len(y))
    boots = [_mean_over_seeds(y, decoded, dataset, rows) for rows in patient_bootstrap(groups, n_boot)]
    primary = {"mean_qwk": _mean_over_seeds(y, decoded, dataset, everyone),
               "low": float(np.quantile(boots, 0.025)), "high": float(np.quantile(boots, 0.975))}
    return table, primary


def paired_difference(y, decoded_a, decoded_b, dataset, groups, n_boot=1000):
    """Mean QWK of B minus A, with a 95% interval computed on the same resampled patients."""
    everyone = np.arange(len(y))
    boots = []
    for rows in patient_bootstrap(groups, n_boot):
        boots.append(_mean_over_seeds(y, decoded_b, dataset, rows) - _mean_over_seeds(y, decoded_a, dataset, rows))
    difference = _mean_over_seeds(y, decoded_b, dataset, everyone) - _mean_over_seeds(y, decoded_a, dataset, everyone)
    return {"difference": difference, "low": float(np.quantile(boots, 0.025)), "high": float(np.quantile(boots, 0.975))}


# Interpretability

def permutation_importance_cv(name, strategy, frame, features, choices, label="rating", seed=0, folds=5, repeats=3,
                              training=None, weight=None):
    """Drop of the mean QWK when one feature is shuffled inside the held out fold (Molnar, 2022).

    Uses the outer folds of one seed, rebuilt with the same seed so that they come in the order in which nested_cv stored
    its choices. In each fold the model is refitted with the hyperparameters and the rule chosen for that fold.
    """
    frame = frame.reset_index(drop=True)
    if training is None:
        training = frame
    classes = np.sort(frame[label].unique())
    strata = frame["dataset"] + "_" + frame[label].astype(str)
    fold_choices = choices[choices["seed"] == seed].reset_index(drop=True)
    splitter = StratifiedGroupKFold(folds, shuffle=True, random_state=seed)
    folds_and_choices = zip(splitter.split(frame, strata, frame["patient_id"]), fold_choices.itertuples(index=False))
    rng = np.random.default_rng(seed)
    drops = {feature: [] for feature in features}

    for (train, test), choice in folds_and_choices:
        params = grid_points(name)[int(choice.grid_index)]
        fit_rows = training[training["patient_id"].isin(frame["patient_id"].iloc[train])]
        model = _fit(name, params, strategy, features, fit_rows, _train_label(training, label), weight)

        held_out = frame.iloc[test]
        y = held_out[label].to_numpy()
        dataset = held_out["dataset"].to_numpy()
        base = mean_qwk(y, decode(model.predict_proba(held_out), classes, choice.rule), dataset)

        for feature in features:
            for _ in range(repeats):
                shuffled = held_out.copy()
                shuffled[feature] = rng.permutation(shuffled[feature].to_numpy())
                score = mean_qwk(y, decode(model.predict_proba(shuffled), classes, choice.rule), dataset)
                drops[feature].append(base - score)

    table = pd.DataFrame({
        "importance": {feature: np.mean(values) for feature, values in drops.items()},
        "sd": {feature: np.std(values) for feature, values in drops.items()},
    })
    return table.sort_values("importance", ascending=False)
