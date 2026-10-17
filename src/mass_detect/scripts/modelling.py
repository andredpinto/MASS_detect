"""Models, ways of pooling LNDb and LIDC IDRI, the nested cross validation and the validation across datasets of notebook 06.

Every model gives a probability for each rating from 1 to 5, and the predicted rating is the most probable one (argmax).
The binary answer, suspicious or not, compares the probability of a rating of 4 or 5 with a threshold. The nested cross
validation is grouped by patient: the outer folds measure the performance, and the inner folds only choose the
hyperparameters and the threshold, so the outer folds never take part in any choice (Cawley and Talbot, 2010). The inner
folds are repeated with two seeds, and the choice follows the one standard error rule (Hastie et al., 2009, section
7.10): the simplest candidate whose score is within one standard error of the best.

Besides the four models of stage 2, two extensions of the chosen model are compared in stage 3: the ordinal model with
the rare ratings weighted up (ordinal_balanced), and an elastic net logistic regression that selects features inside the
folds (elastic_net).
"""
import json
import warnings
from itertools import product
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from scipy.stats import rankdata
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score, mean_absolute_error, roc_auc_score
from sklearn.model_selection import ParameterSampler, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .models import OrdinalLogisticRegression, ReferenceBatchComBat

warnings.filterwarnings("ignore", category=ConvergenceWarning)   # saga may stop at max_iter on the weakest penalties


# From probabilities to an answer

def decode(proba, classes):
    """The most probable rating of each row of probabilities (argmax)."""
    return np.asarray(classes)[proba.argmax(axis=-1)]


def suspicious_probability(proba, classes):
    """Probability that a nodule is suspicious: the probability of a rating of 4 or 5."""
    return proba[..., np.asarray(classes) >= 4].sum(axis=-1)


def is_suspicious(y):
    """The binary label of every nodule: 1 for a rating of 4 or 5."""
    return (np.asarray(y) >= 4).astype(int)


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
        return decode(self.predict_proba(X), self.classes_)


class ClassWeighted(BaseEstimator, ClassifierMixin):
    """Fits a model with every nodule weighted by (1 / share of its rating) ** power, so rare ratings count more.

    power 0 keeps the weights as they are, power 1 gives every rating the same total weight, and 0.5 is half way.
    The weights are scaled to a mean of 1, so the penalty of the model keeps its meaning.
    """

    def __init__(self, estimator=None, power=0.0):
        self.estimator = estimator
        self.power = power

    def fit(self, X, y, sample_weight=None):
        y = np.asarray(y)
        ratings, counts = np.unique(y, return_counts=True)
        share = dict(zip(ratings, counts / len(y)))
        weight = np.array([share[rating] for rating in y]) ** -self.power
        weight = weight / weight.mean()
        if sample_weight is not None:
            weight = weight * np.asarray(sample_weight, dtype=float)
        self.model_ = clone(self.estimator).fit(X, y, sample_weight=weight)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


# The models and their hyperparameter grids.
# Every list goes from the simplest value to the most complex one (the strongest regularisation first), so the position
# of a value in its list measures how complex the model is. The one standard error rule of the inner loop relies on it.
# Penalties, learning rates and kernel widths are spaced on a logarithmic scale, because their effect depends on their
# order of magnitude.
#   ordinal            the penalty alpha, over five orders of magnitude so that the chosen value falls inside the grid
#   random_forest      the hyperparameters that matter most for a forest (Probst, Wright and Boulesteix, 2019):
#                      the share of features tried at each split, the size of the leaves, the depth, and an optional
#                      reweighting of the rare ratings. The number of trees is not tuned, because more trees never hurt
#                      (Probst and Boulesteix, 2018): SEARCH_TREES while searching, FINAL_TREES in the final model
#   gradient_boosting  learning rate, size of the trees, size of the leaves and L2 penalty; the number of boosting
#                      rounds is set by early stopping on a part of the training data
#   svm                C and the width gamma of the RBF kernel together, because they interact (Hsu, Chang and Lin)

GRIDS = {
    "ordinal": {"alpha": [3000.0, 1000.0, 300.0, 100.0, 30.0, 10.0, 3.0, 1.0, 0.3, 0.1]},
    "random_forest": {
        "max_features": [0.02, 0.035, 0.05, 0.1, 0.2, 0.3, 0.45, 0.6],
        "min_samples_leaf": [40, 20, 10, 5, 3, 1],
        "max_depth": [8, 16, None],
        "class_weight": [None, "balanced_subsample"],
    },
    "gradient_boosting": {
        "learning_rate": [0.01, 0.02, 0.035, 0.05, 0.08, 0.12, 0.2],
        "max_leaf_nodes": [3, 7, 15, 31, 63],
        "min_samples_leaf": [100, 50, 20, 10],
        "l2_regularization": [30.0, 10.0, 3.0, 1.0, 0.3, 0.1],
    },
    "svm": {
        "C": [2.0 ** -5, 2.0 ** -3, 2.0 ** -1, 2.0, 2.0 ** 3, 2.0 ** 5, 2.0 ** 7],
        "gamma": [2.0 ** -13, 2.0 ** -11, 2.0 ** -9, 2.0 ** -7, 2.0 ** -5, 2.0 ** -3, 2.0 ** -1, 2.0],
    },
    # Stage 3, extensions of the chosen model
    "ordinal_balanced": {                      # the ordinal model with the rare ratings weighted up (see ClassWeighted)
        "alpha": [30000.0, 10000.0, 3000.0, 1000.0, 300.0, 100.0, 30.0, 10.0, 3.0, 1.0, 0.3, 0.1],
        "class_power": [0.0, 0.5, 1.0],
    },
    "elastic_net": {                           # L1 + L2 logistic regression: C is the inverse of the penalty, and the
        "C": [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0],   # L1 share sets how many coefficients
                                                                           # become exactly zero
        "l1_ratio": [1.0, 0.5, 0.2],
    },
}
SAMPLED = {"random_forest": 15, "gradient_boosting": 20}   # random search: candidates drawn from these grids
SEARCH_TREES = 150     # trees of a forest while the inner loop compares candidates
FINAL_TREES = 500      # trees of the forest refitted with the chosen hyperparameters
INNER_FOLDS = 3        # folds of the inner loop
INNER_REPEATS = 2      # the inner folds are drawn twice, with two seeds, to make the choice less noisy
EDGE_LIMIT = 0.2       # a grid must be widened when more than this share of the outer folds choose one of its ends
BOUNDED = {"l1_ratio", "class_power"}   # their ends are natural limits (pure L1, no reweighting), so a choice at an
                                        # end is not a grid that stops too early


def make_model(name, params, final=True):
    """A new, unfitted model: median imputation, scaling and the classifier.

    'final' is False inside the inner loop, where a forest uses fewer trees to compare candidates faster.
    """
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    if name == "ordinal":
        classifier = OrdinalLogisticRegression(alpha=params["alpha"])
    elif name == "random_forest":
        forest = RandomForestClassifier(n_estimators=FINAL_TREES if final else SEARCH_TREES,
                                        max_features=params["max_features"], min_samples_leaf=params["min_samples_leaf"],
                                        max_depth=params["max_depth"], class_weight=params["class_weight"],
                                        n_jobs=1, random_state=0)
        classifier = FrankHallOrdinal(forest)
    elif name == "gradient_boosting":
        # up to 1000 rounds, stopped when the loss on 20% of the training nodules has not improved for 10 rounds
        boosting = HistGradientBoostingClassifier(learning_rate=params["learning_rate"],
                                                  max_leaf_nodes=params["max_leaf_nodes"],
                                                  min_samples_leaf=params["min_samples_leaf"],
                                                  l2_regularization=params["l2_regularization"],
                                                  max_iter=1000, early_stopping=True, validation_fraction=0.2,
                                                  n_iter_no_change=10, random_state=0)
        classifier = FrankHallOrdinal(boosting)
    elif name == "svm":
        svm = SVC(C=params["C"], gamma=params["gamma"], probability=True, random_state=0)
        classifier = FrankHallOrdinal(svm)
    elif name == "ordinal_balanced":
        classifier = ClassWeighted(OrdinalLogisticRegression(alpha=params["alpha"]), power=params["class_power"])
    elif name == "elastic_net":
        # the L1 part sets the coefficients of unhelpful features to zero: feature selection inside every fit
        logistic = LogisticRegression(penalty="elasticnet", solver="saga", C=params["C"], l1_ratio=params["l1_ratio"],
                                      max_iter=2000)
        classifier = FrankHallOrdinal(logistic)
    else:
        raise ValueError(name)

    return make_pipeline(imputer, scaler, classifier)


def grid_points(name):
    """The candidates of one model: every point of its grid, or a random sample of them for the models in SAMPLED.

    A random sample of the grid finds good values faster than walking the whole grid, because only a few
    hyperparameters matter much (Bergstra and Bengio, 2012). It is drawn once, with a fixed seed, so every outer fold
    compares the same candidates.
    """
    grid = GRIDS[name]
    if name in SAMPLED:
        return list(ParameterSampler(grid, SAMPLED[name], random_state=0))
    return [dict(zip(grid, values)) for values in product(*grid.values())]


def _values(params):
    """(name shown in the tables, hyperparameter, value) for every value."""
    return [(key, key, value) for key, value in params.items()]


def flat(params):
    """The hyperparameters as a dictionary, for example {"alpha": 100.0, "class_power": 0.5}."""
    return {shown: value for shown, _, value in _values(params)}


def complexity(name, params):
    """Sum of the positions of the values in their grids: 0 is the simplest point, larger is more complex."""
    return sum(GRIDS[name][key].index(value) for _, key, value in _values(params))


def edge_share(name, choices):
    """Share of the outer folds whose chosen value is the first or the last of its grid, for numeric grids of 3 values
    or more (a grid with None or text, such as the depth of a tree or the class weights, has no "beyond").

    A value chosen at an end of the grid means that the best value may lie beyond it, so the grid has to be widened
    whenever this share is above EDGE_LIMIT.
    """
    at_edge = {}
    for text in choices["params"]:
        for shown, key, value in _values(json.loads(text)):
            grid = GRIDS[name][key]
            numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in grid)
            if len(grid) >= 3 and numeric and key not in BOUNDED:
                at_edge.setdefault(shown, []).append(value in (grid[0], grid[-1]))
    return pd.Series({shown: float(np.mean(flags)) for shown, flags in at_edge.items()}, dtype=float)


# Ways of pooling the two datasets

class PooledModel(BaseEstimator, ClassifierMixin):
    """One way of using LNDb and LIDC IDRI together, around any model of make_model.

    pooled       both datasets as they are
    indicator    one more column that says whether the nodule comes from LIDC IDRI
    combat       the LNDb features moved onto the scale of LIDC IDRI (reference batch), log diameter as covariate

    'params' holds the hyperparameters of the model.
    """

    def __init__(self, name="ordinal", params=None, strategy="pooled", features=(), final=True):
        self.name = name
        self.params = params
        self.strategy = strategy
        self.features = features
        self.final = final      # False inside the inner loop (see make_model)

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

        self.model_ = _fit_pipeline(make_model(self.name, self.params, self.final), self._matrix(frame), y, weight)
        return self

    def predict_proba(self, frame):
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


def quadratic_kappa(y, prediction):
    """Quadratic weighted kappa, the same value as cohen_kappa_score(weights="quadratic") of sklearn.

    It is computed directly from the confusion matrix, because the inner loop scores thousands of candidates and the
    checks of the sklearn function would dominate the time.
    """
    labels = np.union1d(y, prediction)
    k = len(labels)
    observed = np.bincount(np.searchsorted(labels, y) * k + np.searchsorted(labels, prediction),
                           minlength=k * k).reshape(k, k).astype(float)
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / observed.sum()
    weights = (np.arange(k)[:, None] - np.arange(k)[None, :]) ** 2
    return 1 - (weights * observed).sum() / (weights * expected).sum()


def mean_qwk(y, prediction, dataset):
    """The primary metric: the QWK in each dataset, averaged over the datasets present."""
    scores = []
    for d in np.unique(dataset):
        rows = dataset == d
        scores.append(quadratic_kappa(y[rows], prediction[rows]))
    return float(np.mean(scores))


def auc(y, probability, sample_weight=None):
    """Area under the ROC curve: the chance that a suspicious nodule gets a higher probability than a nodule that is not
    (Hanley and McNeil, 1982). Computed from ranks, which is fast enough for the inner loop."""
    y, probability = np.asarray(y), np.asarray(probability)
    if sample_weight is not None:
        return float(roc_auc_score(y, probability, sample_weight=sample_weight))
    positive = y == 1
    n_positive, n_negative = positive.sum(), (~positive).sum()
    if n_positive == 0 or n_negative == 0:
        return np.nan                                  # undefined without both kinds of nodules
    ranks = rankdata(probability)
    return float((ranks[positive].sum() - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative))


def mean_auc(y, probability, dataset):
    """The primary metric of the binary scale: the AUC in each dataset, averaged over the datasets present."""
    return float(np.mean([auc(y[dataset == d], probability[dataset == d]) for d in np.unique(dataset)]))


def binary_metrics(y, probability, prediction):
    """AUC, sensitivity, specificity and balanced accuracy of one set of binary answers."""
    y, prediction = np.asarray(y), np.asarray(prediction)
    sensitivity = float(prediction[y == 1].mean())
    specificity = float(1 - prediction[y == 0].mean())
    return {"auc": auc(y, probability), "sensitivity": sensitivity, "specificity": specificity,
            "balanced_accuracy_binary": (sensitivity + specificity) / 2}


def best_threshold(y, probability, dataset):
    """The threshold with the highest balanced accuracy, averaged over the datasets present.

    The balanced accuracy is the mean of the sensitivity and the specificity, so its best threshold is the one of the
    Youden index (Youden, 1950). It does not depend on the share of suspicious nodules, which differs between the two
    datasets. The candidates are 201 quantiles of the probabilities.
    """
    y, probability, dataset = np.asarray(y), np.asarray(probability), np.asarray(dataset)
    thresholds = np.unique(np.quantile(probability, np.linspace(0, 1, 201)))
    flagged = probability[None, :] >= thresholds[:, None]          # one row per threshold
    scores = []
    for d in np.unique(dataset):
        rows = dataset == d
        positive, negative = rows & (y == 1), rows & (y == 0)
        if not positive.any() or not negative.any():
            continue                                   # a dataset without both kinds of nodules cannot score a threshold
        sensitivity = flagged[:, positive].mean(axis=1)
        specificity = 1 - flagged[:, negative].mean(axis=1)
        scores.append((sensitivity + specificity) / 2)
    return float(thresholds[np.argmax(np.mean(scores, axis=0))])


# Nested cross validation

def _fit(name, params, strategy, features, train, label, weight, final=True):
    """Fits one PooledModel on the rows of 'train'."""
    model = PooledModel(name, params, strategy, features, final)
    sample_weight = None if weight is None else train[weight]
    return model.fit(train, train[label], sample_weight)


def inner_splits(inner_frame, label, seed, inner_folds=INNER_FOLDS, inner_repeats=INNER_REPEATS):
    """The folds of the inner loop: grouped by patient, stratified by dataset and rating, drawn once per repeat."""
    strata = inner_frame["dataset"] + "_" + inner_frame[label].astype(str)
    splits = []
    for repeat in range(inner_repeats):
        splitter = StratifiedGroupKFold(inner_folds, shuffle=True, random_state=seed + 1000 * repeat)
        splits += list(splitter.split(inner_frame, strata, inner_frame["patient_id"]))
    return splits


def inner_scores(name, strategy, features, inner_frame, training, train_label, weight, splits, classes, label):
    """Mean QWK of the most probable rating for every candidate on every inner split: one row per candidate and split.

    A candidate is one set of hyperparameters, written as JSON.
    """
    candidates = grid_points(name)
    y = inner_frame[label].to_numpy()
    dataset = inner_frame["dataset"].to_numpy()
    rows = []
    for split, (inner_train, inner_test) in enumerate(splits):
        fit_rows = training[training["patient_id"].isin(inner_frame["patient_id"].iloc[inner_train])]
        held_out = inner_frame.iloc[inner_test]
        y_test, dataset_test = y[inner_test], dataset[inner_test]
        masks = [dataset_test == d for d in np.unique(dataset_test)]      # one per dataset, computed once per split

        def score(proba):
            # mean QWK of one candidate, with the dataset masks of this split
            prediction = decode(proba, classes)
            return float(np.mean([quadratic_kappa(y_test[m], prediction[m]) for m in masks]))

        for params in candidates:
            model = _fit(name, params, strategy, features, fit_rows, train_label, weight, final=False)
            rows.append((json.dumps(params), split, score(model.predict_proba(held_out))))
    return pd.DataFrame(rows, columns=["params", "split", "score"])


def summarise_candidates(name, scores):
    """Mean score over the inner splits, its standard error and the complexity of every candidate."""
    table = scores.groupby("params")["score"].agg(["mean", "std", "count"]).reset_index()
    table = table.rename(columns={"mean": "score"})
    table["se"] = table["std"] / np.sqrt(table["count"])
    table["complexity"] = [complexity(name, json.loads(key)) for key in table["params"]]
    return table[["params", "score", "se", "complexity"]]


def one_standard_error(table):
    """The one standard error rule (Hastie et al., 2009, section 7.10) on a table of candidates.

    'table' has one row per candidate with its mean score, the standard error of that mean and the
    complexity of the candidate. Every candidate whose score is at least the best score minus its standard error is
    as good as the best within the noise of the inner folds; among them the simplest wins, and among equally simple
    ones the highest score.
    """
    best = table.loc[table["score"].idxmax()]
    good_enough = table[table["score"] >= best["score"] - best["se"]]
    return good_enough.sort_values(["complexity", "score"], ascending=[True, False]).iloc[0]


def inner_threshold(name, params, strategy, features, inner_frame, training, train_label, weight, splits, classes, label):
    """The threshold of the binary answer, chosen in the inner folds.

    The model with the chosen hyperparameters predicts the held out nodules of every inner split, and the threshold with
    the best balanced accuracy over those predictions is kept. The nodules of the outer fold never take part, because a
    threshold chosen on the nodules that are evaluated makes sensitivity and specificity look better than they are
    (Leeflang et al., 2008).
    """
    probability, y, dataset = [], [], []
    for inner_train, inner_test in splits:
        fit_rows = training[training["patient_id"].isin(inner_frame["patient_id"].iloc[inner_train])]
        model = _fit(name, params, strategy, features, fit_rows, train_label, weight, final=False)
        held_out = inner_frame.iloc[inner_test]
        probability.append(suspicious_probability(model.predict_proba(held_out), classes))
        y.append(is_suspicious(held_out[label]))
        dataset.append(held_out["dataset"].to_numpy())
    return best_threshold(np.concatenate(y), np.concatenate(probability), np.concatenate(dataset))


def _outer_job(job):
    """One outer fold: the inner loop chooses the hyperparameters and the threshold, then the model is refitted and predicts."""
    name, strategy, features, frame, training, label, weight, train_patients, test_rows, seed = job
    classes = np.sort(frame[label].unique())
    train_label = label

    # Inner loop, on the training part of this outer fold only: 3 folds grouped by patient, drawn with 2 seeds
    inner_frame = frame[frame["patient_id"].isin(train_patients)].reset_index(drop=True)
    splits = inner_splits(inner_frame, label, seed)
    scores = inner_scores(name, strategy, features, inner_frame, training, train_label, weight, splits, classes, label)
    table = summarise_candidates(name, scores)
    params = json.loads(one_standard_error(table)["params"])
    threshold = inner_threshold(name, params, strategy, features, inner_frame, training, train_label, weight, splits,
                                classes, label)

    # Refit on the whole training part with the chosen hyperparameters and predict the held out fold
    fit_rows = training[training["patient_id"].isin(train_patients)]
    model = _fit(name, params, strategy, features, fit_rows, train_label, weight)
    proba = model.predict_proba(frame.iloc[test_rows])
    return seed, test_rows, proba, params, threshold, table


def nested_cv(name, strategy, frame, features, label="rating", seeds=(0, 1, 2), folds=5, training=None, weight=None,
              processes=10):
    """Nested cross validation grouped by patient, with the outer folds stratified by dataset and rating.

    'frame' holds the nodules that are evaluated, with their label. 'training' is the table the models learn from: the
    same nodules by default, or the same nodules with another label. Returns the out of fold probabilities
    (seeds x nodules x ratings), the most probable rating, the probability of a suspicious nodule and the binary answer
    with the threshold of each outer fold, the hyperparameters and threshold chosen in each outer fold, and the inner
    scores of every candidate in every outer fold (to draw the validation curves).
    """
    frame = frame.reset_index(drop=True)
    if training is None:
        training = frame
    classes = np.sort(frame[label].unique())
    strata = frame["dataset"] + "_" + frame[label].astype(str)

    # One job per outer fold and seed, run in parallel
    jobs, fold_of = [], []
    for seed in seeds:
        splitter = StratifiedGroupKFold(folds, shuffle=True, random_state=seed)
        for fold, (train, test) in enumerate(splitter.split(frame, strata, frame["patient_id"])):
            train_patients = frame["patient_id"].iloc[train].unique()
            jobs.append((name, strategy, features, frame, training, label, weight, train_patients, test, seed))
            fold_of.append(fold)
    with Pool(min(processes, len(jobs))) as pool:
        results = pool.map(_outer_job, jobs)

    # Put every fold back in place
    proba = np.zeros((len(seeds), len(frame), len(classes)))
    flagged = np.zeros((len(seeds), len(frame)), dtype=int)
    choices, inner_scores = [], []
    for fold, (seed, test, fold_proba, params, threshold, table) in zip(fold_of, results):
        s = list(seeds).index(seed)
        proba[s, test] = fold_proba
        flagged[s, test] = suspicious_probability(fold_proba, classes) >= threshold
        choices.append({"seed": seed, "outer_fold": fold, "fold_size": len(test), "threshold": threshold,
                        "params": json.dumps(params), **flat(params)})
        inner_scores.append(table.assign(seed=seed, outer_fold=fold))
    return {"proba": proba, "decoded": decode(proba, classes), "suspicious_proba": suspicious_probability(proba, classes),
            "suspicious_pred": flagged, "choices": pd.DataFrame(choices),
            "inner_scores": pd.concat(inner_scores, ignore_index=True), "classes": classes}


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


def _mean_auc_over_seeds(y, probability, dataset, rows):
    """Mean AUC of the given rows, averaged over the seeds (one row of 'probability' per seed)."""
    return np.mean([mean_auc(y[rows], p[rows], dataset[rows]) for p in probability])


def binary_summary(y, probability, prediction, dataset, groups, n_boot=1000):
    """Binary metrics per dataset (mean over the seeds), and the mean AUC with its 95% interval."""
    table = {}
    for d in np.unique(dataset):
        rows = dataset == d
        table[d] = pd.DataFrame([binary_metrics(y[rows], p[rows], f[rows]) for p, f in zip(probability, prediction)]).mean()
    table = pd.DataFrame(table).T

    everyone = np.arange(len(y))
    boots = [_mean_auc_over_seeds(y, probability, dataset, rows) for rows in patient_bootstrap(groups, n_boot)]
    primary = {"mean_auc": _mean_auc_over_seeds(y, probability, dataset, everyone),
               "auc_low": float(np.quantile(boots, 0.025)), "auc_high": float(np.quantile(boots, 0.975))}
    return table, primary


def paired_auc_difference(y, probability_a, probability_b, dataset, groups, n_boot=1000):
    """Mean AUC of B minus A, with a 95% interval computed on the same resampled patients."""
    everyone = np.arange(len(y))
    boots = [_mean_auc_over_seeds(y, probability_b, dataset, rows) - _mean_auc_over_seeds(y, probability_a, dataset, rows)
             for rows in patient_bootstrap(groups, n_boot)]
    difference = (_mean_auc_over_seeds(y, probability_b, dataset, everyone)
                  - _mean_auc_over_seeds(y, probability_a, dataset, everyone))
    return {"difference": difference, "low": float(np.quantile(boots, 0.025)), "high": float(np.quantile(boots, 0.975))}


def reading_agreement(frame, decoded, probability, readings):
    """Agreement of the model with every single radiologist, whatever rule combined their ratings into the label.

    Every reading of a nodule is compared with the answer of the model for that nodule, with a weight of one divided by
    the number of readers, so that every nodule weighs the same. Returns, per dataset and averaged over the seeds, the
    QWK of the rating on the 1 to 5 scale and the AUC of the probability of a suspicious nodule against each reader's
    binary answer.
    """
    readings = readings.dropna(subset=["reader_rating"])
    readings = readings.assign(weight=1 / readings.groupby("patch_id")["reader_rating"].transform("size"))
    position = pd.Series(np.arange(len(frame)), index=frame["patch_id"].to_numpy())
    rows = position[readings["patch_id"]].to_numpy()
    rating = readings["reader_rating"].to_numpy().astype(int)
    out = {}
    for d in np.unique(readings["dataset"]):
        mask = (readings["dataset"] == d).to_numpy()
        weight = readings["weight"].to_numpy()[mask]
        out[f"reader_qwk_{d}"] = np.mean([cohen_kappa_score(rating[mask], prediction[rows][mask], weights="quadratic",
                                                            sample_weight=weight) for prediction in decoded])
        out[f"reader_auc_{d}"] = np.mean([auc((rating[mask] >= 4).astype(int), p[rows][mask], sample_weight=weight)
                                          for p in probability])
    out["reader_qwk"] = np.mean([v for k, v in out.items() if k.startswith("reader_qwk_")])
    out["reader_auc"] = np.mean([v for k, v in out.items() if k.startswith("reader_auc_")])
    return out


# Validation across datasets: a model trained in one hospital and used in another

def harmonise_to(frame, features, reference):
    """ComBat with 'reference' as the reference batch, fitted on the features of both datasets without any label.

    The reference dataset keeps its values, and the other dataset is moved onto its scale, with the log diameter as the
    covariate that is kept (Orlhac et al., 2022). A hospital can do this with its own scans before using a model trained
    elsewhere, because no rating is needed.
    """
    covariate = np.log(frame["roi_diameter_mm"].fillna(frame["roi_diameter_mm"].median())).to_numpy()
    combat = ReferenceBatchComBat(list(features), reference=reference)
    combat.fit(frame[list(features)], frame["dataset"].to_numpy(), covariate)
    return combat.transform(frame, frame["dataset"].to_numpy(), covariate)


def external_validation(name, frame, features, source, label="rating", seeds=(0, 1, 2), harmonise=False, processes=3):
    """Trains on every nodule of the 'source' dataset and predicts every nodule of the other dataset.

    The hyperparameters and the threshold are chosen in inner folds drawn from the source dataset only, grouped by
    patient, once per seed; the other dataset never takes part in any choice. With 'harmonise', the features of the
    other dataset are first moved onto the scale of the source dataset (harmonise_to).
    """
    frame = frame.reset_index(drop=True)
    if harmonise:
        frame = harmonise_to(frame, features, reference=source)
    classes = np.sort(frame[label].unique())
    train_patients = frame.loc[frame["dataset"] == source, "patient_id"].unique()
    test_rows = np.flatnonzero((frame["dataset"] != source).to_numpy())
    jobs = [(name, "pooled", features, frame, frame, label, None, train_patients, test_rows, seed) for seed in seeds]
    with Pool(min(processes, len(jobs))) as pool:
        results = pool.map(_outer_job, jobs)

    proba = np.stack([fold_proba for _, _, fold_proba, _, _, _ in results])
    thresholds = np.array([threshold for _, _, _, _, threshold, _ in results])
    probability = suspicious_probability(proba, classes)
    choices = pd.DataFrame([{"seed": seed, "threshold": threshold, "params": json.dumps(params), **flat(params)}
                            for seed, _, _, params, threshold, _ in results])
    return {"proba": proba, "decoded": decode(proba, classes), "suspicious_proba": probability,
            "suspicious_pred": (probability >= thresholds[:, None]).astype(int), "choices": choices,
            "rows": test_rows, "classes": classes}


# Interpretability

def permutation_importance_cv(name, strategy, frame, features, choices, label="rating", seed=0, folds=5, repeats=3,
                              training=None, weight=None):
    """Drop of the mean QWK when one feature is shuffled inside the held out fold (Molnar, 2022).

    Uses the outer folds of one seed, rebuilt with the same seed so that they come in the order in which nested_cv stored
    its choices. In each fold the model is refitted with the hyperparameters chosen for that fold.
    """
    frame = frame.reset_index(drop=True)
    if training is None:
        training = frame
    classes = np.sort(frame[label].unique())
    strata = frame["dataset"] + "_" + frame[label].astype(str)
    fold_choices = choices[choices["seed"] == seed].sort_values("outer_fold").reset_index(drop=True)
    splitter = StratifiedGroupKFold(folds, shuffle=True, random_state=seed)
    folds_and_choices = zip(splitter.split(frame, strata, frame["patient_id"]), fold_choices.itertuples(index=False))
    rng = np.random.default_rng(seed)
    drops = {feature: [] for feature in features}

    for (train, test), choice in folds_and_choices:
        params = json.loads(choice.params)
        fit_rows = training[training["patient_id"].isin(frame["patient_id"].iloc[train])]
        model = _fit(name, params, strategy, features, fit_rows, label, weight)

        held_out = frame.iloc[test]
        y = held_out[label].to_numpy()
        dataset = held_out["dataset"].to_numpy()
        base = mean_qwk(y, decode(model.predict_proba(held_out), classes), dataset)

        for feature in features:
            for _ in range(repeats):
                shuffled = held_out.copy()
                shuffled[feature] = rng.permutation(shuffled[feature].to_numpy())
                score = mean_qwk(y, decode(model.predict_proba(shuffled), classes), dataset)
                drops[feature].append(base - score)

    table = pd.DataFrame({
        "importance": {feature: np.mean(values) for feature, values in drops.items()},
        "sd": {feature: np.std(values) for feature, values in drops.items()},
    })
    return table.sort_values("importance", ascending=False)
