"""Models, ways of pooling LNDb and LIDC IDRI, decision rules and the nested cross validation of notebook 06.

Every model gives a probability for each rating from 1 to 5, and a decision rule turns those probabilities into one
rating. The nested cross validation is grouped by patient: the outer folds measure the performance, and the inner folds
only choose the hyperparameters and the decision rule, so the outer folds never take part in any choice
(Cawley and Talbot, 2010). The inner folds are repeated with two seeds, and the choice follows the one standard error
rule (Hastie et al., 2009, section 7.10): the simplest candidate whose score is within one standard error of the best.

Besides the four models of stage 2, three extensions of the chosen model are compared in stage 5: the ordinal model with
the rare ratings weighted up (ordinal_balanced), an elastic net logistic regression that selects features inside the
folds (elastic_net), and a weighted average of the ordinal model and the random forest (ordinal_rf).
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
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score, mean_absolute_error
from sklearn.model_selection import ParameterSampler, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .models import OrdinalLogisticRegression, ReferenceBatchComBat

RULES = ("argmax", "median", "mean")
warnings.filterwarnings("ignore", category=ConvergenceWarning)   # saga may stop at max_iter on the weakest penalties


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


class WeightedEnsemble(BaseEstimator, ClassifierMixin):
    """Weighted average of the probabilities of several models, each a full pipeline of make_model.

    A weighted average with weights between 0 and 1 that sum to 1 is a constrained form of stacking, the most stable
    one when data are few (Breiman, 1996). A model with weight 0 is not trained at all.
    """

    def __init__(self, estimators=(), weights=()):
        self.estimators = estimators
        self.weights = weights

    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.unique(np.asarray(y))
        self.models_ = [_fit_pipeline(clone(model), X, y, sample_weight) if w > 0 else None
                        for model, w in zip(self.estimators, self.weights)]
        return self

    def predict_proba(self, X):
        proba = np.zeros((len(X), len(self.classes_)))
        for model, w in zip(self.models_, self.weights):
            if model is not None:
                proba += w * _full(model, model.predict_proba(X), self.classes_)
        return proba


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
    # Stage 5, extensions of the chosen model
    "ordinal_balanced": {                      # the ordinal model with the rare ratings weighted up (see ClassWeighted)
        "alpha": [30000.0, 10000.0, 3000.0, 1000.0, 300.0, 100.0, 30.0, 10.0, 3.0, 1.0, 0.3, 0.1],
        "class_power": [0.0, 0.5, 1.0],
    },
    "elastic_net": {                           # L1 + L2 logistic regression: C is the inverse of the penalty, and the
        "C": [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0],   # L1 share sets how many coefficients
                                                                           # become exactly zero
        "l1_ratio": [1.0, 0.5, 0.2],
    },
    "ordinal_rf": {                            # weight of the ordinal model; the random forest gets the rest
        "w": [1.0, 0.75, 0.5, 0.25, 0.0],
    },
}
ENSEMBLES = {"ordinal_rf": ("ordinal", "random_forest")}   # weighted averages and the models they combine
SAMPLED = {"random_forest": 15, "gradient_boosting": 20}   # random search: candidates drawn from these grids
SEARCH_TREES = 150     # trees of a forest while the inner loop compares candidates
FINAL_TREES = 500      # trees of the forest refitted with the chosen hyperparameters
INNER_FOLDS = 3        # folds of the inner loop
INNER_REPEATS = 2      # the inner folds are drawn twice, with two seeds, to make the choice less noisy
EDGE_LIMIT = 0.2       # a grid must be widened when more than this share of the outer folds choose one of its ends
BOUNDED = {"w", "l1_ratio", "class_power"}   # their ends are natural limits (only the ordinal model, pure L1, no
                                             # reweighting), so a choice at an end is not a grid that stops too early


def make_model(name, params, final=True):
    """A new, unfitted model: median imputation, scaling and the classifier.

    'final' is False inside the inner loop, where a forest uses fewer trees to compare candidates faster.
    """
    if name in ENSEMBLES:
        # each model of the average keeps the hyperparameters chosen for it in its own inner loop
        first, second = ENSEMBLES[name]
        models = [make_model(first, params[first], final), make_model(second, params[second], final)]
        return WeightedEnsemble(models, [params["w"], 1 - params["w"]])

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
    """(name shown in the tables, hyperparameter, value) for every value; one set per dataset with per_dataset models."""
    if all(isinstance(value, dict) for value in params.values()):
        return [(f"{key}_{dataset}", key, value) for dataset, own in params.items() for key, value in own.items()
                if not isinstance(value, dict)]
    return [(key, key, value) for key, value in params.items() if not isinstance(value, dict)]


def flat(params):
    """The hyperparameters as a one level dictionary, for example {"alpha_LNDb": 100.0, "alpha_LIDC": 3.0}."""
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
    per_dataset  one model per dataset, and the average of their probabilities

    'params' holds the hyperparameters of the model. With per_dataset it holds one set per dataset, for example
    {"LNDb": {"alpha": 100.0}, "LIDC": {"alpha": 3.0}}: the two datasets differ in size and in label noise, so each
    model gets its own amount of regularisation.
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

        if self.strategy == "per_dataset":
            self.models_ = {}
            for dataset in frame["dataset"].unique():
                rows = (frame["dataset"] == dataset).to_numpy()
                X = frame.loc[rows, list(self.features)]
                rows_weight = None if weight is None else weight[rows]
                self.models_[dataset] = _fit_pipeline(make_model(self.name, self.params[dataset], self.final), X, y[rows], rows_weight)
        else:
            self.model_ = _fit_pipeline(make_model(self.name, self.params, self.final), self._matrix(frame), y, weight)
        return self

    def predict_proba(self, frame):
        if self.strategy == "per_dataset":
            X = frame[list(self.features)]
            each = [_full(model, model.predict_proba(X), self.classes_) for model in self.models_.values()]
            return np.mean(each, axis=0)
        return _full(self.model_, self.model_.predict_proba(self._matrix(frame)), self.classes_)


def _fit_pipeline(model, X, y, weight):
    """Fits a pipeline; the sample weights go to its last step, the classifier (or to an ensemble, which shares them)."""
    if weight is None:
        return model.fit(X, y)
    if not hasattr(model, "steps"):
        return model.fit(X, y, sample_weight=weight)
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


# Nested cross validation

def _train_label(training, label):
    """The column to learn from: the reader's own rating when training on readings, the nodule label otherwise."""
    if "reader_rating" in training and label == "rating":
        return "reader_rating"
    return label


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


def candidates_for(name, dataset, base=None):
    """The candidates of one dataset model. For an ensemble, every weight of its grid, with the hyperparameters that the
    inner loops of its models already chose for this outer fold ('base'); otherwise the points of the grid."""
    if name in ENSEMBLES:
        return [{"w": w, **base[dataset]} for w in GRIDS[name]["w"]]
    return grid_points(name)


def dataset_probabilities(name, features, fit_rows, held_out, train_label, weight, classes, base=None):
    """For every dataset, the probabilities that its own model gives to the held out nodules, one per candidate.

    For an ensemble, its two models are trained once and every weight only mixes their probabilities.
    """
    own = {}
    for dataset in sorted(fit_rows["dataset"].unique()):
        rows = fit_rows[fit_rows["dataset"] == dataset]
        rows_weight = None if weight is None else rows[weight]
        X, y = rows[list(features)], rows[train_label]
        if name in ENSEMBLES:
            each = []
            for model_name in ENSEMBLES[name]:
                model = _fit_pipeline(make_model(model_name, base[dataset][model_name], final=False), X, y, rows_weight)
                each.append(_full(model, model.predict_proba(held_out[list(features)]), classes))
            own[dataset] = [w * each[0] + (1 - w) * each[1] for w in GRIDS[name]["w"]]
            continue
        own[dataset] = []
        for params in grid_points(name):
            model = _fit_pipeline(make_model(name, params, final=False), X, y, rows_weight)
            own[dataset].append(_full(model, model.predict_proba(held_out[list(features)]), classes))
    return own


def inner_scores(name, strategy, features, inner_frame, training, train_label, weight, splits, classes, label, rules,
                 base=None):
    """Mean QWK of every (candidate, decision rule) on every inner split: one row per candidate, rule and split.

    A candidate is one set of hyperparameters, written as JSON. With per_dataset each dataset model has its own
    hyperparameters. The two models are trained independently of each other, so each one is trained once per
    candidate, and every pair (one for LNDb, one for LIDC IDRI) is scored by averaging their probabilities, as
    PooledModel does. The pairs are scored one at a time and never stored, which keeps the memory low.
    """
    candidates = grid_points(name) if name not in ENSEMBLES else None
    y = inner_frame[label].to_numpy()
    dataset = inner_frame["dataset"].to_numpy()
    rows = []
    for split, (inner_train, inner_test) in enumerate(splits):
        fit_rows = training[training["patient_id"].isin(inner_frame["patient_id"].iloc[inner_train])]
        held_out = inner_frame.iloc[inner_test]
        y_test, dataset_test = y[inner_test], dataset[inner_test]
        masks = [dataset_test == d for d in np.unique(dataset_test)]      # one per dataset, computed once per split

        def score(proba, rule):
            # mean_qwk of one candidate and rule, with the dataset masks of this split already computed
            prediction = decode(proba, classes, rule)
            return float(np.mean([quadratic_kappa(y_test[m], prediction[m]) for m in masks]))

        if strategy == "per_dataset":
            own = dataset_probabilities(name, features, fit_rows, held_out, train_label, weight, classes, base)
            datasets = list(own)
            options = {d: candidates_for(name, d, base) for d in datasets}
            for pair in product(*[range(len(options[d])) for d in datasets]):
                key = json.dumps({d: options[d][i] for d, i in zip(datasets, pair)})
                proba = np.mean([own[d][i] for d, i in zip(datasets, pair)], axis=0)
                for rule in rules:
                    rows.append((key, rule, split, score(proba, rule)))
        else:
            for params in candidates:
                model = _fit(name, params, strategy, features, fit_rows, train_label, weight, final=False)
                proba = model.predict_proba(held_out)
                for rule in rules:
                    rows.append((json.dumps(params), rule, split, score(proba, rule)))
    return pd.DataFrame(rows, columns=["params", "rule", "split", "score"])


def summarise_candidates(name, scores):
    """Mean score over the inner splits, its standard error and the complexity of every (candidate, rule)."""
    table = scores.groupby(["params", "rule"])["score"].agg(["mean", "std", "count"]).reset_index()
    table = table.rename(columns={"mean": "score"})
    table["se"] = table["std"] / np.sqrt(table["count"])
    table["complexity"] = [complexity(name, json.loads(key)) for key in table["params"]]
    return table[["params", "rule", "score", "se", "complexity"]]


def one_standard_error(table):
    """The one standard error rule (Hastie et al., 2009, section 7.10) on a table of candidates.

    'table' has one row per (candidate, decision rule) with its mean score, the standard error of that mean and the
    complexity of the candidate. Every candidate whose score is at least the best score minus its standard error is
    as good as the best within the noise of the inner folds; among them the simplest wins, and among equally simple
    ones the highest score.
    """
    best = table.loc[table["score"].idxmax()]
    good_enough = table[table["score"] >= best["score"] - best["se"]]
    return good_enough.sort_values(["complexity", "score"], ascending=[True, False]).iloc[0]


def _outer_job(job):
    """One outer fold: the inner loop chooses the hyperparameters and the rule, then the model is refitted and predicts."""
    name, strategy, features, frame, training, label, weight, train_patients, test_rows, seed, rules, base = job
    classes = np.sort(frame[label].unique())
    train_label = _train_label(training, label)

    # Inner loop, on the training part of this outer fold only: 3 folds grouped by patient, drawn with 2 seeds
    inner_frame = frame[frame["patient_id"].isin(train_patients)].reset_index(drop=True)
    splits = inner_splits(inner_frame, label, seed)
    scores = inner_scores(name, strategy, features, inner_frame, training, train_label, weight, splits, classes, label,
                          rules, base)
    table = summarise_candidates(name, scores)
    chosen = one_standard_error(table)
    params = json.loads(chosen["params"])

    # Refit on the whole training part with the chosen hyperparameters and predict the held out fold
    fit_rows = training[training["patient_id"].isin(train_patients)]
    model = _fit(name, params, strategy, features, fit_rows, train_label, weight)
    proba = model.predict_proba(frame.iloc[test_rows])
    return seed, test_rows, proba, params, chosen["rule"], table


def nested_cv(name, strategy, frame, features, label="rating", seeds=(0, 1, 2), folds=5, rules=RULES,
              training=None, weight=None, processes=10, base_params=None):
    """Nested cross validation grouped by patient, with the outer folds stratified by dataset and rating.

    'frame' holds the nodules that are evaluated. 'training' is the table the models learn from: the same nodules by
    default, or the readings. Returns the out of fold probabilities (seeds x nodules x ratings), the rating decoded with
    the rule chosen for each outer fold, the hyperparameters and rule chosen in each outer fold, and the inner scores
    of every candidate in every outer fold (to draw the validation curves).

    'base_params' is needed for an ensemble: {(seed, outer fold): {dataset: {model: hyperparameters}}}, the choices
    that the inner loops of its models made in each outer fold. Reusing them is fair, because they were made with the
    training part of that outer fold only.
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
            base = None if base_params is None else base_params[(seed, fold)]
            jobs.append((name, strategy, features, frame, training, label, weight, train_patients, test, seed, rules,
                         base))
            fold_of.append(fold)
    with Pool(min(processes, len(jobs))) as pool:
        results = pool.map(_outer_job, jobs)

    # Put every fold back in place
    proba = np.zeros((len(seeds), len(frame), len(classes)))
    decoded = np.zeros((len(seeds), len(frame)), dtype=int)
    choices, inner_scores = [], []
    for fold, (seed, test, fold_proba, params, rule, table) in zip(fold_of, results):
        s = list(seeds).index(seed)
        proba[s, test] = fold_proba
        decoded[s, test] = decode(fold_proba, classes, rule)
        choices.append({"seed": seed, "outer_fold": fold, "fold_size": len(test), "rule": rule,
                        "params": json.dumps(params), **flat(params)})
        inner_scores.append(table.assign(seed=seed, outer_fold=fold))
    return {"proba": proba, "decoded": decoded, "choices": pd.DataFrame(choices),
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
    fold_choices = choices[choices["seed"] == seed].sort_values("outer_fold").reset_index(drop=True)
    splitter = StratifiedGroupKFold(folds, shuffle=True, random_state=seed)
    folds_and_choices = zip(splitter.split(frame, strata, frame["patient_id"]), fold_choices.itertuples(index=False))
    rng = np.random.default_rng(seed)
    drops = {feature: [] for feature in features}

    for (train, test), choice in folds_and_choices:
        params = json.loads(choice.params)
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
