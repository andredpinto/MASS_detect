"""Unit tests of the models, decision rules, pooling and nested cross validation of notebook 06."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
import json

from sklearn.metrics import cohen_kappa_score

from src.mass_detect.scripts.modelling import (GRIDS, SAMPLED, ClassWeighted, FrankHallOrdinal, PooledModel,
                                                dataset_probabilities, decode, edge_share, grid_points, inner_splits,
                                                make_model, mean_qwk, nested_cv, one_standard_error, quadratic_kappa)
from src.mass_detect.scripts.models import OrdinalLogisticRegression

def synthetic_table(n=400, seed=0): # Two datasets, patients with several nodules, a rating driven by two features
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    latent = X @ np.array([1.5, -1.0, 0.0]) + rng.logistic(size=n)
    frame = pd.DataFrame(X, columns=["f1", "f2", "f3"])
    frame["rating"] = 1 + np.searchsorted([-2.0, -0.5, 0.5, 2.0], latent)
    frame["dataset"] = np.where(np.arange(n) < n // 4, "LNDb", "LIDC")
    frame["patient_id"] = [f"p{i // 3}" for i in range(n)]
    frame["roi_diameter_mm"] = rng.uniform(3, 20, n)
    frame.loc[frame.dataset == "LNDb", "f1"] += 2.0          # a scanner shift in LNDb
    return frame

def test_frank_hall_gives_valid_ordered_probabilities():
    frame = synthetic_table()
    model = FrankHallOrdinal(LogisticRegression()).fit(frame[["f1", "f2", "f3"]], frame["rating"])
    proba = model.predict_proba(frame[["f1", "f2", "f3"]])
    assert proba.shape == (len(frame), 5) and np.all(proba >= 0) and np.allclose(proba.sum(axis=1), 1)
    expected = proba @ np.arange(1, 6)
    assert np.corrcoef(expected, frame["f1"] - frame["f2"])[0, 1] > 0.5     # higher latent value, higher expected rating

def test_decision_rules_on_a_known_distribution():
    proba = np.array([[0.35, 0.10, 0.15, 0.20, 0.20]])
    classes = np.arange(1, 6)
    assert decode(proba, classes, "argmax")[0] == 1          # the most probable rating
    assert decode(proba, classes, "median")[0] == 3          # cumulative 0.35, 0.45, 0.60: the median is 3
    assert decode(proba, classes, "mean")[0] == 3            # expected rating 2.80, rounded half up

def test_combat_leaves_lidc_untouched():
    frame = synthetic_table()
    model = PooledModel("ordinal", {"alpha": 1.0}, "combat", ["f1", "f2", "f3"]).fit(frame, frame["rating"])
    X = model._matrix(frame)
    lidc = (frame.dataset == "LIDC").to_numpy()
    assert np.allclose(X.loc[lidc, "f1"], frame.loc[lidc, "f1"])                       # the reference is not changed
    assert abs(X.loc[~lidc, "f1"].mean() - frame.loc[lidc, "f1"].mean()) < 0.5         # the shift of LNDb is removed

def test_every_strategy_predicts_every_rating():
    frame = synthetic_table()
    for strategy in ("pooled", "indicator", "combat", "per_dataset"):
        params = {"LNDb": {"alpha": 1.0}, "LIDC": {"alpha": 1.0}} if strategy == "per_dataset" else {"alpha": 1.0}
        proba = PooledModel("ordinal", params, strategy, ["f1", "f2", "f3"]).fit(frame, frame["rating"]).predict_proba(frame)
        assert proba.shape == (len(frame), 5) and np.allclose(proba.sum(axis=1), 1), strategy

def test_nested_cv_never_splits_a_patient_and_predicts_every_nodule():
    frame = synthetic_table()
    strata = frame["dataset"] + "_" + frame["rating"].astype(str)
    for train, test in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(frame, strata, frame["patient_id"]):
        assert not set(frame.patient_id.iloc[train]) & set(frame.patient_id.iloc[test])
    result = nested_cv("ordinal", "pooled", frame, ["f1", "f2", "f3"], seeds=(0,), processes=2)
    assert result["decoded"].shape == (1, len(frame)) and set(np.unique(result["decoded"])) <= set(range(1, 6))
    assert mean_qwk(frame["rating"].to_numpy(), result["decoded"][0], frame["dataset"].to_numpy()) > 0.3


def test_one_standard_error_rule_prefers_the_simplest_candidate_within_the_noise():
    table = pd.DataFrame({"params": ["complex", "simple", "simplest"], "rule": "argmax",
                          "score": [0.60, 0.59, 0.55], "se": [0.02, 0.02, 0.02], "complexity": [5, 1, 0]})
    assert one_standard_error(table)["params"] == "simple"      # 0.59 is within 0.02 of 0.60, 0.55 is not


def test_per_dataset_pairs_match_the_pooled_model():
    # In the inner loop each dataset model is trained once per candidate and the pairs are averaged afterwards;
    # this must give exactly the probabilities of a PooledModel trained with that pair of hyperparameters
    frame = synthetic_table()
    features, classes = ["f1", "f2", "f3"], np.arange(1, 6)
    train, test = inner_splits(frame, "rating", seed=0, inner_folds=3, inner_repeats=1)[0]
    own = dataset_probabilities("ordinal", features, frame.iloc[train], frame.iloc[test], "rating", None, classes)
    candidates = grid_points("ordinal")
    pair = (np.mean([own["LIDC"][candidates.index({"alpha": 3.0})], own["LNDb"][candidates.index({"alpha": 300.0})]], axis=0))
    params = {"LIDC": {"alpha": 3.0}, "LNDb": {"alpha": 300.0}}
    model = PooledModel("ordinal", params, "per_dataset", features).fit(frame.iloc[train], frame["rating"].iloc[train])
    assert np.allclose(pair, model.predict_proba(frame.iloc[test]))


def test_fast_kappa_equals_sklearn():
    rng = np.random.default_rng(0)
    for _ in range(50):
        y, prediction = rng.integers(1, 6, 80), rng.integers(1, 6, 80)
        prediction[prediction == 4] = 3          # a rating that is never predicted, as happens with rare classes
        assert np.isclose(quadratic_kappa(y, prediction), cohen_kappa_score(y, prediction, weights="quadratic"))


def test_random_search_draws_the_same_candidates_from_the_grid():
    for name, size in SAMPLED.items():
        first, second = grid_points(name), grid_points(name)
        assert first == second and len(first) == size
        assert all(value in GRIDS[name][key] for params in first for key, value in params.items())


def test_edge_share_flags_values_at_the_ends_of_the_grid():
    choices = pd.DataFrame({"params": [json.dumps({"alpha": 3000.0}), json.dumps({"alpha": 10.0}),
                                       json.dumps({"alpha": 0.1}), json.dumps({"alpha": 10.0})]})
    assert edge_share("ordinal", choices)["alpha"] == 0.5     # 3000 and 0.1 are the two ends of the grid


def test_class_weights_give_every_rating_the_same_total_weight():
    class Recorder(OrdinalLogisticRegression):     # remembers the weights it was trained with
        def fit(self, X, y, sample_weight=None):
            self.seen_weight_ = sample_weight
            return super().fit(X, y, sample_weight)
    frame = synthetic_table()
    y = frame["rating"].to_numpy()
    model = ClassWeighted(Recorder(alpha=1.0), power=1.0).fit(frame[["f1", "f2", "f3"]], y)
    totals = pd.Series(model.model_.seen_weight_).groupby(y).sum()
    assert np.allclose(totals, totals.iloc[0])                              # power 1: every rating weighs the same
    plain = ClassWeighted(Recorder(alpha=1.0), power=0.0).fit(frame[["f1", "f2", "f3"]], y)
    assert np.allclose(plain.model_.seen_weight_, 1.0)                      # power 0: weights unchanged


def test_weighted_ensemble_mixes_its_models():
    frame = synthetic_table()
    X, y = frame[["f1", "f2", "f3"]], frame["rating"]
    base = {"ordinal": {"alpha": 1.0},
            "random_forest": {"max_features": 0.6, "min_samples_leaf": 5, "max_depth": 8, "class_weight": None}}
    alone = make_model("ordinal", base["ordinal"]).fit(X, y).predict_proba(X)
    forest = make_model("random_forest", base["random_forest"], final=False).fit(X, y).predict_proba(X)
    only_ordinal = make_model("ordinal_rf", {"w": 1.0, **base}, final=False).fit(X, y).predict_proba(X)
    half = make_model("ordinal_rf", {"w": 0.5, **base}, final=False).fit(X, y).predict_proba(X)
    assert np.allclose(only_ordinal, alone)                                 # weight 1: the ordinal model alone
    assert np.allclose(half, 0.5 * alone + 0.5 * forest)                    # weight 0.5: the plain average


def test_elastic_net_sets_useless_features_to_zero():
    frame = synthetic_table()
    model = make_model("elastic_net", {"C": 0.03, "l1_ratio": 1.0}).fit(frame[["f1", "f2", "f3"]], frame["rating"])
    coefficients = np.array([binary.coef_[0] for binary in model[-1].models_])
    assert np.all(coefficients[:, 2] == 0)                                  # f3 carries no signal in synthetic_table
