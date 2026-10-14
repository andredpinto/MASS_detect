"""Unit tests of the models, decision rules, pooling and nested cross validation of notebook 06."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from src.mass_detect.scripts.modelling import FrankHallOrdinal, PooledModel, decode, mean_qwk, nested_cv

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
        proba = PooledModel("ordinal", {"alpha": 1.0}, strategy, ["f1", "f2", "f3"]).fit(frame, frame["rating"]).predict_proba(frame)
        assert proba.shape == (len(frame), 5) and np.allclose(proba.sum(axis=1), 1), strategy

def test_nested_cv_never_splits_a_patient_and_predicts_every_nodule():
    frame = synthetic_table()
    strata = frame["dataset"] + "_" + frame["rating"].astype(str)
    for train, test in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(frame, strata, frame["patient_id"]):
        assert not set(frame.patient_id.iloc[train]) & set(frame.patient_id.iloc[test])
    result = nested_cv("ordinal", "pooled", frame, ["f1", "f2", "f3"], seeds=(0,), processes=2)
    assert result["decoded"].shape == (1, len(frame)) and set(np.unique(result["decoded"])) <= set(range(1, 6))
    assert mean_qwk(frame["rating"].to_numpy(), result["decoded"][0], frame["dataset"].to_numpy()) > 0.3
