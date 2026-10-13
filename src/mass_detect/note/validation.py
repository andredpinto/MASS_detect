"""Competition metrics, grouped cross validation and the submission check."""
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedGroupKFold
import numpy as np
import pandas as pd

CLASSES = np.arange(1, 6)

def competition_metrics(y_true, y_pred): # The four leaderboard metrics.
    return {"balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro"),
            "qwk": cohen_kappa_score(y_true, y_pred, weights="quadratic"),
            "mae": mean_absolute_error(y_true, y_pred)}

def all_metrics(y_true, y_pred): # Competition metrics plus the extra ones described above.
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    scores = competition_metrics(y_true, y_pred)
    scores["composite"] = scores["balanced_accuracy"] + scores["macro_f1"] + scores["qwk"] - scores["mae"]
    scores["within_one"] = float(np.mean(np.abs(y_true - y_pred) <= 1))
    high = y_true >= 4
    scores["sensitivity_4_5"] = float(np.mean(y_pred[high] >= 4)) if high.any() else np.nan
    for k in CLASSES:
        scores[f"recall_{k}"] = float(np.mean(y_pred[y_true == k] == k)) if (y_true == k).any() else np.nan
    return scores

def grouped_cv_predictions(make_model, X, y, groups, repeats=5, folds=5, evaluate_mask=None, first_seed=0): #Out of fold predictions, one array per repeat, with folds that keep every CT on one side.
    """
    make_model() must return a fresh, unfitted model. first_seed changes the folds (to confirm a choice on
    splits that were not used to make it). evaluate_mask marks the rows that may be used for
    validation (for example only LNDb rows when LIDC is extra training data): the other rows are always in
    the training part and get no prediction (they stay 0).
    """
    y, groups = np.asarray(y), np.asarray(groups)
    evaluate = np.ones(len(y), bool) if evaluate_mask is None else np.asarray(evaluate_mask)
    candidates = np.flatnonzero(evaluate)
    always_train = np.flatnonzero(~evaluate)
    predictions = []
    for seed in range(first_seed, first_seed + repeats):
        predicted = np.zeros(len(y), dtype=int)
        splitter = StratifiedGroupKFold(folds, shuffle=True, random_state=seed)
        for train, test in splitter.split(candidates, y[candidates], groups[candidates]):
            train_rows = np.r_[candidates[train], always_train]
            model = make_model().fit(X.iloc[train_rows], y[train_rows])
            predicted[candidates[test]] = model.predict(X.iloc[candidates[test]])
        predictions.append(predicted)
    return predictions

def summarise(y, groups, predictions, n_boot=500, seed=0, keys=("balanced_accuracy", "macro_f1", "qwk", "mae", "composite")):
    # Mean over repeats and a 95% interval obtained by resampling whole CTs.
    y, groups = np.asarray(y), np.asarray(groups)
    mean = pd.DataFrame([all_metrics(y, p) for p in predictions]).mean()
    rows_of = pd.Series(np.arange(len(y))).groupby(groups).apply(np.array).values
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        rows = np.concatenate(rows_of[rng.integers(0, len(rows_of), len(rows_of))])
        boots.append(pd.DataFrame([all_metrics(y[rows], p[rows]) for p in predictions[:2]]).mean())
    low, high = pd.DataFrame(boots)[list(keys)].quantile([0.025, 0.975]).values
    return pd.DataFrame({"mean": mean[list(keys)], "ci_low": low, "ci_high": high}).round(3)

def paired_difference(y, groups, predictions_a, predictions_b, n_boot=1000, seed=0):
    # Composite of B minus composite of A, with a 95% interval from the same resampled CTs (paired).
    y, groups = np.asarray(y), np.asarray(groups)
    rows_of = pd.Series(np.arange(len(y))).groupby(groups).apply(np.array).values
    rng = np.random.default_rng(seed)
    score = lambda preds, rows: np.mean([all_metrics(y[rows], p[rows])["composite"] for p in preds[:2]])
    differences = []
    for _ in range(n_boot):
        rows = np.concatenate(rows_of[rng.integers(0, len(rows_of), len(rows_of))])
        differences.append(score(predictions_b, rows) - score(predictions_a, rows))
    full = np.arange(len(y))
    return {"difference": round(score(predictions_b, full) - score(predictions_a, full), 3),
            "ci_low": round(np.quantile(differences, 0.025), 3), "ci_high": round(np.quantile(differences, 0.975), 3)}

def check_submission(submission, test_nodules): # Stops with a clear message if the file cannot be scored as intended. Run it before every upload.
    assert len(submission) == len(test_nodules), "number of rows differs from testNodules.csv"
    keys = ["LNDbID", "FindingID"]
    assert (submission[keys].to_numpy() == test_nodules[keys].to_numpy()).all(), "rows are not in the order of testNodules.csv"
    assert submission["prediction"].notna().all(), "missing predictions"
    assert submission["prediction"].dtype.kind == "i", "predictions must be integers"
    assert submission["prediction"].between(1, 5).all(), "predictions must be between 1 and 5"
    return True
