"""Ordinal logistic regression, size model, the malignancy model, ComBat harmonisation."""
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import TransformerMixin
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

class OrdinalLogisticRegression(BaseEstimator, ClassifierMixin): #Proportional odds model with L2 penalty. Use it after StandardScaler like any penalised model.

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def _unpack(self, params, n_features):
        weights = params[:n_features]
        cuts = np.cumsum(np.r_[params[n_features], np.exp(params[n_features + 1:])])   # always increasing
        return weights, cuts

    def _cumulative(self, X, weights, cuts): # P(rating <= k) for every row and every class, with 0 before the first class and 1 after the last.
        below = expit(cuts[None, :] - (X @ weights)[:, None])
        return np.hstack([np.zeros((len(X), 1)), below, np.ones((len(X), 1))])

    def fit(self, X, y, sample_weight=None):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        self.classes_ = np.unique(y)
        index = np.searchsorted(self.classes_, y)            # class position 0..K-1
        weight = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        n_features, rows = X.shape[1], np.arange(len(y))
        n_cuts = len(self.classes_) - 1

        def loss_and_gradient(params):# Penalised negative log likelihood and its exact gradient (the optimiser converges reliably with it).
            weights, cuts = self._unpack(params, n_features)
            cumulative = self._cumulative(X, weights, cuts)
            likelihood = np.clip(cumulative[rows, index + 1] - cumulative[rows, index], 1e-12, None)
            loss = -(weight * np.log(likelihood)).sum() + 0.5 * self.alpha * weights @ weights

            # The derivative of sigmoid(z) is s(1 - s). Upper and lower are the two cut points around each rating.
            density = cumulative[:, 1:-1] * (1 - cumulative[:, 1:-1])
            density = np.hstack([np.zeros((len(X), 1)), density, np.zeros((len(X), 1))])
            upper, lower = density[rows, index + 1] / likelihood, density[rows, index] / likelihood
            grad_weights = X.T @ (weight * (upper - lower)) + self.alpha * weights
            grad_cuts = (np.bincount(index, -weight * upper, minlength=n_cuts + 1)[:n_cuts]
                         + np.bincount(index, weight * lower, minlength=n_cuts + 1)[1:])
            # Chain rule for cuts = first + cumulative sum of exp(steps)
            steps = np.exp(params[n_features + 1:])
            grad_steps = steps * np.cumsum(grad_cuts[::-1])[::-1][1:]
            return loss, np.r_[grad_weights, grad_cuts.sum(), grad_steps]

        # Start: all weights at zero and cut points that reproduce the class frequencies of the training data
        share_below = np.cumsum(np.bincount(index, minlength=len(self.classes_)))[:-1] / len(y)
        start_cuts = np.log(share_below / (1 - share_below))
        start = np.r_[np.zeros(n_features), start_cuts[0], np.log(np.maximum(np.diff(start_cuts), 1e-3))]
        result = minimize(loss_and_gradient, start, jac=True, method="L-BFGS-B", options={"maxiter": 2000})
        self.coef_, self.cuts_ = self._unpack(result.x, n_features)
        self.converged_ = bool(result.success)
        return self

    def predict_proba(self, X):
        cumulative = self._cumulative(np.asarray(X, dtype=float), self.coef_, self.cuts_)
        return np.diff(cumulative, axis=1)

    def predict(self, X): # Most probable class (argmax).
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def predict_median(self, X): # Median of the predicted distribution: the first class whose cumulative probability reaches 0.5.
        return self.classes_[(self.predict_proba(X).cumsum(axis=1) >= 0.5).argmax(axis=1)]

SMALL_NODULE_DIAMETER_MM = 2.5   # nodules < 3 mm have no segmentation: we use a value below the 3 mm limit

def size_target(diameter_mm): # Training target of the size model: log of the radiologists' equivalent diameter.
    # The log makes the error relative (1 mm matters more for a 4 mm nodule than for a 25 mm one).
    return np.log(diameter_mm.fillna(SMALL_NODULE_DIAMETER_MM))

def make_size_model(): # Shallow gradient boosting: handles missing values and non linear relations, few settings to choose.
    return HistGradientBoostingRegressor(max_leaf_nodes=8, learning_rate=0.03, max_iter=400, random_state=0)

class NoduleMalignancyModel(BaseEstimator, ClassifierMixin): #Ordinal logistic regression on 'features; optional predicted size learned from 'size_inputs'.
    #The df given to fit() must also contain 'size_target_column' (log diameter of radiologists) and 'group_column' (CT or patient) - not needed at predict time.

    def __init__(self, features, size_inputs=None, size_target_column="size_target", group_column="group",
                 alpha=1.0, decision="argmax", prior_power=0.0, size_only_column=None):
        self.features = features
        self.size_inputs = size_inputs
        self.size_target_column = size_target_column
        self.group_column = group_column
        self.alpha = alpha
        self.decision = decision
        self.prior_power = prior_power
        self.size_only_column = size_only_column   # rows with 1 here train the size model only (e.g. LIDC)

    def _inputs(self, X, size):
        columns = X[self.features].to_numpy(dtype=float)
        return columns if size is None else np.column_stack([columns, size])

    def fit(self, X, y, sample_weight=None):
        size = None
        if self.size_inputs:
            inputs, target = X[self.size_inputs].to_numpy(dtype=float), X[self.size_target_column].to_numpy()
            size = np.zeros(len(X))
            for train, test in GroupKFold(5).split(inputs, target, X[self.group_column]):
                size[test] = make_size_model().fit(inputs[train], target[train]).predict(inputs[test])
            self.size_model_ = make_size_model().fit(inputs, target)
        # Rows marked in size_only_column helped the size model; the ordinal model learns only from the others
        keep = np.ones(len(X), bool) if self.size_only_column is None else X[self.size_only_column].to_numpy() == 0
        y, size = np.asarray(y)[keep], None if size is None else size[keep]
        weight = None if sample_weight is None else np.asarray(sample_weight)[keep]
        self.model_ = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                    OrdinalLogisticRegression(alpha=self.alpha))
        self.model_.fit(self._inputs(X[keep], size), y, ordinallogisticregression__sample_weight=weight)
        self.classes_ = self.model_.classes_
        self.prior_ = np.bincount(np.searchsorted(self.classes_, y), minlength=len(self.classes_)) / len(y)
        return self

    def predicted_size(self, X): # Predicted equivalent diameter in mm (useful to inspect the size model).
        return np.exp(self.size_model_.predict(X[self.size_inputs].to_numpy(dtype=float)))

    def predict_proba(self, X):
        size = self.size_model_.predict(X[self.size_inputs].to_numpy(dtype=float)) if self.size_inputs else None
        return self.model_.predict_proba(self._inputs(X, size))

    def predict(self, X): # Most probable class (or median), after an optional correction for the class frequencies.
        #Dividing probabilities by the training share of each class, raised to prior_power, moves them towards the rare classes (Saerens et al., 2002). 
        # With prior_power = 1 the decision treats all classes as equally frequent, which favours balanced accuracy and macro F1; 
        # with 0 it is the plain argmax, which favours MAE. An intermediate value balances the four competition metrics.
        probabilities = self.predict_proba(X) / self.prior_ ** self.prior_power
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
        if self.decision == "median": return self.classes_[(probabilities.cumsum(axis=1) >= 0.5).argmax(axis=1)]
        return self.classes_[probabilities.argmax(axis=1)]

class RidgeWithTunedCuts(BaseEstimator, ClassifierMixin): # First model (v1, the 1st submission), kept as a baseline for fair comparisons on the same folds.
    # Ridge regression predicts a score, cut into classes 1..5. 
    # The four cut points maximise 0.5 * balanced accuracy + 0.5 * QWK on out of fold scores of the training data (grouped by CT).
    # It raises balanced accuracy, but it pushes predictions away from the centre and costs MAE.

    def __init__(self, features, group_column="group", alpha=10.0):
        self.features = features
        self.group_column = group_column
        self.alpha = alpha

    def _ridge(self):
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(self.alpha))

    @staticmethod
    def _to_class(score, cuts):
        return 1 + np.searchsorted(np.sort(cuts), score)

    def fit(self, X, y, sample_weight=None):
        y = np.asarray(y)
        inputs = X[self.features]
        score = np.zeros(len(y))
        for train, test in GroupKFold(4).split(inputs, y, X[self.group_column]):
            score[test] = self._ridge().fit(inputs.iloc[train], y[train]).predict(inputs.iloc[test])

        def objective(cuts):
            predicted = self._to_class(score, cuts)
            return 0.5 * balanced_accuracy_score(y, predicted) + 0.5 * cohen_kappa_score(y, predicted, weights="quadratic")

        cuts = np.array([1.5, 2.5, 3.5, 4.5])
        best = objective(cuts)
        for _ in range(3):                                # a few passes of coordinate search
            for i in range(4):
                for value in np.linspace(cuts[i] - 0.6, cuts[i] + 0.6, 25):
                    trial = cuts.copy()
                    trial[i] = value
                    if objective(trial) > best:
                        best, cuts = objective(trial), np.sort(trial)
        self.cuts_ = cuts
        self.model_ = self._ridge().fit(inputs, y)
        self.classes_ = np.arange(1, 6)
        return self

    def predict(self, X):
        return self._to_class(self.model_.predict(X[self.features]), self.cuts_)

class CorrelationFilter(BaseEstimator, TransformerMixin): # Drops a feature when its absolute Spearman correlation with another feature is above 'threshold'.
    # Radiomic features are highly redundant, and with a few hundred nodules redundant inputs only add variance. 
    # Fitted on the training part of each fold only.

    def __init__(self, threshold=0.9):
        self.threshold = threshold

    def fit(self, X, y=None):
        correlation = np.abs(pd.DataFrame(np.asarray(X, dtype=float)).rank().corr().to_numpy())
        keep = []
        for j in range(correlation.shape[0]):
            if all(correlation[j, k] <= self.threshold for k in keep):
                keep.append(j)
        self.keep_ = keep
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.keep_]

def batch_of(dataset, manufacturer): # LNDb is one batch (the reference); LIDC is split by scanner manufacturer.
    return np.where(dataset == "LNDb", "LNDb", "LIDC_" + pd.Series(manufacturer).fillna("unknown").astype(str).to_numpy())

class ReferenceBatchComBat: # Moves every batch onto the reference batch, feature by feature, keeping the effect of a covariate.
    def __init__(self, columns, reference="LNDb"):
        self.columns = columns
        self.reference = reference

    def fit(self, X, batch, covariate):
        batch, covariate = np.asarray(batch), np.asarray(covariate, dtype=float)
        self.batches_ = [b for b in np.unique(batch) if b != self.reference]
        self.params_ = {}
        for column in self.columns:
            values = X[column].to_numpy(dtype=float)
            ok = ~np.isnan(values)
            # Least squares: value ~ intercept + slope * covariate + one shift per non reference batch
            design = np.column_stack([np.ones(ok.sum()), covariate[ok]] + [(batch[ok] == b).astype(float) for b in self.batches_])
            coef = np.linalg.lstsq(design, values[ok], rcond=None)[0]
            residual = values[ok] - design @ coef
            spread = {b: residual[batch[ok] == b].std() + 1e-9 for b in [self.reference] + self.batches_}
            self.params_[column] = {"intercept": coef[0], "slope": coef[1],
                                    "shift": dict(zip(self.batches_, coef[2:])), "spread": spread}
        return self

    def transform(self, X, batch, covariate):
        X, batch, covariate = X.copy(), np.asarray(batch), np.asarray(covariate, dtype=float)
        for column, p in self.params_.items():
            values = X[column].to_numpy(dtype=float, copy=True)
            for b in self.batches_:
                rows = batch == b
                expected = p["intercept"] + p["slope"] * covariate[rows]          # what the reference batch would show
                residual = values[rows] - expected - p["shift"][b]
                values[rows] = expected + residual * p["spread"][self.reference] / p["spread"][b]
            X[column] = values
        return X

class HarmonisedModel(BaseEstimator, ClassifierMixin): # Fits ComBat on the training rows, harmonises the non reference rows, then fits the model.
    # Predictions are only made for reference rows (LNDb), which ComBat leaves unchanged.
    def __init__(self, model, columns, batch_column="batch", reference="LNDb"):
        self.model = model
        self.columns = columns
        self.batch_column = batch_column
        self.reference = reference

    def fit(self, X, y, sample_weight=None):
        combat = ReferenceBatchComBat(self.columns, self.reference).fit(X, X[self.batch_column], y)
        self.model_ = clone(self.model).fit(combat.transform(X, X[self.batch_column], y), y, sample_weight=sample_weight)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        assert (X[self.batch_column] == self.reference).all(), "only reference batch rows can be predicted"
        return self.model_.predict_proba(X)

    def predict(self, X):
        assert (X[self.batch_column] == self.reference).all(), "only reference batch rows can be predicted"
        return self.model_.predict(X)
