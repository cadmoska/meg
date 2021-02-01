import numpy as np
import pandas as pd

from sklearn import linear_model
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import ClassifierMixin, RegressorMixin



def _get_coef(est):
    """Get coefficients from a fitted regression estimator."""
    if hasattr(est, 'steps'):
        return est.steps[-1][1].coef_
    return est.coef_


class SparseRegressor(BaseEstimator, ClassifierMixin, TransformerMixin):
    """ Provided regression estimator (ie model) solves inverse problem
        using data X and lead field L. The estimated coefficients (est_coef
        sometimes called z) are then used to predict which parcels are active.

        X must be of a specific structure with a column name 'subject' and
        'L_path' which gives the path to lead_field files for each subject
    """

    def __init__(self, model, n_jobs=1):
        self.model = model
        self.n_jobs = n_jobs
        self.parcel_indices = {}
        self.Ls = {}

    def fit(self, X, y):
        return self

    def predict(self, X):
        return (self.decision_function(X) > 0).astype(int)

    def _run_model(self, model, L, X):
        norms = np.linalg.norm(L, axis=0)
        L = L / norms[None, :]

        est_coefs = np.empty((X.shape[0], L.shape[1]))
        for idx in range(len(X)):
            x = X.iloc[idx].values
            model.fit(L, x)
            est_coef = np.abs(_get_coef(model))
            est_coef /= norms
            est_coefs[idx] = est_coef

        return est_coefs.T

    def decision_function(self, X):
        X = X.reset_index(drop=True)

        for subject_id in np.unique(X['subject']):
            if subject_id not in self.Ls:
                # load corresponding L if it's not already in
                L_used = X[X['subject'] == subject_id]['L_path'].iloc[0]
                lead_field = np.load(L_used)
                self.parcel_indices[subject_id] = lead_field['parcel_indices']

                # scale L to avoid tiny numbers
                self.Ls[subject_id] = 1e8 * lead_field['lead_field']
                assert (self.parcel_indices[subject_id].shape[0] ==
                        self.Ls[subject_id].shape[1])

        n_parcels = np.max([np.max(s) for s in self.parcel_indices.values()])
        betas = np.empty((len(X), n_parcels))
        for subj_idx in np.unique(X['subject']):
            L_used = self.Ls[subj_idx]

            X_used = X[X['subject'] == subj_idx]
            X_used = X_used.drop(['subject', 'L_path'], axis=1)

            est_coef = self._run_model(self.model, L_used, X_used)

            beta = pd.DataFrame(
                np.abs(est_coef)
            ).groupby(self.parcel_indices[subj_idx]).max().transpose()
            betas[X['subject'] == subj_idx] = np.array(beta)
        return betas


class CustomSparseEstimatorAd(BaseEstimator, RegressorMixin):
    """ Regression estimator which uses LassoLars with an adaptative Lasso configuration algorithm with given alpha
        normalized for each lead field L and x. """
    def __init__(self, alpha=0.01, max_iter = 1, n_lasso_iterations = 15):
        self.alpha = alpha
        self.max_iter = max_iter
        self.n_lasso_iterations = n_lasso_iterations

    def fit(self, L, x):
        alpha_max = abs(L.T.dot(x)).max() / len(L)
        alpha = self.alpha * alpha_max
        max_iter = self.max_iter
        n_lasso_iterations = self.n_lasso_iterations

      
        eps = 0.01
        g = lambda w: np.log(eps + np.abs(w))
        gprime = lambda w: 1. / (eps + np.abs(w))
        p_obj = lambda w: 1. / (2 * n_samples) * np.sum((x - np.dot(L, w)) ** 2) \
                  + alpha * np.sum(g(w))
        lasso = linear_model.LassoLars(alpha=alpha,
                                       normalize=False, fit_intercept=False, max_iter=max_iter)
        objective = []
        n_samples, n_features = L.shape
        weights = np.ones(n_features)
        for k in range(n_lasso_iterations):
            X_w = L / weights[np.newaxis, :]
            lasso.fit(X_w, x)
            coef_ = lasso.coef_ / weights
            weights = gprime(coef_)
            objective.append(p_obj(coef_)) 
        self.coef_ = coef_


def get_estimator():
    custom_model = CustomSparseEstimatorAd(alpha=0.4, max_iter=7)
    lasso_lars_alpha = SparseRegressor(custom_model)

    return lasso_lars_alpha
