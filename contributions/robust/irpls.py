"""
IRPLS: Iteratively Reweighted PLS regression.

Refactored from the inline irpls class in robust_analysis.ipynb into a
reusable sklearn-compatible estimator. Based on the IRNPLS algorithm:
https://github.com/puneetmishra2/IRNPLS

Candidate for upstream contribution to chemotools.
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted


class IRPLSRegressor(BaseEstimator, RegressorMixin):
    """
    Iteratively Reweighted PLS (IRPLS) regressor.

    Robust to outliers in both X and y by iteratively downweighting samples
    whose residuals exceed a threshold based on Median Absolute Deviation (MAD).

    Parameters
    ----------
    n_components : int, default=2
        Number of PLS latent variables.
    alpha : float, default=4.685
        Tuning constant for the Tukey bisquare weight function.
        Larger values are more permissive (less robust). Typical range: 2–7.

    Attributes
    ----------
    beta_ : list of ndarray
        Regression coefficients per response, per component.
    weights_ : ndarray, shape (n_components, n_samples)
        Final sample weights per component.
    x_median_ : ndarray
        Median of X used for centering.
    y_median_ : ndarray
        Median of y used for centering.

    Examples
    --------
    >>> model = IRPLSRegressor(n_components=5, alpha=4.685)
    >>> model.fit(X_train, y_train)
    >>> y_pred = model.predict(X_test)
    """

    def __init__(self, n_components=2, alpha=4.685):
        self.n_components = n_components
        self.alpha = alpha

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if y.ndim == 1:
            y = y[:, None]
        elif y.shape[0] == 1:
            y = y.T

        n_samples, n_features = X.shape
        n_responses = y.shape[1]
        lvs = min(self.n_components, n_samples, n_features)

        self.x_median_ = np.median(X, axis=0)
        self.y_median_ = np.median(y, axis=0)
        X = X - self.x_median_
        Y = y - self.y_median_

        q = np.zeros((n_responses, lvs))
        T = np.zeros((n_samples, lvs))
        weights = np.zeros((lvs, n_samples))
        w = np.zeros((n_features, lvs))
        Pb = np.zeros((n_features, lvs))

        for i in range(lvs):
            D = np.eye(n_samples) / n_samples
            crit = 1.0
            while crit > 1e-5:
                vt = X.T @ D @ Y
                v = vt.copy()
                t_temp = X @ v / np.linalg.norm(X @ v)
                q_temp = (Y.T @ D) @ t_temp
                r = Y - np.outer(t_temp, q_temp)
                lo = 1 - t_temp ** 2
                r = r / np.sqrt(lo)
                mad_r = np.median(np.abs(r - np.median(r, axis=0)), axis=0)
                r = r * (0.6745 / self.alpha) / mad_r
                temp_D = (1 - r ** 2) ** 2
                temp_D[np.abs(r) >= 1] = 0
                temp_D = np.prod(temp_D, axis=1)
                crit = np.sum(np.abs(temp_D) - np.abs(np.diag(D)))
                D = np.diag(temp_D)

            weights[i, :] = np.diag(D)
            vt = X.T @ D @ Y
            w[:, i] = vt.flatten()
            t = X @ w[:, i]
            t /= np.linalg.norm(t)
            T[:, i] = t
            q[:, i] = (Y.T @ D) @ t
            Pb[:, i] = (X.T @ D) @ t
            X = X - np.outer(t, Pb[:, i])
            Y = Y - np.outer(t, q[:, i])
            w[:, i] /= np.linalg.norm(w[:, i])

        PtW = np.triu(Pb.T @ w)
        beta = []
        for i in range(n_responses):
            w_div_PtW = w @ np.linalg.inv(PtW)
            temp = w_div_PtW * q[i, :]
            temp_cumsum = np.cumsum(temp, axis=1)
            intercepts = self.y_median_[i] - self.x_median_ @ temp_cumsum
            beta_i_all = np.vstack([intercepts[np.newaxis, :], temp_cumsum])
            beta.append(beta_i_all)

        self.beta_ = beta
        self.weights_ = weights
        self.attributes_ = {"T": T, "Pb": Pb, "W": w, "PtW": PtW}
        self.n_components_ = lvs
        return self

    def predict(self, X, component=None):
        """
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
        component : int or None
            Which component index to use for prediction (0-based).
            Defaults to the last component (n_components - 1).

        Returns
        -------
        y_pred : ndarray, shape (n_samples,)
        """
        check_is_fitted(self, "beta_")
        if component is None:
            component = self.n_components_ - 1
        beta_k = self.beta_[0][:, component]
        return np.asarray(X, dtype=np.float64) @ beta_k[1:] + beta_k[0]
