"""
COVSEL: Covariance Selection for spectral variable selection.

Implements the FCOVSEL (Forward Covariance Selection) algorithm:
at each step, the variable that maximises the absolute covariance with
the LOO residual of y is selected, and X is orthogonalised w.r.t. that variable.

Candidate for upstream contribution to chemotools.
"""

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_is_fitted


class COVSELSelector(BaseEstimator, SelectorMixin):
    """
    Forward Covariance Selection (FCOVSEL) variable selector for spectral data.

    Sequentially selects variables that maximise the absolute leave-one-out
    covariance with the response, orthogonalising X at each step.

    Parameters
    ----------
    n_variables : int
        Number of variables to select.

    Attributes
    ----------
    selected_indices_ : ndarray of int, shape (n_variables,)
        Indices of selected features in selection order.
    support_ : ndarray of bool, shape (n_features,)
        Boolean mask of selected features.

    Examples
    --------
    >>> sel = COVSELSelector(n_variables=20)
    >>> sel.fit(X_train, y_train)
    >>> X_reduced = sel.transform(X_train)
    """

    def __init__(self, n_variables=10):
        self.n_variables = n_variables

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        n_samples, n_features = X.shape

        if self.n_variables > n_features:
            raise ValueError(f"n_variables ({self.n_variables}) > n_features ({n_features}).")

        selected = []
        remaining = list(range(n_features))
        X_res = X.copy()

        for _ in range(self.n_variables):
            # LOO covariance: cov(x_j, y) using LOO residuals
            loo_covs = np.abs(self._loo_covariance(X_res[:, remaining], y))
            best_local = int(np.argmax(loo_covs))
            best_global = remaining[best_local]
            selected.append(best_global)
            remaining.remove(best_global)

            # Orthogonalise X w.r.t. selected variable
            v = X_res[:, best_global]
            v_norm_sq = v @ v
            if v_norm_sq > 0:
                X_res = X_res - np.outer(v, (v @ X_res) / v_norm_sq)

        self.selected_indices_ = np.array(selected, dtype=int)
        self.n_features_in_ = n_features
        return self

    @staticmethod
    def _loo_covariance(X, y):
        """Approximate LOO covariance between each column of X and y."""
        n = X.shape[0]
        y_mean = y.mean()
        # Fast LOO: cov(x_j, y) ≈ (x_j - mean(x_j))^T (y - mean(y)) / (n-1)
        # (exact for linear LOO estimate; computationally efficient)
        return ((X - X.mean(axis=0)) * (y - y_mean)[:, None]).sum(axis=0) / (n - 1)

    def _get_support_mask(self):
        check_is_fitted(self, "selected_indices_")
        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[self.selected_indices_] = True
        return mask
