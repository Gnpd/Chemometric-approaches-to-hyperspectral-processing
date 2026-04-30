"""
ROSA (Residual Orthogonalized Sequential Alternation) PLS regressor.

Refactored from helperfunctions/rosa_pls.py into a sklearn-compatible estimator.
Candidate for upstream contribution to chemotools.
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


class _WeightedPLS:
    """Internal single-block weighted PLS used by ROSARegressor."""

    def __init__(self, n_components, weights=None):
        self.n_components = n_components
        self.weights = weights

    def fit(self, X, Y):
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        if Y.ndim == 1:
            Y = Y[:, None]
        n, p = X.shape
        q = Y.shape[1]

        w = np.ones(n, dtype=np.float64) / n if self.weights is None else np.asarray(self.weights, dtype=np.float64)
        w = w / w.sum()

        self.x_mean_ = (w[:, None] * X).sum(axis=0)
        self.y_mean_ = (w[:, None] * Y).sum(axis=0)
        X = X - self.x_mean_
        Y = Y - self.y_mean_

        T = np.zeros((n, self.n_components))
        R = np.zeros((p, self.n_components))
        W = np.zeros((p, self.n_components))
        P = np.zeros((p, self.n_components))
        C = np.zeros((q, self.n_components))
        TT = np.zeros(self.n_components)

        Xd = w[:, None] * X
        XtY = Xd.T @ Y

        for a in range(self.n_components):
            wvec = XtY[:, 0] if q == 1 else XtY @ np.linalg.svd(XtY.T, full_matrices=False)[0][:, 0]
            wvec = wvec / np.linalg.norm(wvec)

            r = wvec.copy()
            for j in range(a):
                r -= (P[:, j] @ wvec) * R[:, j]

            t = X @ r
            tt = (w * t * t).sum()
            c = (XtY.T @ r) / tt
            p = (Xd.T @ t) / tt
            XtY -= (p[:, None] @ c[None]) * tt

            T[:, a], P[:, a], W[:, a], R[:, a], C[:, a], TT[a] = t, p, wvec, r, c, tt

        self.T_, self.P_, self.W_, self.R_, self.C_, self.TT_ = T, P, W, R, C, TT
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=np.float64) - self.x_mean_) @ self.R_

    def predict(self, X, n_components=None):
        nc = self.n_components if n_components is None else min(n_components, self.n_components)
        X = np.asarray(X, dtype=np.float64) - self.x_mean_
        B = self.R_[:, :nc] @ np.linalg.solve(self.P_[:, :nc].T @ self.R_[:, :nc], self.C_[:, :nc].T)
        return X @ B + self.y_mean_


class ROSARegressor(BaseEstimator, RegressorMixin):
    """
    ROSA-PLS: Residual Orthogonalized Sequential Alternation for multi-block PLS regression.

    Fits one PLS block at a time; each subsequent block is orthogonalized with respect to
    the scores of all previous blocks and fitted to the Y residual.

    Parameters
    ----------
    n_components : list of int
        Number of PLS components for each block. Must have the same length as X_blocks.

    Examples
    --------
    >>> rosa = ROSARegressor(n_components=[3, 2])
    >>> rosa.fit([X1, X2], y)
    >>> y_pred = rosa.predict([X1_new, X2_new])
    """

    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X_blocks, y):
        """
        Parameters
        ----------
        X_blocks : list of ndarray, shape [(n, p1), (n, p2), ...]
        y : ndarray, shape (n,) or (n, q)
        """
        if self.n_components is None:
            raise ValueError("n_components must be a list of ints, one per block.")

        X_blocks = [np.asarray(X, dtype=np.float64) for X in X_blocks]
        y = np.asarray(y, dtype=np.float64)
        if y.ndim == 1:
            y = y[:, None]

        n_blocks = len(X_blocks)
        if len(self.n_components) != n_blocks:
            raise ValueError("len(n_components) must equal len(X_blocks).")

        Y_res = y.copy()
        X_res = [X.copy() for X in X_blocks]
        self.block_models_ = []

        for i, (X, nc) in enumerate(zip(X_res, self.n_components)):
            pls = _WeightedPLS(n_components=nc).fit(X, Y_res)
            Y_pred = pls.predict(X, nc)
            self.block_models_.append(pls)
            Y_res = Y_res - Y_pred

            # Orthogonalize remaining blocks w.r.t. current block scores
            T = pls.T_[:, :nc]
            for j in range(i + 1, n_blocks):
                P = np.linalg.pinv(T) @ X_res[j]
                X_res[j] = X_res[j] - T @ P

        self.n_blocks_ = n_blocks
        self.n_features_in_ = sum(X.shape[1] for X in X_blocks)
        return self

    def predict(self, X_blocks):
        """
        Parameters
        ----------
        X_blocks : list of ndarray, shape [(m, p1), (m, p2), ...]

        Returns
        -------
        y_pred : ndarray, shape (m,) or (m, q)
        """
        check_is_fitted(self, "block_models_")
        X_res = [np.asarray(X, dtype=np.float64).copy() for X in X_blocks]
        n_targets = self.block_models_[0].y_mean_.shape[0]
        y_pred = np.zeros((X_res[0].shape[0], n_targets))

        for i, (pls, nc) in enumerate(zip(self.block_models_, self.n_components)):
            y_pred += pls.predict(X_res[i], nc)
            if i < self.n_blocks_ - 1:
                T = pls.transform(X_res[i])[:, :nc]
                for j in range(i + 1, self.n_blocks_):
                    P = np.linalg.pinv(T) @ X_res[j]
                    X_res[j] -= T @ P

        return y_pred.ravel() if n_targets == 1 else y_pred


def rosa_cross_val_rmsep(X_blocks, y, n_components_candidates, n_splits=5, random_state=42):
    """
    Select the best n_components configuration for ROSARegressor via k-fold CV.

    Parameters
    ----------
    X_blocks : list of ndarray
    y : ndarray
    n_components_candidates : list of list of int
        Each element is a candidate n_components list to evaluate.
    n_splits : int
    random_state : int

    Returns
    -------
    best_n_components : list of int
    best_rmsep : float
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    best_config, best_rmsep = None, np.inf

    for nc_list in n_components_candidates:
        errors = []
        for train_idx, val_idx in kf.split(X_blocks[0]):
            X_tr = [X[train_idx] for X in X_blocks]
            X_val = [X[val_idx] for X in X_blocks]
            y_tr = y[train_idx]
            y_val = y[val_idx]

            model = ROSARegressor(n_components=nc_list).fit(X_tr, y_tr)
            errors.append(np.sqrt(mean_squared_error(y_val, model.predict(X_val))))

        rmsep = np.mean(errors)
        if rmsep < best_rmsep:
            best_rmsep = rmsep
            best_config = nc_list

    return best_config, best_rmsep
