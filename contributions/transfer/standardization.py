"""
Direct Standardization (DS) and Piecewise Direct Standardization (PDS)
for spectral model transfer between instruments.

These sklearn-compatible transformers learn a mapping from a slave instrument
to a master instrument using a set of transfer samples measured on both.

Candidates for upstream contribution to chemotools.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.utils.validation import check_is_fitted


class DirectStandardization(BaseEstimator, TransformerMixin):
    """
    Direct Standardization (DS) for spectral model transfer.

    Learns a linear transformation matrix F such that X_master ≈ X_slave @ F
    using ordinary least squares on a set of transfer samples.

    Parameters
    ----------
    n_components : int, default=5
        Number of PLS components used to compute the transfer matrix.
        Using PLS (rather than OLS) regularises the solution when n_features > n_transfer.

    Attributes
    ----------
    F_ : ndarray, shape (p_slave, p_master)
        The learned transfer matrix.

    Examples
    --------
    >>> ds = DirectStandardization(n_components=5)
    >>> ds.fit(X_slave_transfer, X_master_transfer)
    >>> X_corrected = ds.transform(X_slave_new)
    """

    def __init__(self, n_components=5):
        self.n_components = n_components

    def fit(self, X_slave, X_master):
        X_slave = np.asarray(X_slave, dtype=np.float64)
        X_master = np.asarray(X_master, dtype=np.float64)
        pls = PLSRegression(n_components=self.n_components)
        pls.fit(X_slave, X_master)
        self.F_ = pls.coef_.T           # shape: (p_slave, p_master) after sklearn 1.x
        self.x_mean_slave_ = pls.x_mean_
        self.y_mean_master_ = pls.y_mean_
        self.n_features_in_ = X_slave.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self, "F_")
        X = np.asarray(X, dtype=np.float64)
        return (X - self.x_mean_slave_) @ self.F_ + self.y_mean_master_


class PiecewiseDirectStandardization(BaseEstimator, TransformerMixin):
    """
    Piecewise Direct Standardization (PDS) for spectral model transfer.

    Each wavelength channel of the master is predicted from a local window of
    slave channels centred on the corresponding channel. This accounts for
    small wavelength shifts between instruments.

    Parameters
    ----------
    window_width : int, default=5
        Half-width of the local wavelength window (total window = 2*window_width + 1).
    n_components : int, default=3
        Number of PLS components used per local model.

    Attributes
    ----------
    local_models_ : list of PLSRegression
        One fitted model per master channel.
    n_features_in_ : int

    Examples
    --------
    >>> pds = PiecewiseDirectStandardization(window_width=5, n_components=3)
    >>> pds.fit(X_slave_transfer, X_master_transfer)
    >>> X_corrected = pds.transform(X_slave_new)
    """

    def __init__(self, window_width=5, n_components=3):
        self.window_width = window_width
        self.n_components = n_components

    def fit(self, X_slave, X_master):
        X_slave = np.asarray(X_slave, dtype=np.float64)
        X_master = np.asarray(X_master, dtype=np.float64)
        n_channels = X_master.shape[1]
        self.local_models_ = []
        self.windows_ = []

        for k in range(n_channels):
            lo = max(0, k - self.window_width)
            hi = min(n_channels, k + self.window_width + 1)
            X_win = X_slave[:, lo:hi]
            nc = min(self.n_components, X_win.shape[1], X_win.shape[0] - 1)
            pls = PLSRegression(n_components=nc)
            pls.fit(X_win, X_master[:, k])
            self.local_models_.append(pls)
            self.windows_.append((lo, hi))

        self.n_features_in_ = X_slave.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self, "local_models_")
        X = np.asarray(X, dtype=np.float64)
        n_channels = len(self.local_models_)
        X_corrected = np.zeros((X.shape[0], n_channels))
        for k, (pls, (lo, hi)) in enumerate(zip(self.local_models_, self.windows_)):
            X_corrected[:, k] = pls.predict(X[:, lo:hi]).ravel()
        return X_corrected
