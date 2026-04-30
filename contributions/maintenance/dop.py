"""
DOP: Dynamic Orthogonal Projection for model maintenance and drift correction.

Removes instrument/environmental drift from new spectra by projecting out the
subspace spanned by the difference between reference and drifted spectra.

Candidate for upstream contribution to chemotools.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.utils.validation import check_is_fitted


class DOPCorrector(BaseEstimator, TransformerMixin):
    """
    Dynamic Orthogonal Projection (DOP) for spectral drift correction.

    Computes a low-rank basis of the drift subspace from paired reference and
    new-condition spectra, then removes that subspace from incoming spectra.

    Parameters
    ----------
    n_components : int, default=2
        Number of principal components used to model the drift subspace.

    Attributes
    ----------
    drift_basis_ : ndarray, shape (n_components, n_features)
        Orthonormal basis vectors of the drift subspace.
    n_features_in_ : int

    Examples
    --------
    >>> dop = DOPCorrector(n_components=2)
    >>> dop.fit(X_reference, X_new_condition)
    >>> X_corrected = dop.transform(X_future)
    """

    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X_reference, X_new):
        """
        Parameters
        ----------
        X_reference : ndarray, shape (n, p)
            Spectra from the reference condition (e.g., earlier season / original instrument).
        X_new : ndarray, shape (m, p)
            Spectra from the drifted condition (e.g., new season / after instrument change).
        """
        X_ref = np.asarray(X_reference, dtype=np.float64)
        X_new = np.asarray(X_new, dtype=np.float64)

        # Drift signal: difference between new and reference mean spectra
        drift = X_new - X_ref.mean(axis=0)

        # PCA on drift matrix to find the dominant drift directions
        nc = min(self.n_components, drift.shape[0], drift.shape[1])
        pca = PCA(n_components=nc)
        pca.fit(drift)
        self.drift_basis_ = pca.components_   # shape: (nc, p)
        self.n_features_in_ = X_ref.shape[1]
        return self

    def transform(self, X):
        """
        Remove drift subspace from X.

        Parameters
        ----------
        X : ndarray, shape (n, p)

        Returns
        -------
        X_corrected : ndarray, shape (n, p)
        """
        check_is_fitted(self, "drift_basis_")
        X = np.asarray(X, dtype=np.float64)
        B = self.drift_basis_          # (nc, p)
        # Project onto drift basis and subtract
        projection = X @ B.T @ B      # (n, p)
        return X - projection
