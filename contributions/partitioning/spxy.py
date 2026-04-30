"""
SPXY and DUPLEX sample partitioning algorithms.

SPXY (Sample set Partitioning based on joint X-Y distances) extends Kennard-Stone
by accounting for both spectral (X) and reference (y) distances.

DUPLEX is an interleaved selection strategy that fills calibration and test sets
alternately, preserving the joint distribution in both sets.

Candidates for upstream contribution to chemotools.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler


def _pairwise_distance(A, B=None):
    """Euclidean pairwise distance matrix between rows of A (and B)."""
    if B is None:
        B = A
    diff = A[:, None, :] - B[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))


def spxy(X, y, test_size=0.2, scale=True):
    """
    SPXY sample partitioning (Galvao et al., 2005).

    Selects a calibration set that spans the joint X-y space using a
    max-min distance strategy analogous to Kennard-Stone.

    Parameters
    ----------
    X : ndarray, shape (n, p)
    y : ndarray, shape (n,) or (n, q)
    test_size : float, default=0.2
        Fraction of samples to hold out as test set.
    scale : bool, default=True
        Standardise X and y before computing distances.

    Returns
    -------
    cal_idx : ndarray of int
    test_idx : ndarray of int
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]

    n = X.shape[0]
    n_test = max(1, int(np.round(n * test_size)))
    n_cal = n - n_test

    if scale:
        X = StandardScaler().fit_transform(X)
        y = StandardScaler().fit_transform(y)

    # Normalise distances by variable ranges so X and y contribute equally
    dx = _pairwise_distance(X) / X.shape[1]
    dy = _pairwise_distance(y) / y.shape[1]
    D = dx + dy

    selected = []
    # Start with the pair of samples with maximum distance
    i, j = np.unravel_index(np.argmax(D), D.shape)
    selected.extend([int(i), int(j)])
    remaining = list(set(range(n)) - set(selected))

    while len(selected) < n_cal:
        min_dists = D[remaining][:, selected].min(axis=1)
        best = remaining[int(np.argmax(min_dists))]
        selected.append(best)
        remaining.remove(best)

    cal_idx = np.array(selected, dtype=int)
    test_idx = np.array(remaining, dtype=int)
    return cal_idx, test_idx


def duplex(X, y=None, test_size=0.2, scale=True):
    """
    DUPLEX sample partitioning (Snee, 1977).

    Interleaves Kennard-Stone selection between calibration and test sets so
    that both cover the feature space uniformly.

    Parameters
    ----------
    X : ndarray, shape (n, p)
    y : ndarray or None
        Ignored (accepted for API symmetry with spxy).
    test_size : float, default=0.2
        Fraction of samples allocated to test set.
    scale : bool, default=True
        Standardise X before computing distances.

    Returns
    -------
    cal_idx : ndarray of int
    test_idx : ndarray of int
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    n_test = max(1, int(np.round(n * test_size)))
    n_cal = n - n_test

    if scale:
        X = StandardScaler().fit_transform(X)

    D = _pairwise_distance(X)
    cal, test = [], []
    remaining = list(range(n))

    def _pick_farthest(selected, pool):
        if not selected:
            i, j = np.unravel_index(np.argmax(D[np.ix_(pool, pool)]), (len(pool), len(pool)))
            return pool[i], pool[j]
        dists = D[pool][:, selected].min(axis=1)
        return pool[int(np.argmax(dists))], None

    # Seed both sets with their most distant pair
    pool = list(range(n))
    ci, cj = _pick_farthest([], pool)
    cal.extend([ci, cj])
    remaining = [x for x in remaining if x not in [ci, cj]]

    ti, tj = _pick_farthest([], remaining)
    test.extend([ti, tj])
    remaining = [x for x in remaining if x not in [ti, tj]]

    # Alternate selection
    toggle = True
    while remaining and (len(cal) < n_cal or len(test) < n_test):
        target = cal if (toggle and len(cal) < n_cal) else test
        all_selected = cal + test
        dists = D[remaining][:, all_selected].min(axis=1)
        best = remaining[int(np.argmax(dists))]
        target.append(best)
        remaining.remove(best)
        toggle = not toggle

    return np.array(cal, dtype=int), np.array(test + remaining, dtype=int)
