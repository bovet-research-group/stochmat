"""
Pure Python fallback implementations for fast.pyx functions.

This module provides drop-in replacement functions in case the Cython
extension module 'fast' fails to compile or load.

Performance Note:
    These implementations use NumPy and are significantly slower than the
    compiled Cython versions, especially for large datasets. Consider ensuring
    Cython is properly installed and the extensions are compiled for production
    use.
"""
import numpy as np


def sum_Sto(S, k, ix_cf):
    """Pure Python fallback for sum_Sto.

    Computes the stability gain from moving node k INTO community defined by
    ix_cf.

    Parameters
    ----------
    S : ndarray
        2D array representing the stability matrix
    k : int
        Node index
    ix_cf : list
        List of node indices in the target community

    Returns
    -------
    float
        Stability gain value
    """
    ix_cf_arr = np.asarray(ix_cf, dtype=np.intp)
    return float(S[k, ix_cf_arr].sum() + S[ix_cf_arr, k].sum() + S[k, k])


def sum_Sout(S, k, ix_ci):
    """Pure Python fallback for sum_Sout.

    Computes the stability gain from moving node k OUT OF community defined by
    ix_ci.

    Parameters
    ----------
    S : ndarray
        2D array representing the stability matrix
    k : int
        Node index
    ix_ci : list
        List of node indices in the current community

    Returns
    -------
    float
        Stability gain value
    """
    ix_ci_arr = np.asarray(ix_ci, dtype=np.intp)
    return float(-S[k, ix_ci_arr].sum() - S[ix_ci_arr, k].sum() + S[k, k])


def compute_S(p1, p2, T):
    """Pure Python fallback for compute_S.

    Computes the internal matrix comparing probabilities for each node:
        S[i,j] = p1[i]*T[i,j] - p1[i]*p2[j]

    Parameters
    ----------
    p1 : ndarray
        1D array of probabilities
    p2 : ndarray
        1D array of probabilities
    T : ndarray
        2D transition matrix

    Returns
    -------
    ndarray
        2D stability matrix S
    """
    # Convert to numpy arrays if needed
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)

    # Compute: diag(p1) @ T - outer(p1, p2)
    S = np.diag(p1) @ T - np.outer(p1, p2)

    return S


def compute_S_0t0(p0, pt, T):
    """Pure Python fallback for compute_S_0t0.

    Computes a specialized stability matrix for the 0-to-0 case.

    Parameters
    ----------
    p0 : ndarray
        1D array of initial probabilities
    pt : ndarray
        1D array of probabilities at time t
    T : ndarray
        2D transition matrix

    Returns
    -------
    ndarray
        2D stability matrix S
    """
    p0 = np.asarray(p0, dtype=np.float64)
    pt = np.asarray(pt, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)

    # Avoid division by zero: where pt[k] == 0 the term contributes nothing.
    inv_pt = np.zeros_like(pt)
    nz = pt != 0
    inv_pt[nz] = 1.0 / pt[nz]

    # M[i, j] = sum_k T[i, k] * inv_pt[k] * T[j, k]
    M = (T * inv_pt) @ T.T
    S = np.outer(p0, p0) * (M - 1.0)

    return S


def nmi(clusters1, clusters2, N, n1, n2):
    """Pure Python fallback for nmi.

    Computes normalized mutual information between two clusterings.

    Parameters
    ----------
    clusters1 : list
        List of sets, each set containing node indices in a cluster
    clusters2 : list
        List of sets, each set containing node indices in a cluster
    N : int
        Total number of nodes
    n1 : int
        Number of clusters in clusters1
    n2 : int
        Number of clusters in clusters2

    Returns
    -------
    float
        Normalized mutual information value
    """
    # Cluster sizes and joint counts
    sizes1 = np.fromiter((len(c) for c in clusters1), dtype=np.float64, count=n1)
    sizes2 = np.fromiter((len(c) for c in clusters2), dtype=np.float64, count=n2)
    inter = np.array(
        [[len(c1.intersection(c2)) for c2 in clusters2] for c1 in clusters1],
        dtype=np.float64,
    )

    p1 = sizes1 / N
    p2 = sizes2 / N
    p12 = inter / N

    def _entropy(p):
        p = p[p != 0]
        return -np.sum(p * np.log2(p))

    H1 = _entropy(p1)
    H2 = _entropy(p2)
    H12 = _entropy(p12)

    # Return normalized mutual information
    return (H1 + H2 - H12) / max(H1, H2)


def nvi(clusters1, clusters2, N):
    """Pure Python fallback for nvi.

    Computes normalized variation of information between two clusterings.

    Parameters
    ----------
    clusters1 : list
        List of sets, each set containing node indices in a cluster
    clusters2 : list
        List of sets, each set containing node indices in a cluster
    N : int
        Total number of nodes

    Returns
    -------
    float
        Normalized variation of information value
    """
    n1 = len(clusters1)
    n2 = len(clusters2)

    sizes1 = np.fromiter((len(c) for c in clusters1), dtype=np.float64, count=n1)
    sizes2 = np.fromiter((len(c) for c in clusters2), dtype=np.float64, count=n2)
    inter = np.array(
        [[len(c1.intersection(c2)) for c2 in clusters2] for c1 in clusters1],
        dtype=np.float64,
    )

    # ni * nj outer product, then compute -nij * log2(nij^2 / (ni * nj))
    # only for entries where nij > 0.
    mask = inter > 0
    ni_nj = np.outer(sizes1, sizes2)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_terms = np.where(
            mask, np.log2((inter * inter) / np.where(mask, ni_nj, 1.0)), 0.0
        )
    VI = -np.sum(inter * log_terms)

    return VI / (float(N) * np.log2(float(N)))


def nvi_parallel(clusters1, clusters2, N, num_threads):
    """Pure Python fallback for nvi_parallel.

    The parallel version is not implemented in the pure Python fallback.
    Use the sequential version nvi instead.

    Parameters
    ----------
    clusters1 : list
        List of sets, each set containing node indices in a cluster
    clusters2 : list
        List of sets, each set containing node indices in a cluster
    N : int
        Total number of nodes
    num_threads : int
        Number of threads (ignored in fallback)

    Raises
    ------
    NotImplementedError
        Always raised; parallel version requires compiled Cython extension
    """
    raise NotImplementedError(
        "nvi_parallel is not implemented in the pure Python fallback. "
        "Please use nvi (sequential version) instead, or ensure the "
        "Cython extension module 'stochmat.fast' is properly compiled."
    )


def nvi_vectors(clusters1, clusters2, N):
    """Pure Python fallback for nvi_vectors.

    This optimized vector-based version is not implemented in the pure Python
    fallback. Use the standard version nvi instead.

    Parameters
    ----------
    clusters1 : list
        List of integer arrays, each array containing node indices in a cluster
    clusters2 : list
        List of integer arrays, each array containing node indices in a cluster
    N : int
        Total number of nodes

    Raises
    ------
    NotImplementedError
        Always raised; vector version requires compiled Cython extension
    """
    raise NotImplementedError(
        "nvi_vectors is not implemented in the pure Python fallback. "
        "Please use nvi instead, or ensure the Cython extension module "
        "'stochmat.fast' is properly compiled."
    )


def nvi_mat(clusters1, clusters2, N):
    """Pure Python fallback for nvi_mat.

    This optimized matrix-based version is not implemented in the pure Python
    fallback.
    Use the standard version nvi instead.

    Parameters
    ----------
    clusters1 : list
        List of sets, each set containing node indices in a cluster
    clusters2 : list
        List of sets, each set containing node indices in a cluster
    N : int
        Total number of nodes

    Raises
    ------
    NotImplementedError
        Always raised; matrix version requires compiled Cython extension
    """
    raise NotImplementedError(
        "nvi_mat (matrix-based version) is not implemented in the pure Python fallback. "
        "Please use nvi instead, or ensure the Cython extension module "
        "'stochmat.fast' is properly compiled."
    )


def nvi_mat_test(clusters1, clusters2, N):
    """Pure Python fallback for nvi_mat_test.

    This is a test function not intended for production use.

    Parameters
    ----------
    clusters1 : list
        List of sets, each set containing node indices in a cluster
    clusters2 : list
        List of sets, each set containing node indices in a cluster
    N : int
        Total number of nodes

    Raises
    ------
    NotImplementedError
        Always raised; test function only available in Cython extension
    """
    raise NotImplementedError(
        "nvi_mat_test is a test function only available in the "
        "compiled Cython extension module 'stochmat.fast'."
    )


def test():
    """Pure Python fallback for test.

    This is a test function not intended for production use.

    Raises
    ------
    NotImplementedError
        Always raised; test function only available in Cython extension
    """
    raise NotImplementedError(
        "test() is a test function only available in the compiled "
        "Cython extension module 'stochmat.fast'."
    )
