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
    ix_cf_arr = np.array(ix_cf, dtype=np.int32)
    delta_r = 0.0

    # TODO: full numpy implementation (ditch loop)
    for i in ix_cf_arr:
        delta_r += S[k, i]
        delta_r += S[i, k]
    delta_r += S[k, k]

    return delta_r


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
    ix_ci_arr = np.array(ix_ci, dtype=np.int32)
    delta_r = 0.0

    # TODO: use native numpy (no python loop)
    for i in ix_ci_arr:
        delta_r -= S[k, i]
        delta_r -= S[i, k]
    delta_r += S[k, k]

    return delta_r


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

    imax = T.shape[0]
    S = np.zeros((imax, imax), dtype=np.float64)

    for i in range(imax):
        for j in range(imax):
            for k in range(imax):
                S[i, j] += p0[i] * T[i, k] * (1.0 / pt[k]) * T[j, k] * p0[j]
            S[i, j] -= p0[i] * p0[j]

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
    # Compute probability distributions
    p1 = np.zeros(n1, dtype=np.float64)
    p2 = np.zeros(n2, dtype=np.float64)
    p12 = np.zeros(n1 * n2, dtype=np.float64)

    k = 0
    for i, clust1 in enumerate(clusters1):
        p1[i] = len(clust1) / N
        for j, clust2 in enumerate(clusters2):
            p12[k] = len(clust1.intersection(clust2)) / N
            k += 1

    for j, clust2 in enumerate(clusters2):
        p2[j] = len(clust2) / N

    # Compute Shannon entropies
    H1 = 0.0
    H2 = 0.0
    H12 = 0.0

    # TODO use proper numpy operatios, (np.sum with !=0 condition)
    for i in range(n1):
        if p1[i] != 0:
            H1 -= p1[i] * np.log2(p1[i])

    for j in range(n2):
        if p2[j] != 0:
            H2 -= p2[j] * np.log2(p2[j])

    for j in range(n1 * n2):
        if p12[j] != 0:
            H12 -= p12[j] * np.log2(p12[j])

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
    VI = 0.0

    # Loop over pairs of clusters
    for i in range(n1):
        clust1 = clusters1[i]
        ni = float(len(clust1))
        n_inter = 0

        for j in range(n2):
            clust2 = clusters2[j]
            l_inter = len(clust1.intersection(clust2))
            nij = float(l_inter)
            n_inter += l_inter

            if nij > 0:
                nj = float(len(clust2))
                VI -= nij * np.log2((nij * nij) / (ni * nj))

            if n_inter >= ni:
                # We have found all possible intersections
                break

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
        "nvi_mat is not implemented in the pure Python fallback. "
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
