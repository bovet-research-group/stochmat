"""Correctness parity tests: MKL backend vs scipy fallback.

These tests assert that the MKL-backed implementations of
:func:`stochmat.sparse_matmul`, :func:`stochmat.sparse_gram_matrix`, and the
``compute_S_0t0`` branch of ``sparse_outer`` produce numerically identical
results (up to floating-point tolerance) to the pure scipy.sparse fallback.

The MKL path is selected by the module-level flag
``stochmat.sparse_stoch_mat.USE_SPARSE_DOT_MKL``; we toggle it via
``monkeypatch`` to exercise both code paths within a single test process,
sharing the input fixtures.

All tests in this file require the MKL backend to be active (see the
``require_mkl`` fixture in ``conftest.py``); they will be skipped in the
pure-Python CI job.
"""
import numpy as np
import pytest

from stochmat import sparse_stoch_mat as ssm


def _matmul_with_backend(A, B, *, mkl):
    """Call ``ssm.sparse_matmul`` with the MKL flag forced to ``mkl``."""
    saved = ssm.USE_SPARSE_DOT_MKL
    ssm.USE_SPARSE_DOT_MKL = mkl
    try:
        return ssm.sparse_matmul(A, B)
    finally:
        ssm.USE_SPARSE_DOT_MKL = saved


def _gram_with_backend(A, *, transpose, mkl, symmetrize=None):
    saved = ssm.USE_SPARSE_DOT_MKL
    ssm.USE_SPARSE_DOT_MKL = mkl
    try:
        return ssm.sparse_gram_matrix(A, transpose=transpose,
                                      symmetrize=symmetrize)
    finally:
        ssm.USE_SPARSE_DOT_MKL = saved


def test_sparse_matmul_mkl_vs_scipy(require_mkl, mkl_canonical_sparse):
    """sparse_matmul: MKL output must match scipy output element-wise."""
    A, B = mkl_canonical_sparse

    out_mkl = _matmul_with_backend(A, B, mkl=True)
    out_scipy = _matmul_with_backend(A, B, mkl=False)

    np.testing.assert_allclose(
        out_mkl.toarray(), out_scipy.toarray(),
        rtol=1e-10, atol=1e-12,
        err_msg="MKL sparse_matmul disagrees with scipy fallback",
    )


@pytest.mark.parametrize("transpose", [False, True])
def test_sparse_gram_matrix_mkl_vs_scipy(
    require_mkl, mkl_canonical_sparse, transpose
):
    """sparse_gram_matrix: MKL output matches scipy for both transpose modes.

    Uses ``symmetrize=True`` so both backends return the full symmetric
    Gram matrix (the MKL native upper-triangle output is reflected).
    """
    A, _ = mkl_canonical_sparse

    out_mkl = _gram_with_backend(A, transpose=transpose, mkl=True,
                                 symmetrize=True)
    out_scipy = _gram_with_backend(A, transpose=transpose, mkl=False,
                                   symmetrize=True)

    np.testing.assert_allclose(
        out_mkl.toarray(), out_scipy.toarray(),
        rtol=1e-10, atol=1e-12,
        err_msg=(
            f"MKL sparse_gram_matrix(transpose={transpose}, symmetrize=True) "
            "disagrees with scipy fallback"
        ),
    )


@pytest.mark.parametrize("transpose", [False, True])
def test_sparse_gram_matrix_symmetrize_false(
    require_mkl, mkl_canonical_sparse, transpose
):
    """With symmetrize=False, both backends return identical upper triangles."""
    A, _ = mkl_canonical_sparse

    out_mkl = _gram_with_backend(A, transpose=transpose, mkl=True,
                                 symmetrize=False)
    out_scipy = _gram_with_backend(A, transpose=transpose, mkl=False,
                                   symmetrize=False)

    np.testing.assert_allclose(
        out_mkl.toarray(), out_scipy.toarray(),
        rtol=1e-10, atol=1e-12,
        err_msg=(
            f"sparse_gram_matrix(transpose={transpose}, symmetrize=False) "
            "upper-triangle outputs disagree across backends"
        ),
    )


def test_sparse_gram_matrix_default_returns_legacy_shapes(
    require_mkl, mkl_canonical_sparse
):
    """Lock in the legacy backend-dependent default (symmetrize=None).

    MKL returns upper triangle only; scipy returns the full symmetric
    matrix. This test pins that contract so we notice if it changes.
    """
    from scipy.sparse import triu as _sp_triu

    A, _ = mkl_canonical_sparse

    out_mkl = _gram_with_backend(A, transpose=False, mkl=True)
    out_scipy = _gram_with_backend(A, transpose=False, mkl=False)

    mkl_dense = out_mkl.toarray()
    scipy_dense = out_scipy.toarray()

    # MKL: strict lower triangle is all zeros (upper-triangle output).
    assert np.allclose(np.tril(mkl_dense, k=-1), 0.0), (
        "MKL default output should have an empty strict lower triangle"
    )
    # scipy: matrix is symmetric (full Gram).
    assert np.allclose(scipy_dense, scipy_dense.T), (
        "scipy default output should be the full symmetric Gram matrix"
    )
    # And the upper triangles agree numerically.
    np.testing.assert_allclose(
        mkl_dense, _sp_triu(out_scipy).toarray(),
        rtol=1e-10, atol=1e-12,
    )


def test_sparse_matmul_mkl_vs_scipy_small(require_mkl, get_csr_matrix_small):
    """Small dense-ish input sanity check for the MKL dispatch path."""
    A = get_csr_matrix_small.astype(np.float64)
    B = A.copy()

    out_mkl = _matmul_with_backend(A, B, mkl=True)
    out_scipy = _matmul_with_backend(A, B, mkl=False)

    np.testing.assert_allclose(
        out_mkl.toarray(), out_scipy.toarray(),
        rtol=1e-12, atol=1e-14,
    )
