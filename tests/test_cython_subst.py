import pytest

import numpy as np
from scipy.sparse import (
    eye,
    )

def test_sparse_stoch_from_full_csr(cs_matrix_creator):
    """
    """
    from stochmat._cython_subst import sparse_stoch_from_full_csr as ssffc_subst
    from stochmat.sparse_stoch_mat import _css
    A_csr = cs_matrix_creator(nbr=1, size=100000, nbr_non_zeros=1000)[0]
    diag_val = 1.0
    nz_rows, nz_cols = (A_csr - diag_val * eye(A_csr.shape[0], format="csr")).nonzero()
    nz_rowcols = np.union1d(nz_rows, nz_cols)
    ssm_args = _css.sparse_stoch_from_full_csr(nz_rowcols, A_csr.data,
                                                   A_csr.indices,
                                                   A_csr.indptr, diag_val)
    ssm_args_subst = ssffc_subst(nz_rowcols, A_csr.data, A_csr.indices,
                                     A_csr.indptr, diag_val)
    # size
    assert ssm_args[0] == ssm_args_subst[0]
    # data
    np.testing.assert_equal(ssm_args_subst[1], ssm_args[1])
    # indices
    np.testing.assert_equal(ssm_args_subst[2], ssm_args[2])
    # indptr
    np.testing.assert_equal(ssm_args_subst[3], ssm_args[3])
    # diag val
    np.testing.assert_equal(ssm_args_subst[4], ssm_args[4])


def test_inplace_csr_row_normalize():
    """Verify in-place row normalization against a deterministic input.

    Historically this test built its matrix via the ``cs_matrix_creator``
    fixture (100000x100000 with 1000 random nonzeros) and asserted that
    ``A_csr.data`` had changed after calling
    ``inplace_csr_row_normalize(..., n_row=333, row_sum=1)``. Two
    independent issues conspired:

    * ``n_row`` is the *count* of leading rows to normalize (the
      implementation iterates ``range(n_row)``), not an index — so the
      original test was normalizing rows ``[0, 333)``, not row 333.
    * With density 1e-4, almost every leading row was either empty or a
      single self-loop with value ``1.0`` (added by the fixture for
      zero-sum rows). Normalizing those to ``row_sum=1`` is a bit-exact
      no-op, so the "data changed" assertion fired on roughly 99% of
      unseeded RNG states. The test only "passed" by luck when several
      of the leading rows happened to land multiple random nonzeros that
      summed to a value not bit-exactly equal to 1.0.

    The new construction pins down a deterministic dense-ish input,
    normalizes every row, and asserts:

    * each (post-normalization) row sums to ``row_sum`` -- the actual
      contract of the function, and
    * the compiled ``_css`` and pure-Python ``_cython_subst`` paths
      produce bit-equivalent results (parity).
    """
    from scipy.sparse import csr_matrix

    from stochmat._cython_subst import inplace_csr_row_normalize as icrn_subst
    from stochmat.sparse_stoch_mat import _css

    size = 1000
    # ``n_row`` is the *count* of leading rows to normalize (the function
    # iterates ``range(n_row)``), not an index of a specific row.
    n_row = size
    row_sum = 1.0

    rng = np.random.default_rng(20260430)
    dense = rng.random((size, size))
    dense[dense < 0.95] = 0.0
    # Pin a couple of specific rows to known multi-entry contents that
    # do not bit-exactly sum to ``row_sum`` so that normalization is a
    # real (non-identity) operation we can post-condition on.
    pinned_rows = (333, 777)
    dense[333, :] = 0.0
    dense[333, :5] = [2.0, 3.0, 5.0, 7.0, 11.0]  # sum = 28.0
    dense[777, :] = 0.0
    dense[777, :3] = [0.25, 0.75, 1.5]           # sum = 2.5
    # Ensure no row is empty so every row has a well-defined normalization.
    empty = np.where(dense.sum(axis=1) == 0)[0]
    for r in empty:
        dense[r, r] = 1.0

    A_csr = csr_matrix(dense)
    A_csr.data = A_csr.data.astype(np.float64, copy=False)
    A_csr.indices = A_csr.indices.astype(np.int32, copy=False)
    A_csr.indptr = A_csr.indptr.astype(np.int32, copy=False)
    B_csr = A_csr.copy()

    _css.inplace_csr_row_normalize(
        A_csr.data, A_csr.indptr.astype(np.int64), n_row, row_sum,
    )
    icrn_subst(B_csr.data, B_csr.indptr, n_row, row_sum)

    # Positive post-condition: every normalized row sums to ``row_sum``.
    for r in pinned_rows:
        np.testing.assert_allclose(A_csr[r].sum(), row_sum, rtol=1e-12)
    np.testing.assert_allclose(
        np.asarray(A_csr.sum(axis=1)).ravel(),
        np.full(size, row_sum, dtype=np.float64),
        rtol=1e-12,
    )
    # Parity: compiled (or its pure-Python substitute when no compiled
    # extension is available) and the explicit pure-Python fallback agree.
    np.testing.assert_allclose(A_csr.data, B_csr.data, rtol=1e-12, atol=0.0)
    

def test_stoch_mat_add(SSM_matrix_creator, compare_SSM_args):
    """
    """
    from stochmat._cython_subst import stoch_mat_add as sma_subst
    from stochmat.sparse_stoch_mat import _css
    size=100000
    full_size = 10*size
    A, B = SSM_matrix_creator(nbr=2, size=100000, nbr_non_zeros=1000)

    ssm_args = _css.stoch_mat_add(
        size=A.size,
        Adata=A.T_small.data,
        Aindices=A.T_small.indices,
        Aindptr=A.T_small.indptr,
        Anz_rowcols=A.nz_rowcols,
        Adiag_val=A.diag_val,
        Bdata=B.T_small.data,
        Bindices=B.T_small.indices,
        Bindptr=B.T_small.indptr,
        Bnz_rowcols=B.nz_rowcols,
        Bdiag_val=B.diag_val,
    )
    ssm_args_subst = sma_subst(
        size=A.size,
        Adata=A.T_small.data,
        Aindices=A.T_small.indices,
        Aindptr=A.T_small.indptr,
        Anz_rowcols=A.nz_rowcols,
        Adiag_val=A.diag_val,
        Bdata=B.T_small.data,
        Bindices=B.T_small.indices,
        Bindptr=B.T_small.indptr,
        Bnz_rowcols=B.nz_rowcols,
        Bdiag_val=B.diag_val,
    )
    compare_SSM_args(ssm_args, ssm_args_subst)
    

def test_stoch_mat_sub(SSM_matrix_creator, compare_SSM_args):
    """
    """
    from stochmat._cython_subst import stoch_mat_sub as sma_subst
    from stochmat.sparse_stoch_mat import _css
    size=100000
    full_size = 10*size
    A, B = SSM_matrix_creator(nbr=2, size=100000, nbr_non_zeros=1000)

    ssm_args = _css.stoch_mat_sub(
        size=A.size,
        Adata=A.T_small.data,
        Aindices=A.T_small.indices,
        Aindptr=A.T_small.indptr,
        Anz_rowcols=A.nz_rowcols,
        Adiag_val=A.diag_val,
        Bdata=B.T_small.data,
        Bindices=B.T_small.indices,
        Bindptr=B.T_small.indptr,
        Bnz_rowcols=B.nz_rowcols,
        Bdiag_val=B.diag_val,
    )
    ssm_args_subst = sma_subst(
        size=A.size,
        Adata=A.T_small.data,
        Aindices=A.T_small.indices,
        Aindptr=A.T_small.indptr,
        Anz_rowcols=A.nz_rowcols,
        Adiag_val=A.diag_val,
        Bdata=B.T_small.data,
        Bindices=B.T_small.indices,
        Bindptr=B.T_small.indptr,
        Bnz_rowcols=B.nz_rowcols,
        Bdiag_val=B.diag_val,
    )
    compare_SSM_args(ssm_args, ssm_args_subst)

def test_rebuild_nnz_rowcol(SSM_matrix_creator, compare_SSM_args):
    """
    """
    from stochmat._cython_subst import rebuild_nnz_rowcol as rnr_subst
    from stochmat.sparse_stoch_mat import _css
    size=100000
    full_size = 10*size
    A = SSM_matrix_creator(nbr=2, size=100000, nbr_non_zeros=1000)[0]
    (data, indices, indptr, size) = _css.rebuild_nnz_rowcol(
        T_data=A.T_small.data,
        T_indices=A.T_small.indices.astype(np.int64),
        T_indptr=A.T_small.indptr.astype(np.int64),
        nonzero_indices=A.nz_rowcols.astype(np.int64),
        size=A.size,
        diag_val=A.diag_val
    )
    ssm_args = (size, np.asarray(data), np.asarray(indices), indptr, A.size)
    (data_, indices_, indptr_, size_) = rnr_subst(
        T_data=A.T_small.data,
        T_indices=A.T_small.indices.astype(np.int64),
        T_indptr=A.T_small.indptr.astype(np.int64),
        nonzero_indices=A.nz_rowcols.astype(np.int64),
        size=A.size,
        diag_val=A.diag_val
    )
    ssm_args_subst = (size_, data_, indices_, indptr_, A.size)
    compare_SSM_args(ssm_args, ssm_args_subst)

def test_get_submat_sum(cs_matrix_creator):
    """
    """
    from stochmat._cython_subst import get_submat_sum as gss_subst
    from stochmat.sparse_stoch_mat import _css
    A_csr = cs_matrix_creator(nbr=1, size=100000, nbr_non_zeros=1000)[0]
    row_idx=np.arange(20, 200).astype(np.int32)
    col_idx=np.arange(5,500).astype(np.int32)
    subm_sum = _css.get_submat_sum(
        Adata=A_csr.data,
        Aindices=A_csr.indices,
        Aindptr=A_csr.indptr,
        row_idx=row_idx,
        col_idx=col_idx
    )
    subm_sum_subst = gss_subst(
        Adata=A_csr.data,
        Aindices=A_csr.indices,
        Aindptr=A_csr.indptr,
        row_idx=row_idx,
        col_idx=col_idx
    )
    assert subm_sum == subm_sum_subst


# ---------------------------------------------------------------------------
# Parity tests added to cover previously untested fallback functions.
# ---------------------------------------------------------------------------

@pytest.fixture(scope='function')
def small_csr_for_aggregate():
    """A small deterministic CSR matrix and an idx_list partition for
    aggregation parity tests.
    """
    from scipy.sparse import csr_matrix
    rng = np.random.default_rng(123)
    size = 20
    dense = rng.random((size, size))
    dense[dense < 0.6] = 0.0
    A = csr_matrix(dense)
    A.data = A.data.astype(np.float64, copy=False)
    A.indices = A.indices.astype(np.int32, copy=False)
    A.indptr = A.indptr.astype(np.int32, copy=False)
    # partition rows/cols 0..19 into 5 contiguous clusters of size 4
    idx_list = [list(range(4 * i, 4 * (i + 1))) for i in range(5)]
    idxs_array = np.array([i for idx in idx_list for i in idx], dtype=np.int32)
    idxptr = np.cumsum([0] + [len(idx) for idx in idx_list], dtype=np.int32)
    return A, idxs_array, idxptr


def _coo_tuple_to_dense(Bdata, Brows, Bcols, new_size):
    """Materialize an aggregation result tuple as a dense matrix for
    order-independent comparison.
    """
    from scipy.sparse import coo_matrix
    return coo_matrix(
        (np.asarray(Bdata, dtype=np.float64),
         (np.asarray(Brows, dtype=np.int64),
          np.asarray(Bcols, dtype=np.int64))),
        shape=(new_size, new_size),
    ).toarray()


def test_aggregate_csr_mat_parity(small_csr_for_aggregate):
    from stochmat._cython_subst import aggregate_csr_mat as agg_subst
    from stochmat.sparse_stoch_mat import _css
    A, idxs_array, idxptr = small_csr_for_aggregate

    cy = _css.aggregate_csr_mat(
        A.data, A.indices, A.indptr, idxs_array, idxptr,
    )
    py = agg_subst(
        A.data, A.indices, A.indptr, idxs_array, idxptr,
    )
    assert cy[3] == py[3]
    np.testing.assert_allclose(
        _coo_tuple_to_dense(*cy),
        _coo_tuple_to_dense(*py),
        rtol=1e-12, atol=0.0,
    )


def test_aggregate_csr_mat_2_parity(small_csr_for_aggregate):
    from stochmat._cython_subst import aggregate_csr_mat_2 as agg2_subst
    from stochmat.sparse_stoch_mat import _css
    A, idxs_array, idxptr = small_csr_for_aggregate

    cy = _css.aggregate_csr_mat_2(
        A.data, A.indices, A.indptr, idxs_array, idxptr,
    )
    py = agg2_subst(
        A.data, A.indices, A.indptr, idxs_array, idxptr,
    )
    assert cy[3] == py[3]
    np.testing.assert_allclose(
        _coo_tuple_to_dense(*cy),
        _coo_tuple_to_dense(*py),
        rtol=1e-12, atol=0.0,
    )


def test_aggregate_csr_mat_variants_agree(small_csr_for_aggregate):
    """The two aggregation algorithms must produce the same dense result
    (different algorithms, same answer) — sanity-check both fallbacks.
    """
    from stochmat._cython_subst import (
        aggregate_csr_mat as agg_subst,
        aggregate_csr_mat_2 as agg2_subst,
    )
    A, idxs_array, idxptr = small_csr_for_aggregate
    a = _coo_tuple_to_dense(
        *agg_subst(A.data, A.indices, A.indptr, idxs_array, idxptr),
    )
    b = _coo_tuple_to_dense(
        *agg2_subst(A.data, A.indices, A.indptr, idxs_array, idxptr),
    )
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=0.0)


@pytest.fixture(scope='function')
def small_pt_csr_csc():
    """A small symmetric-ish PT matrix as csr+csc plus p1, p2, idx, k."""
    from scipy.sparse import csr_matrix
    rng = np.random.default_rng(321)
    size = 25
    dense = rng.random((size, size))
    dense[dense < 0.5] = 0.0
    PT = csr_matrix(dense)
    PT.data = PT.data.astype(np.float64, copy=False)
    PT.indices = PT.indices.astype(np.int32, copy=False)
    PT.indptr = PT.indptr.astype(np.int32, copy=False)
    PTcsc = PT.tocsc()
    PTcsc.data = PTcsc.data.astype(np.float64, copy=False)
    PTcsc.indices = PTcsc.indices.astype(np.int32, copy=False)
    PTcsc.indptr = PTcsc.indptr.astype(np.int32, copy=False)
    p1 = rng.random(size).astype(np.float64)
    p1 /= p1.sum()
    p2 = rng.random(size).astype(np.float64)
    p2 /= p2.sum()
    k = 7
    idx = np.array([1, 3, 5, 7, 11, 13, 17], dtype=np.int32)
    return PT, PTcsc, p1, p2, k, idx


def _delta_args(PT, PTcsc, k, idx):
    return (PT.data, PT.indices, PT.indptr,
            PTcsc.data, PTcsc.indices, PTcsc.indptr,
            k, idx)


def test_compute_delta_PT_moveto_parity(small_pt_csr_csc):
    from stochmat._cython_subst import compute_delta_PT_moveto as fn_subst
    from stochmat.sparse_stoch_mat import _css
    PT, PTcsc, _, _, k, idx = small_pt_csr_csc
    args = _delta_args(PT, PTcsc, k, idx)
    np.testing.assert_allclose(
        _css.compute_delta_PT_moveto(*args),
        fn_subst(*args),
        rtol=1e-12, atol=0.0,
    )


def test_compute_delta_PT_moveout_parity(small_pt_csr_csc):
    from stochmat._cython_subst import compute_delta_PT_moveout as fn_subst
    from stochmat.sparse_stoch_mat import _css
    PT, PTcsc, _, _, k, idx = small_pt_csr_csc
    args = _delta_args(PT, PTcsc, k, idx)
    np.testing.assert_allclose(
        _css.compute_delta_PT_moveout(*args),
        fn_subst(*args),
        rtol=1e-12, atol=0.0,
    )


def test_compute_delta_S_moveto_parity(small_pt_csr_csc):
    from stochmat._cython_subst import compute_delta_S_moveto as fn_subst
    from stochmat.sparse_stoch_mat import _css
    PT, PTcsc, p1, p2, k, idx = small_pt_csr_csc
    args = (*_delta_args(PT, PTcsc, k, idx), p1, p2)
    np.testing.assert_allclose(
        _css.compute_delta_S_moveto(*args),
        fn_subst(*args),
        rtol=1e-12, atol=0.0,
    )


def test_compute_delta_S_moveout_parity(small_pt_csr_csc):
    """Regression test for the rename of ``cython_compute_delta_S_moveout``
    → ``compute_delta_S_moveout`` in the pure-Python fallback. Without the
    rename, ``_css.compute_delta_S_moveout`` would be unresolvable when the
    compiled extension is not present.
    """
    from stochmat._cython_subst import compute_delta_S_moveout as fn_subst
    from stochmat.sparse_stoch_mat import _css
    PT, PTcsc, p1, p2, k, idx = small_pt_csr_csc
    args = (*_delta_args(PT, PTcsc, k, idx), p1, p2)
    np.testing.assert_allclose(
        _css.compute_delta_S_moveout(*args),
        fn_subst(*args),
        rtol=1e-12, atol=0.0,
    )


def test_inplace_csr_row_normalize_array_parity(cs_matrix_creator):
    from stochmat._cython_subst import (
        inplace_csr_row_normalize_array as fn_subst,
    )
    from stochmat.sparse_stoch_mat import _css
    A_csr = cs_matrix_creator(nbr=1, size=1000, nbr_non_zeros=200)[0]
    n_row = 500
    rng = np.random.default_rng(7)
    row_sum = rng.random(n_row).astype(np.float64)
    A = A_csr.copy()
    B = A_csr.copy()
    _css.inplace_csr_row_normalize_array(
        A.data, A.indptr.astype(np.int64), n_row, row_sum,
    )
    fn_subst(B.data, B.indptr.astype(np.int64), n_row, row_sum)
    np.testing.assert_allclose(A.data, B.data, rtol=1e-12, atol=0.0)


# ---------------------------------------------------------------------------
# Fallback NotImplementedError stubs that intentionally have no fallback
# implementation. Asserting their behaviour pins it down.
# ---------------------------------------------------------------------------

def test_nvi_fallbacks_raise_not_implemented():
    from stochmat import fast_subst
    with pytest.raises(NotImplementedError):
        fast_subst.nvi_parallel([], [], 0, 1)
    with pytest.raises(NotImplementedError):
        fast_subst.nvi_vectors([], [], 0)
    with pytest.raises(NotImplementedError):
        fast_subst.nvi_mat([], [], 0)
    with pytest.raises(NotImplementedError):
        fast_subst.nvi_mat_test([], [], 0)
    with pytest.raises(NotImplementedError):
        fast_subst.test()
