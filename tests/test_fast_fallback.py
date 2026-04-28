"""
Tests verifying that ``stochmat.fast_subst`` (pure-Python fallback) produces
results identical to the compiled Cython ``stochmat.fast`` module.

These tests assume Cython is available; both implementations are loaded via
the session-scoped ``fast_modules`` fixture.
"""
import numpy as np
import pytest


# =============================================================================
# Helpers
# =============================================================================

def _assert_same(fast_modules, fn_name, args, rtol=1e-12, atol=0.0):
    """Assert that Cython and fallback produce equivalent results."""
    fc = getattr(fast_modules.cython, fn_name)
    fs = getattr(fast_modules.fallback, fn_name)
    np.testing.assert_allclose(fc(*args), fs(*args), rtol=rtol, atol=atol)


# =============================================================================
# Test: sum_Sto
# =============================================================================

class TestSumSto:
    """Compare ``sum_Sto`` outputs."""

    def test_basic_case(self, fast_modules):
        S = np.array(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64
        )
        _assert_same(fast_modules, "sum_Sto", (S, 0, [1, 2]), rtol=1e-15)

    def test_large_community(self, fast_modules):
        rng = np.random.default_rng(42)
        S = rng.standard_normal((100, 100))
        S = (S + S.T) / 2
        _assert_same(
            fast_modules, "sum_Sto", (S, 10, list(range(20, 40))),
        )

    def test_empty_community(self, fast_modules):
        S = np.array([[1, 2], [3, 4]], dtype=np.float64)
        _assert_same(fast_modules, "sum_Sto", (S, 0, []), rtol=1e-15)
        # Empty community should reduce to S[k, k]
        assert fast_modules.cython.sum_Sto(S, 0, []) == S[0, 0]


# =============================================================================
# Test: sum_Sout
# =============================================================================

class TestSumSout:
    """Compare ``sum_Sout`` outputs."""

    def test_basic_case(self, fast_modules):
        S = np.array(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64
        )
        _assert_same(fast_modules, "sum_Sout", (S, 1, [0, 1]), rtol=1e-15)

    def test_large_community(self, fast_modules):
        rng = np.random.default_rng(123)
        S = rng.standard_normal((100, 100))
        _assert_same(
            fast_modules, "sum_Sout", (S, 25, list(range(10, 50))),
        )


# =============================================================================
# Test: compute_S
# =============================================================================

class TestComputeS:
    """Compare ``compute_S`` outputs."""

    def test_basic_case(self, fast_modules):
        p1 = np.array([0.5, 0.3, 0.2], dtype=np.float64)
        p2 = np.array([0.4, 0.3, 0.3], dtype=np.float64)
        T = np.array(
            [[0.8, 0.1, 0.1],
             [0.2, 0.7, 0.1],
             [0.1, 0.2, 0.7]], dtype=np.float64,
        )
        _assert_same(fast_modules, "compute_S", (p1, p2, T), rtol=1e-15)

    def test_large_matrices(self, fast_modules, propa_transproba_creator):
        p1, p2, T = propa_transproba_creator(nbr=1, size=1000)[0]
        _assert_same(fast_modules, "compute_S", (p1, p2, T))

    def test_rectangular_T(self, fast_modules):
        p1 = np.array([0.4, 0.6], dtype=np.float64)
        p2 = np.array([0.3, 0.4, 0.3], dtype=np.float64)
        T = np.array(
            [[0.5, 0.3, 0.2],
             [0.3, 0.4, 0.3]], dtype=np.float64,
        )
        _assert_same(fast_modules, "compute_S", (p1, p2, T), rtol=1e-15)


# =============================================================================
# Test: compute_S_0t0
# =============================================================================

class TestComputeS0t0:
    """Compare ``compute_S_0t0`` outputs."""

    def test_basic_case(self, fast_modules):
        p0 = np.array([0.5, 0.3, 0.2], dtype=np.float64)
        pt = np.array([0.4, 0.35, 0.25], dtype=np.float64)
        T = np.array(
            [[0.8, 0.1, 0.1],
             [0.2, 0.7, 0.1],
             [0.1, 0.2, 0.7]], dtype=np.float64,
        )
        _assert_same(fast_modules, "compute_S_0t0", (p0, pt, T))

    @pytest.mark.xfail(
        reason=(
            "Cython compute_S_0t0 has a division-by-zero bug at "
            "fast.pyx:108 when pt[k] == 0"
        ),
    )
    def test_large_matrices(self, fast_modules, propa_transproba_creator):
        p1, p2, T = propa_transproba_creator(nbr=1, size=100)[0]
        _assert_same(
            fast_modules, "compute_S_0t0", (p1, p2, T), rtol=1e-10,
        )


# =============================================================================
# Test: nmi (Normalized Mutual Information)
# =============================================================================

class TestNmi:
    """Compare ``nmi`` outputs."""

    def test_identical_clusterings(self, fast_modules):
        clusters = [{0, 1, 2}, {3, 4}]
        args = (clusters, clusters, 5, len(clusters), len(clusters))
        _assert_same(fast_modules, "nmi", args, rtol=1e-15)
        assert fast_modules.cython.nmi(*args) == pytest.approx(1.0)

    def test_different_clusterings(self, fast_modules):
        c1 = [{0, 1}, {2, 3, 4}]
        c2 = [{0, 1, 2}, {3, 4}]
        args = (c1, c2, 5, len(c1), len(c2))
        _assert_same(fast_modules, "nmi", args, rtol=1e-15)
        assert 0 <= fast_modules.cython.nmi(*args) <= 1

    def test_large_clusterings(self, fast_modules, make_equal_clusterings_fixture):
        c1, c2 = make_equal_clusterings_fixture(N=100, k1=5, k2=4, seed=456)
        args = (c1, c2, 100, len(c1), len(c2))
        _assert_same(fast_modules, "nmi", args)


# =============================================================================
# Test: nvi (Normalized Variation of Information)
# =============================================================================

class TestNvi:
    """Compare ``nvi`` outputs."""

    def test_identical_clusterings(self, fast_modules):
        clusters = [{0, 1, 2}, {3, 4}]
        args = (clusters, clusters, 5)
        _assert_same(fast_modules, "nvi", args, rtol=1e-15)
        assert fast_modules.cython.nvi(*args) == pytest.approx(0.0, abs=1e-10)

    def test_different_clusterings(self, fast_modules):
        c1 = [{0, 1}, {2, 3, 4}]
        c2 = [{0, 1, 2}, {3, 4}]
        args = (c1, c2, 5)
        _assert_same(fast_modules, "nvi", args, rtol=1e-15)
        assert 0 <= fast_modules.cython.nvi(*args) <= 1

    def test_large_clusterings(self, fast_modules, make_equal_clusterings_fixture):
        c1, c2 = make_equal_clusterings_fixture(N=100, k1=5, k2=4, seed=789)
        _assert_same(fast_modules, "nvi", (c1, c2, 100))

    def test_early_termination(self, fast_modules):
        # Sparse intersections trigger the early-termination branch.
        c1 = [{0, 1}, {2, 3}, {4, 5}]
        c2 = [{0, 2, 4}, {1, 3, 5}]
        _assert_same(fast_modules, "nvi", (c1, c2, 6), rtol=1e-15)


# =============================================================================
# NotImplementedError stubs
# =============================================================================

@pytest.mark.parametrize(
    "fn_name, args, match",
    [
        ("nvi_parallel",
         ([{0, 1}, {2, 3}], [{0, 2}, {1, 3}], 4, 4),
         "parallel"),
        ("nvi_vectors",
         ([np.array([0, 1]), np.array([2, 3])],
          [np.array([0, 2]), np.array([1, 3])], 4),
         "vector"),
        ("nvi_mat",
         ([{0, 1}, {2, 3}], [{0, 2}, {1, 3}], 4),
         "matrix"),
        ("nvi_mat_test",
         ([{0, 1}, {2, 3}], [{0, 2}, {1, 3}], 4),
         "test function"),
        ("test", (), "test.*function"),
    ],
)
def test_fallback_raises_not_implemented(fast_modules, fn_name, args, match):
    """Stubs in the fallback module must raise ``NotImplementedError``."""
    fn = getattr(fast_modules.fallback, fn_name)
    with pytest.raises(NotImplementedError, match=match):
        fn(*args)


# =============================================================================
# Integration: full workflows
# =============================================================================

class TestFullWorkflow:
    """Combine multiple functions in a typical pipeline."""

    def test_stability_workflow(self, fast_modules, propa_transproba_creator):
        p1, p2, T = propa_transproba_creator(nbr=1, size=50)[0]
        S_c = fast_modules.cython.compute_S(p1, p2, T)
        S_f = fast_modules.fallback.compute_S(p1, p2, T)

        k, ix_cf, ix_ci = 5, [10, 15, 20], [5, 6, 7, 8]
        np.testing.assert_allclose(
            fast_modules.cython.sum_Sto(S_c, k, ix_cf),
            fast_modules.fallback.sum_Sto(S_f, k, ix_cf),
        )
        np.testing.assert_allclose(
            fast_modules.cython.sum_Sout(S_c, k, ix_ci),
            fast_modules.fallback.sum_Sout(S_f, k, ix_ci),
        )

    def test_clustering_comparison_workflow(
        self, fast_modules, make_equal_clusterings_fixture,
    ):
        c1, c2 = make_equal_clusterings_fixture(N=50, k1=5, k2=5, seed=999)
        nmi_args = (c1, c2, 50, len(c1), len(c2))
        _assert_same(fast_modules, "nmi", nmi_args)
        _assert_same(fast_modules, "nvi", (c1, c2, 50))
