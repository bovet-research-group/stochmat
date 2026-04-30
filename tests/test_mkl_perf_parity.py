"""Performance parity tests for the MKL backend.

Two complementary checks against the scipy.sparse fallback:

1. **Speed parity** -- ``pytest-benchmark`` records wall-clock medians for
   both backends in the two ``test_sparse_matmul_speed_{mkl,scipy}`` tests
   (recorded as informational benchmark output). A separate
   ``test_sparse_matmul_speed_parity_backstop`` test does its own
   ``time.perf_counter`` timing and asserts a generous regression backstop:
   MKL must not be more than 2x slower than scipy.
2. **Peak-allocation parity** -- ``test_sparse_matmul_peak_alloc_parity``
   measures peak heap allocation (via ``tracemalloc``) for both backends.
   Both numbers are reported via ``request.node.add_report_section`` for
   visibility; a loose backstop asserts MKL peak <= 2.5x scipy peak.

Only the two ``_speed_{mkl,scipy}`` recording tests carry the
``@pytest.mark.benchmark`` marker (they use the ``benchmark`` fixture);
the backstop and peak-allocation tests run in the regular test suite so
that regressions are caught without needing to opt into the benchmarks
job. The dedicated ``mkl-benchmarks`` CI job filters in via
``-m benchmark`` to record the medians.

All tests require the MKL backend to be active (``require_mkl`` fixture).
"""
import gc
import tracemalloc

import pytest

from stochmat import backends as _backends
from stochmat import sparse_stoch_mat as ssm


# ---------------------------------------------------------------------------
# Backend toggle helpers (mirror of test_mkl_parity.py to avoid cross-imports
# between test files)
# ---------------------------------------------------------------------------

def _matmul_with_backend(A, B, *, mkl):
    saved = _backends.mkl
    _backends.mkl = mkl
    try:
        return ssm.sparse_matmul(A, B)
    finally:
        _backends.mkl = saved


# ---------------------------------------------------------------------------
# Speed parity
# ---------------------------------------------------------------------------

@pytest.mark.benchmark(group="sparse_matmul")
def test_sparse_matmul_speed_mkl(
    require_mkl, mkl_canonical_sparse, benchmark
):
    """Record MKL median time for sparse_matmul (informational)."""
    A, B = mkl_canonical_sparse
    benchmark(_matmul_with_backend, A, B, mkl=True)


@pytest.mark.benchmark(group="sparse_matmul")
def test_sparse_matmul_speed_scipy(
    require_mkl, mkl_canonical_sparse, benchmark
):
    """Record scipy median time for sparse_matmul (informational baseline)."""
    A, B = mkl_canonical_sparse
    benchmark(_matmul_with_backend, A, B, mkl=False)


def test_sparse_matmul_speed_parity_backstop(
    require_mkl, mkl_canonical_sparse, request
):
    """Sanity backstop: MKL must not be more than 2x slower than scipy.

    This is a regression guard, not a performance assertion -- MKL is
    normally several times faster than scipy. The 2x tolerance is generous
    to avoid CI flakes from cold caches / shared runners.

    Both medians are reported via the test's stdout section.
    """
    import time

    A, B = mkl_canonical_sparse

    def _time_one(mkl, n=5):
        # warmup
        _matmul_with_backend(A, B, mkl=mkl)
        timings = []
        for _ in range(n):
            t0 = time.perf_counter()
            _matmul_with_backend(A, B, mkl=mkl)
            timings.append(time.perf_counter() - t0)
        timings.sort()
        return timings[len(timings) // 2]

    t_scipy = _time_one(mkl=False)
    t_mkl = _time_one(mkl=True)
    ratio = t_mkl / t_scipy

    msg = (
        f"sparse_matmul speed parity:\n"
        f"  scipy median = {t_scipy * 1e3:.3f} ms\n"
        f"  MKL   median = {t_mkl * 1e3:.3f} ms\n"
        f"  ratio (MKL / scipy) = {ratio:.3f}\n"
        f"  threshold = 2.0x"
    )
    print(msg)
    request.node.add_report_section("call", "speed_parity", msg)

    assert ratio <= 2.0, (
        f"MKL sparse_matmul is {ratio:.2f}x slower than scipy "
        f"(threshold 2.0x). scipy={t_scipy * 1e3:.3f}ms, "
        f"mkl={t_mkl * 1e3:.3f}ms"
    )


# ---------------------------------------------------------------------------
# Peak-allocation parity
# ---------------------------------------------------------------------------

def _measure_peak(callable_, *args, **kwargs):
    """Return peak heap allocation (bytes) during a single call."""
    gc.collect()
    tracemalloc.start()
    try:
        result = callable_(*args, **kwargs)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # Keep result alive until after measurement (tracemalloc cleared)
    del result
    gc.collect()
    return peak


def test_sparse_matmul_peak_alloc_parity(
    require_mkl, mkl_canonical_sparse, request
):
    """Compare MKL vs scipy peak heap allocation for sparse_matmul.

    Both peaks are reported as informational test output. A loose backstop
    asserts MKL peak <= 2.5x scipy peak to catch egregious regressions
    (e.g. accidental dense intermediate); transient MKL scratch buffers
    can legitimately bump the ratio above 1.0 on small-to-medium inputs.

    Named ``peak_alloc`` rather than ``memory`` so that the
    ``-k _memory`` filter used by the dedicated memory-tracing CI job
    does not pick this test up (it would conflict with memray's own
    instrumentation if both were active).
    """
    A, B = mkl_canonical_sparse

    # Warmup each backend to avoid first-call allocator overhead skewing
    # tracemalloc numbers.
    _matmul_with_backend(A, B, mkl=False)
    _matmul_with_backend(A, B, mkl=True)

    peak_scipy = _measure_peak(_matmul_with_backend, A, B, mkl=False)
    peak_mkl = _measure_peak(_matmul_with_backend, A, B, mkl=True)
    ratio = peak_mkl / max(peak_scipy, 1)

    msg = (
        f"sparse_matmul memory parity:\n"
        f"  scipy peak = {peak_scipy / 1024:.1f} KiB\n"
        f"  MKL   peak = {peak_mkl / 1024:.1f} KiB\n"
        f"  ratio (MKL / scipy) = {ratio:.3f}\n"
        f"  threshold = 2.5x"
    )
    print(msg)
    request.node.add_report_section("call", "peak_alloc_parity", msg)

    assert ratio <= 2.5, (
        f"MKL sparse_matmul peak memory is {ratio:.2f}x scipy "
        f"(threshold 2.5x). scipy={peak_scipy / 1024:.1f}KiB, "
        f"mkl={peak_mkl / 1024:.1f}KiB"
    )
