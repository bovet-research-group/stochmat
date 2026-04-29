"""Stochmat - sparse stochastic matrix utilities.

This package provides efficient data structures and routines for working with
sparse stochastic (row-normalized) matrices, including:

- :class:`SparseStochMat`: a memory-efficient sparse stochastic matrix
  representation that avoids materializing identity rows.
- :class:`SparseAutocovMat`: a sparse representation of autocovariance-style
  matrices used in flow-stability community detection.
- In-place helpers (:func:`inplace_csr_matmul_diag`,
  :func:`inplace_diag_matmul_csr`, :func:`inplace_csr_row_normalize`) and
  matrix products (:func:`sparse_matmul`, :func:`sparse_gram_matrix`) that
  optionally dispatch to compiled (Cython / MKL) backends.

The package transparently falls back to pure-Python implementations when the
compiled extensions (``stochmat.fast``, ``stochmat._cython_sparse_stoch``) are
unavailable. The optional MKL acceleration (``sparse_dot_mkl``), in contrast,
is treated as a **hard runtime dependency once opted into**: if
``sparse_dot_mkl`` is installed but the underlying Intel MKL shared libraries
cannot be loaded, ``import stochmat`` will raise ``ImportError`` with an
actionable message rather than silently falling back. See the README section
"Optional: Intel MKL for better performance" for installation guidance.

The module-level flags :data:`USE_FAST` and :data:`USE_SPARSE_DOT_MKL` indicate
which backends are active in the current process.
"""
try:
    # try to import version (provided by hatch (see pyproject.toml)
    from ._version import __version__
except ImportError:
    # Fallback if the package wasn't installed properly
    __version__ = "unknown"

from .sparse_stoch_mat import (  # noqa: F401
    USE_FAST,
    USE_SPARSE_DOT_MKL,
    inplace_csr_matmul_diag,
    inplace_csr_row_normalize,
    inplace_diag_matmul_csr,
    sparse_matmul,
    sparse_gram_matrix,
    SparseAutocovMat,
    SparseStochMat,
    fast,
)

__all__ = [
    "__version__",
    "USE_FAST",
    "USE_SPARSE_DOT_MKL",
    "inplace_csr_matmul_diag",
    "inplace_csr_row_normalize",
    "inplace_diag_matmul_csr",
    "sparse_matmul",
    "sparse_gram_matrix",
    "SparseAutocovMat",
    "SparseStochMat",
    "fast",
]
