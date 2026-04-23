try:
    # try to import version (provided by hatch (see pyproject.toml)
    from ._version import __version__
except ImportError:
    # Fallback if the package wasn't installed properly
    __version__ = "unknown"

from .sparse_stoch_mat import (
    inplace_csr_matmul_diag,
    inplace_csr_row_normalize,
    inplace_diag_matmul_csr,
    sparse_matmul,
    sparse_gram_matrix,
    SparseAutocovMat,
    SparseStochMat
)
