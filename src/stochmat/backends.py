"""Runtime backend probes for stochmat.

This submodule is the **single source of truth** for which optional
acceleration backends are active in the current process. It performs the
import probes for the two compiled Cython extensions
(``stochmat._cython_sparse_stoch`` and ``stochmat.fast``) and for the
optional ``sparse_dot_mkl`` integration, exposing the result as three
public boolean flags:

- :data:`cython_sparse_stoch` -- ``True`` if the compiled
  ``_cython_sparse_stoch`` extension loaded; ``False`` if the package
  fell back to the pure-Python :mod:`stochmat._cython_subst` substitute.
- :data:`fast` -- ``True`` if the compiled ``fast`` extension loaded;
  ``False`` if the package fell back to :mod:`stochmat.fast_subst` (note
  that the fallback raises ``NotImplementedError`` for the parallel /
  vector NVI variants).
- :data:`mkl` -- ``True`` if the ``[mkl]`` extra is installed *and* the
  Intel MKL shared libraries were loadable.

The MKL probe is **fail-fast** when ``sparse_dot_mkl`` is installed but
the underlying MKL native libraries cannot be loaded: importing this
module (and therefore ``stochmat`` itself) raises ``ImportError`` with
an actionable message rather than silently falling back. See the README
section "Optional: Intel MKL for better performance" for details.

Diagnostics
-----------

Use :func:`summary` for a one-shot view suitable for bug reports or
debug logging::

    >>> import stochmat
    >>> stochmat.backends.summary()
    {'cython_sparse_stoch': True, 'fast': True, 'mkl': False}

Internals
---------

Private attributes ``_css``, ``_fast``, ``_dot_product_mkl`` and
``_gram_matrix_mkl`` hold the resolved module / function objects (real
or fallback) and are consumed by :mod:`stochmat.sparse_stoch_mat` to
dispatch to the active backend at call time. They are not part of the
public API.
"""
from __future__ import annotations

import importlib.util as _importlib_util
import logging

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cython _cython_sparse_stoch extension
# ---------------------------------------------------------------------------
try:
    from . import _cython_sparse_stoch as _css
    cython_sparse_stoch: bool = True
except ImportError:
    _logger.warning(
        "Could not load compiled cython extension "
        "'%s._cython_sparse_stoch'; falling back to the pure-Python "
        "substitute. Some functionality may be slower or unavailable.",
        __package__,
    )
    from . import _cython_subst as _css  # type: ignore[no-redef]
    cython_sparse_stoch = False

# ---------------------------------------------------------------------------
# Cython fast extension
# ---------------------------------------------------------------------------
try:
    from . import fast as _fast
    fast: bool = True
except ImportError:
    _logger.warning(
        "Could not load compiled cython extension "
        "'%s.fast'; falling back to the pure-Python substitute. "
        "Some functionality may be slower.",
        __package__,
    )
    from . import fast_subst as _fast  # type: ignore[no-redef]
    fast = False

# ---------------------------------------------------------------------------
# Optional MKL (sparse_dot_mkl) integration
# ---------------------------------------------------------------------------
mkl: bool = False
_dot_product_mkl = None
_gram_matrix_mkl = None

# Detect whether the user opted into the optional ``[mkl]`` extra by
# checking whether ``sparse_dot_mkl`` is *installed* (independent of
# whether its native MKL libraries can be loaded). We intentionally use
# ``find_spec`` here rather than a bare ``try/except ImportError``
# because ``sparse_dot_mkl``'s own ``__init__`` raises ``ImportError``
# when the MKL shared libraries cannot be loaded -- we must distinguish
# that case from "package not installed at all".
if _importlib_util.find_spec("sparse_dot_mkl") is not None:
    # User opted into the [mkl] extra. MKL system libraries are now a HARD
    # dependency: fail fast with an actionable message if they are missing.
    try:
        from sparse_dot_mkl import (
            dot_product_mkl as _dot_product_mkl,
            gram_matrix_mkl as _gram_matrix_mkl,
        )
    except (ImportError, OSError) as _mkl_exc:
        raise ImportError(
            "stochmat was installed with the [mkl] extra, which requires "
            "Intel MKL shared libraries to be loadable at runtime, but "
            "'sparse_dot_mkl' could not be imported "
            f"({type(_mkl_exc).__name__}: {_mkl_exc}).\n"
            "Either:\n"
            "  - install Intel MKL system libraries (e.g. "
            "'conda install mkl', the Intel oneAPI base toolkit, or your "
            "distribution's MKL package such as 'apt-get install "
            "intel-mkl'), or\n"
            "  - reinstall stochmat WITHOUT the [mkl] extra "
            "(e.g. 'pip install stochmat')."
        ) from _mkl_exc
    mkl = True


def summary() -> dict[str, bool]:
    """Return a snapshot of the active backends.

    Useful for one-line diagnostics in bug reports or debug logs::

        >>> import stochmat
        >>> stochmat.backends.summary()
        {'cython_sparse_stoch': True, 'fast': True, 'mkl': False}
    """
    return {
        "cython_sparse_stoch": cython_sparse_stoch,
        "fast": fast,
        "mkl": mkl,
    }


__all__ = ["cython_sparse_stoch", "fast", "mkl", "summary"]
