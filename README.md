# StochMat

[![Coverage Status](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/bovet-research-group/stochmat/python-coverage-comment-action-data/endpoint.json)](https://github.com/bovet-research-group/stochmat/tree/python-coverage-comment-action-data)

## Overview

`stochmat` is a Cython-based library for fast, low-level operations on stochastic matrices.
It provides memory-efficient sparse representations (`SparseStochMat`, `SparseAutocovMat`) and optimized matrix operations, with optional Intel MKL acceleration for high-performance computing workflows.

## Quickstart

```python
import numpy as np
from scipy.sparse import csr_matrix
from stochmat import SparseStochMat

# Create from scipy sparse matrix
A = csr_matrix(np.random.rand(1000, 1000))
ssm = SparseStochMat.from_full_csr_matrix(A)

# Efficient operations
ssm.inplace_row_normalize()  # Normalize rows in-place
full = ssm.to_full_mat()     # Convert back to scipy CSR
```

## Installation

<!-- installation-start -->

**Supported Python versions:**

[![Python 3.14](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/j-i-l/809edea020e4ae8a8ffb82e5cfc2e69f/raw/python-3.14.json)](https://github.com/bovet-research-group/stochmat/actions/workflows/status.yml)
[![Python 3.13](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/j-i-l/809edea020e4ae8a8ffb82e5cfc2e69f/raw/python-3.13.json)](https://github.com/bovet-research-group/stochmat/actions/workflows/status.yml)
[![Python 3.12](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/j-i-l/809edea020e4ae8a8ffb82e5cfc2e69f/raw/python-3.12.json)](https://github.com/bovet-research-group/stochmat/actions/workflows/status.yml)
[![Python 3.11](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/j-i-l/809edea020e4ae8a8ffb82e5cfc2e69f/raw/python-3.11.json)](https://github.com/bovet-research-group/stochmat/actions/workflows/status.yml)
[![Python 3.10](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/j-i-l/809edea020e4ae8a8ffb82e5cfc2e69f/raw/python-3.10.json)](https://github.com/bovet-research-group/stochmat/actions/workflows/status.yml)

### Setting up a virtual environment

We recommend installing stochmat in a virtual environment to avoid dependency conflicts.

On macOS and Linux:

```sh
$ python -m venv .venv
$ source .venv/bin/activate
```

On Windows:

```sh
PS> python -m venv .venv
PS> .venv\Scripts\activate
```

### Installing stochmat

stochmat can be installed directly from GitHub into your virtual environment.
Simply run:

```sh
pip install git+https://github.com/bovet-research-group/stochmat.git
```

<details>
<summary><b>Alternative: Using `uv`</b></summary>

If you have [uv](https://docs.astral.sh/uv/) installed, you can use it as an alternative:

```sh
uv pip install git+https://github.com/bovet-research-group/stochmat.git
```
</details>

### Optional: Intel MKL for better performance

For optimal performance with large sparse matrices, install Intel MKL:

**Ubuntu/Debian:**
```sh
sudo apt-get update
sudo apt-get install intel-mkl
```

**macOS (via Homebrew):**
```sh
brew install intel-mkl
```

After installing MKL, install stochmat with MKL support:
```sh
pip install "stochmat[mkl] @ git+https://github.com/bovet-research-group/stochmat.git"
```

Or with uv:
```sh
uv pip install "stochmat[mkl] @ git+https://github.com/bovet-research-group/stochmat.git"
```

> **Important — `[mkl]` is a hard runtime dependency.**
> Installing `stochmat[mkl]` pulls in
> [`sparse-dot-mkl`](https://pypi.org/project/sparse-dot-mkl/), which in turn
> requires Intel MKL shared libraries (e.g. `libmkl_rt.so`) to be loadable
> at runtime. **stochmat will refuse to import** (raising `ImportError` with
> an actionable message) if `sparse_dot_mkl` is installed but the MKL
> libraries are missing.
>
> If you build stochmat from source with `[mkl]`, ensure MKL is installed
> on the build/host machine *before* running `pip install`. If you do not
> need MKL acceleration, install plain `stochmat` (no extra) and SciPy will
> be used as a transparent fallback for sparse matrix products.

### Runtime backend flags

`stochmat` ships two compiled Cython extensions
(`stochmat._cython_sparse_stoch` and `stochmat.fast`) and integrates the
optional `sparse_dot_mkl` backend. Each has a transparent pure-Python
fallback. The `stochmat.backends` submodule exposes three boolean
attributes that report which backend is active in the current process:

| Attribute | `True` when… | Fallback when `False` |
|---|---|---|
| `stochmat.backends.cython_sparse_stoch` | the compiled `_cython_sparse_stoch` extension loaded successfully | pure-Python `_cython_subst` (functionally complete, slower) |
| `stochmat.backends.fast` | the compiled `fast` extension loaded successfully | pure-Python `fast_subst` — note that `nvi_parallel`, `nvi_vectors`, and `nvi_mat` raise `NotImplementedError` in the fallback |
| `stochmat.backends.mkl` | the `[mkl]` extra is installed *and* MKL native libraries are loadable | SciPy's native sparse matmul (`A @ B`) |

Quick check:

```python
import stochmat
print(stochmat.backends.summary())
# {'cython_sparse_stoch': True, 'fast': True, 'mkl': False}
```

When a compiled extension cannot be loaded, a warning is emitted via the
`stochmat.backends` logger. See the MKL section above for the
fail-fast behavior specific to `[mkl]`.

<details>
<summary><b>Development installation</b></summary>

You can also clone the repository and install the package from your local copy.
This is the recommended strategy if you intend to work on the source code,
allowing you to modify the files in-place.

#### Using pip

1. Clone this repository
2. `cd` into the repository
3. Install in editable mode with all optional dependencies:

```sh
python -m pip install -e ".[mkl]" --config-settings=editable.mode=redirect
```

To include testing and documentation dependencies:
```sh
python -m pip install -e ".[mkl]" --config-settings=editable.mode=redirect
pip install pytest pytest-cov ruff sphinx sphinx-autoapi
```

#### Using uv (fallback)

1. Clone this repository
2. `cd` into the repository
3. Sync all dependencies including extras and development groups:

```sh
uv sync --all-extras --all-groups
```

This installs the package in editable mode with MKL support, testing tools, and documentation dependencies.

**Note:** Building from source requires a C++ compiler and Python development headers.
On Ubuntu/Debian: `sudo apt-get install build-essential python3-dev`

</details>

#### Running tests with Cython coverage

To measure coverage including Cython code (adds ~10-20% test runtime overhead):

```sh
CYTHON_COVERAGE=1 uv sync --all-extras --all-groups
uv run pytest --cov=stochmat --cov-report=html
```

Open  `htmlcov/index.html` to view the coverage report.

## Using stochmat as a dependency in another package

If you build a downstream package (call it `downpkg`) on top of `stochmat`
and want to give *your* users the same opt-in MKL story
(`pip install downpkg` → SciPy fallback; `pip install downpkg[mkl]` →
MKL-accelerated end-to-end), the layout below is the recommended
template.

### The constraint that drives the layout

`stochmat[mkl]` is a **fail-fast hard runtime dependency** (see the MKL
section above): if `sparse_dot_mkl` is importable in the environment but
the Intel MKL shared libraries cannot be loaded, `import stochmat` (and
therefore `import downpkg`) raises `ImportError` immediately. Two
consequences:

1. **Never pull `sparse-dot-mkl` (or `stochmat[mkl]`) into a dependency
   set that runs on machines without MKL system libs** — including
   `testing` / `dev` groups installed in pure-Python CI jobs. Same trap
   `stochmat`'s own `pyproject.toml` documents.
2. **Your downstream `[mkl]` extra must pull `stochmat[mkl]`, not just
   `sparse-dot-mkl`.** Going through stochmat's extra is what activates
   the MKL probe code path and avoids double-declaring the dependency.

### `pyproject.toml` (PEP 621)

```toml
[project]
name = "downpkg"
requires-python = ">=3.10"
dependencies = [
    "stochmat>=X.Y",   # base: SciPy/numpy fallback path
    # ...other base deps
]

[project.optional-dependencies]
# Mirror stochmat's extra name 1:1 so users learn one convention.
# This is the ONLY place sparse_dot_mkl enters downpkg's dep graph.
mkl = [
    "stochmat[mkl]>=X.Y",
]

[dependency-groups]
testing = [
    # IMPORTANT: do NOT include "stochmat[mkl]" or "sparse-dot-mkl" here.
    # The pure-Python CI job installs this group on a runner without MKL
    # libs; pulling sparse_dot_mkl would crash `import stochmat` at
    # collection time (stochmat's MKL probe is fail-fast).
    "pytest>=8",
    # ...
]

testing-mkl = [
    { include-group = "testing" },
    "stochmat[mkl]>=X.Y",     # opt-in MKL test job
]

dev = [
    { include-group = "testing" },
    # NOT testing-mkl by default — let developers opt in explicitly so a
    # fresh `uv sync` on a machine without MKL libs doesn't fail.
]
```

The version requirement is repeated in `dependencies` and in the `mkl`
extra on purpose — the extra is purely additive (it just turns the bare
requirement into one carrying `[mkl]`); pip / uv intersect them and pick
a single resolution. Pinning the version *only* inside the extra
silently weakens the base requirement when users install without
`[mkl]`.

### Runtime usage (Pattern A, recommended)

For the common case where `downpkg` is a thin wrapper that delegates to
`stochmat.sparse_matmul`, `stochmat.sparse_gram_matrix`, etc., **no
downstream code is required** to make MKL work. Those entry points read
`stochmat.backends.mkl` at call time, so users get acceleration purely
by installing `downpkg[mkl]`:

```python
import downpkg
import stochmat
print(stochmat.backends.summary())
# {'cython_sparse_stoch': True, 'fast': True, 'mkl': True}
```

<details>
<summary><b>Pattern B: re-export / aggregate the backend report</b></summary>

If `downpkg` adds its *own* optional accelerators (e.g. a Numba code
path) and wants a single diagnostic surface, expose a
`downpkg.backends` submodule that defers to `stochmat.backends`:

```python
# downpkg/backends.py
"""Active backend report for downpkg."""
import stochmat


def summary() -> dict[str, bool]:
    return {
        "stochmat_cython_sparse_stoch": stochmat.backends.cython_sparse_stoch,
        "stochmat_fast": stochmat.backends.fast,
        "stochmat_mkl": stochmat.backends.mkl,
        # downpkg-specific accelerators here, e.g.:
        # "downpkg_numba": _numba_loaded,
    }


__all__ = ["summary"]
```

Avoid binding `mkl = stochmat.backends.mkl` at module top-level — that
captures a snapshot at import time and breaks if downstream tests
monkey-patch the flag. Always look it up dynamically (as in `summary()`
above) so the source of truth (`stochmat.backends.mkl`) stays
authoritative.

</details>

<details>
<summary><b>Note on the Cython extensions (no equivalent extra)</b></summary>

The MKL story is install-time and **declarative** (an extra in
`pyproject.toml`); the Cython story is build-time and **imperative**
(an environment variable in the shell). They are not symmetric, and
downstream `pyproject.toml` has nothing to declare for the Cython side.

- **Wheels first.** When stochmat publishes pre-built wheels for the
  user's platform/Python version, no C++ toolchain is required and
  `pip install downpkg` just works. The discussion below only matters
  for sdist installs (unsupported platforms, `--no-binary`, dev
  installs from git). For build-from-source prerequisites, see the
  "Note" at the end of the *Development installation* block above
  (`build-essential python3-dev` on Ubuntu/Debian).
- **The pure-Python escape hatch is an env var, not an extra.** End
  users who want to skip the C++ build set
  `STOCHMAT_BUILD_EXTENSIONS=0` *before* `pip install`; this flows
  directly into stochmat's scikit-build-core / CMake build regardless
  of whether the install was triggered transitively by `downpkg`.
  Build-time env vars cannot be encoded in
  `[project.optional-dependencies]`, so downstream cannot offer this
  as a `downpkg[no-ext]` extra — your README can only forward the
  knob to your users with a one-liner.
- **Caveat for downstream code that uses `nvi_*`.** The pure-Python
  fallback `stochmat.fast_subst` raises `NotImplementedError` for
  `nvi_parallel`, `nvi_vectors`, and `nvi_mat`. If `downpkg` calls
  any of those, a no-extensions install will work for everything
  *except* those code paths; either skip them under
  `not stochmat.backends.fast` or implement your own fallback.
- **CI mirroring.** Add a third row to the CI table to validate the
  fallback path:

  | Job | Install | Purpose |
  |---|---|---|
  | `test-no-ext` | `STOCHMAT_BUILD_EXTENSIONS=0 pip install -e ".[testing]"` | Validates downstream code under stochmat's pure-Python fallback (`_cython_subst` / `fast_subst`). Tests touching `nvi_*` must skip when `not stochmat.backends.fast`. |

</details>

### Downstream README — copy-down warning

Because the fail-fast behavior propagates through `import stochmat` into
every downstream import, `downpkg`'s own README should carry the same
caveat:

> **`downpkg[mkl]` is a hard runtime dependency.** It pulls
> `stochmat[mkl]`, which requires Intel MKL shared libraries to be
> loadable at runtime. `import downpkg` will raise `ImportError` if
> `sparse_dot_mkl` is installed but MKL libs are missing. If you do not
> need MKL, install plain `downpkg` and SciPy will be used as a
> transparent fallback.

### Mirroring CI

Two parallel jobs, same shape as stochmat's own workflow:

| Job | Install | Purpose |
|---|---|---|
| `test-cpu` | `uv sync` (no extras) or `pip install -e ".[testing]"` | Validates the SciPy-fallback path; no MKL libs on the runner. |
| `test-mkl` | `apt-get install intel-mkl` *then* `uv sync --group testing-mkl` or `pip install -e ".[mkl]" -e ".[testing]"` | Validates the MKL-accelerated path; system libs installed before pip. |

The `test-cpu` job must **not** have `sparse-dot-mkl` in its lockfile
or install set, for the same reason stochmat's own pure-Python CI job
does not.
