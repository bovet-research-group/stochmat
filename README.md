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
