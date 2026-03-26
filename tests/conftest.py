import pytest

from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, csc_matrix


@pytest.fixture(scope='function')
def get_csr_matrix_small():
    """Create an exemplary csr matrix that can be used for testing
    """
    row = np.array([0, 0, 1, 2, 3, 3])
    col = np.array([0, 2, 2, 1, 3, 4])
    data = np.array([1, 2, 3, 5, 1, 6])
    return csr_matrix((data, (row, col)), shape=(5, 5))


@pytest.fixture(scope='function')
def get_csr_matrix_large():
    """Create an exemplary stochastic (row-normalized) CSR matrix for testing."""
    size = 10000
    nbr_non_zeros = 1000

    # Generate random indices
    row = np.random.randint(0, size, size=nbr_non_zeros)
    col = np.random.randint(0, size, size=nbr_non_zeros)
    # Generate random positive weights
    data = np.random.rand(nbr_non_zeros) 
    # Construct the raw matrix
    raw_matrix = csr_matrix((data, (row, col)), shape=(size, size))
    # Calculate row sums
    row_sums = np.array(raw_matrix.sum(axis=1)).flatten()
    # Handle zero-sum rows (isolated nodes)
    zero_rows = np.where(row_sums == 0)[0]
    if len(zero_rows) > 0:
        # Convert to COO to access indices for reconstruction
        coo = raw_matrix.tocoo()
        extra_data = np.ones(len(zero_rows))
        extra_row = zero_rows
        extra_col = zero_rows
        # Concatenate arrays
        all_data = np.concatenate([coo.data, extra_data])
        all_row = np.concatenate([coo.row, extra_row])
        all_col = np.concatenate([coo.col, extra_col])
        # Reconstruct matrix
        raw_matrix = csr_matrix((all_data, (all_row, all_col)), shape=(size, size))
        # Recalculate sums
        row_sums = np.array(raw_matrix.sum(axis=1)).flatten()
    # Normalize rows
    # Convert to COO to access row indices for scaling
    coo = raw_matrix.tocoo()
    # Safe division: ensure no zeros remain (though we added self-loops, so sums > 0)
    inv_row_sums = 1.0 / row_sums
    # Scale the data: data[i] *= inv_row_sums[coo.row[i]]
    coo.data *= inv_row_sums[coo.row]
    # Convert back to CSR
    raw_matrix = csr_matrix(coo)
    # Calculate density
    density = raw_matrix.nnz / (size * size)
    return raw_matrix, density


@pytest.fixture(scope='function')
def get_csr_matrix_large_wrong():
    """Create an exemplary csr matrix that can be used for testing
    """
    size = 10000
    nbr_non_zeros = 1000
    row = np.sort(np.random.randint(0, size, size=nbr_non_zeros))
    col = np.sort(np.random.randint(0, size, size=nbr_non_zeros))
    data = np.random.randint(0, 100, size=nbr_non_zeros)
    density = nbr_non_zeros / size
    return csr_matrix((data, (row, col)), shape=(size, size)), density


@pytest.fixture(scope='function')
def cs_matrix_creator():
    """
    Factory function to create stochastic CSR or CSC matrices for testing.
    """
    def _get_matrix(nbr: int = 1,
                    size: int = 1000000,
                    nbr_non_zeros: int = 10000,
                    mode: str = 'r'):
        assert mode in ['r', 'c'], "Mode must be 'r' (CSR) or 'c' (CSC)"
        
        MatrixType = csr_matrix if mode == 'r' else csc_matrix
        
        matrices = []
        
        for _ in range(nbr):
            # 1. Generate random indices (pairs)
            row = np.random.randint(0, size, size=nbr_non_zeros)
            col = np.random.randint(0, size, size=nbr_non_zeros)
            data = np.random.random(size=nbr_non_zeros)
            
            # 2. Construct raw matrix
            mat = MatrixType((data, (row, col)), shape=(size, size))
            
            # 3. Calculate sums for normalization
            if mode == 'r':
                row_sums = np.array(mat.sum(axis=1)).flatten()
            else:
                row_sums = np.array(mat.sum(axis=0)).flatten()
            
            # 4. Handle zero-sum rows/cols (add self-loops)
            zero_indices = np.where(row_sums == 0)[0]
            
            if len(zero_indices) > 0:
                # Convert to COO to easily access row/col indices
                coo = mat.tocoo()
                
                # Create self-loop entries
                extra_data = np.ones(len(zero_indices))
                extra_row = zero_indices
                extra_col = zero_indices
                
                # Concatenate arrays
                all_data = np.concatenate([coo.data, extra_data])
                all_row = np.concatenate([coo.row, extra_row])
                all_col = np.concatenate([coo.col, extra_col])
                
                # Reconstruct matrix from COO arrays
                mat = MatrixType((all_data, (all_row, all_col)), shape=(size, size))
                
                # Recalculate sums
                if mode == 'r':
                    row_sums = np.array(mat.sum(axis=1)).flatten()
                else:
                    row_sums = np.array(mat.sum(axis=0)).flatten()

            # 5. Normalize
            # Prevent division by zero (safety check)
            row_sums[row_sums == 0] = 1.0
            inv_sums = 1.0 / row_sums
            
            # Convert to COO again to access indices for scaling
            coo = mat.tocoo()
            
            if mode == 'r':
                # Scale by row sum: data[i] /= row_sum[coo.row[i]]
                coo.data *= inv_sums[coo.row]
            else:
                # Scale by col sum: data[i] /= col_sum[coo.col[i]]
                coo.data *= inv_sums[coo.col]
            
            # Convert back to the requested format (CSR or CSC)
            mat = MatrixType(coo)
            
            matrices.append(mat)
            
        return tuple(matrices)

    return _get_matrix


@pytest.fixture(scope='function')
def SSM_matrix_creator():
    """Create an exemplary csr matrix that can be used for testing
    """
    from stochmat import SparseStochMat as SSM
    size = 1000000
    nbr_non_zeros = 1000

    def _get_matrix(nbr: int = 1,
                    size: int = size,
                    nbr_non_zeros: int = nbr_non_zeros):
        matrices = []
        for _ in range(nbr):
            row = np.sort(np.random.randint(0, size, size=nbr_non_zeros))
            col = np.sort(np.random.randint(0, size, size=nbr_non_zeros))
            data = np.random.randint(0, 1, size=nbr_non_zeros)
            _a_csr = csr_matrix((data, (row, col)), shape=(size, size))
            _a_csr.data = _a_csr.data.astype(np.float64, copy=False)
            _a_csr.indices = _a_csr.indices.astype(np.int32, copy=False)
            matrices.append(SSM.from_full_csr_matrix(_a_csr))
        return tuple(matrices)
    return _get_matrix


@pytest.fixture(scope='session')
def compare_alike():
    def compare_sparse_matrice(A, B):
        """Checks if two csr matrices describe the same matrix

        csr notation can deviate in that data and indices can be re-arranged
        within a indptr slice.
        """
        assert len(A.indptr) == len(B.indptr)
        for i in range(len(A.indptr) - 1):
            A_s, A_e = A.indptr[i:i+2]
            B_s, B_e = B.indptr[i:i+2]
            B_sorted = B.indices[B_s: B_e].argsort()
            A_sorted = A.indices[A_s: A_e].argsort()
            np.testing.assert_equal(
                A.indices[A_s:A_e][A_sorted],
                B.indices[B_s:B_e][B_sorted]
            )
            np.testing.assert_equal(
                B.data[B_s:B_e][B_sorted],
                A.data[A_s:A_e][A_sorted]
            )
    return compare_sparse_matrice


@pytest.fixture(scope='session')
def compare_SSM_args():
    def compare_SSM_args(ssm_args1, ssm_args2):
        """Checks if two csr matrices describe the same matrix

        csr notation can deviate in that data and indices can be re-arranged
        within a indptr slice.
        """
        # size
        assert ssm_args1[0] == ssm_args2[0]
        # data
        np.testing.assert_equal(ssm_args1[1], ssm_args2[1])
        # indices
        np.testing.assert_equal(ssm_args1[2], ssm_args2[2])
        # indptr
        np.testing.assert_equal(ssm_args1[3], ssm_args2[3])
        # diag val
        np.testing.assert_equal(ssm_args1[4], ssm_args2[4])
    return compare_SSM_args


@pytest.fixture(scope='session')
def probabilities_transition():
    """Create exemplary densities and transition probabilities
    """
    nbr_non_zeros = 1000
    p1 = np.ones(shape=(nbr_non_zeros), dtype=np.float64) / nbr_non_zeros
    T = np.zeros(shape=(nbr_non_zeros, nbr_non_zeros), dtype=np.float64)
    for i in range(nbr_non_zeros):
        if np.random.rand() >= 0.5:
            _t = np.random.dirichlet(np.ones(nbr_non_zeros), size=1)
        else:
            _t = np.zeros(shape=(nbr_non_zeros,), dtype=np.float64)
        T[:, i] = _t

    p2 = p1 @ T
    return p1, p2, T


@pytest.fixture(scope='session')
def propa_transproba_creator():
    """Creat an exemplary csr matrix that can be used for testing
    """
    size = 1000
    zero_col_density = 0.05
    def _get_p_tp(nbr: int = 1, size: int = size,
                  zero_col_density: float = zero_col_density):
        """Generates a tuple of p1,p2,T triplets for a network of `size` nodes
        """
        ptps = []
        for _ in range(nbr):
            p1 = np.ones(shape=(size), dtype=np.float64) / size
            T = np.zeros(shape=(size, size), dtype=np.float64)
            for i in range(size):
                if np.random.rand() < 1 - zero_col_density:
                    _t = np.random.dirichlet(np.ones(size), size=1)
                else:
                    _t = np.zeros(shape=(size,), dtype=np.float64)
                T[:, i] = _t
            p2 = p1 @ T
            ptps.append((p1, p2, T))
        return tuple(ptps)
    return _get_p_tp


def temporal_network_to_df(network: SimpleNamespace):
    """Convert a network from a namespace ot a data frame
    """
    as_df = pd.DataFrame({
        "source_nodes": network.source_nodes,
        "target_nodes": network.target_nodes,
        "starting_times": network.starting_times,
    })
    ending_times = getattr(network, 'ending_times', None)
    if ending_times is not None:
        as_df['ending_times'] = ending_times
    return as_df


@pytest.fixture(scope='session')
def get_temporal_network_df_minimal():
    simple = SimpleNamespace()
    # we assume 10 nodes, and each starting a connection in order
    simple.source_nodes = list(range(0, 10))
    # target nodes are also in order
    simple.target_nodes = list(range(1, 10)) + [1]
    simple.starting_times = [0, 0.5, 1, 2, 3, 4, 4, 5, 5, 5]
    simple.ending_times = [3, 1, 2, 7, 4, 5, 6, 6, 6, 7]
    return temporal_network_to_df(network=simple)


@pytest.fixture(params=["csr", "csc"])
def sparse_format(request):
    """Parameterize tests to run against both CSR and CSC formats."""
    return request.param


@pytest.fixture
def random_sparse_matrix(sparse_format):
    """Generates a generic random 10x8 sparse matrix and its dense equivalent."""
    np.random.seed(42)
    A_dense = np.random.rand(10, 10)
    A_dense[A_dense < 0.7] = 0.0

    if sparse_format == "csr":
        return A_dense, csr_matrix(A_dense)
    return A_dense, csc_matrix(A_dense)
