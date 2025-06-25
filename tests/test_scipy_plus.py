""" Tests for a few numerical methods that could be a part of scipy.

.. autosummary::
    :toctree: _generated/
    :template: tests-formatting/base.rst
    :nosignatures:

    test_random_orthogonal_matrix
    test_nullspace
    test_blas_matmul

"""

import pytest
from .test_config import *
import numpy as np
import choreo

@ParametrizeDocstrings
@ProbabilisticTest()
@pytest.mark.parametrize("n", Dense_linalg_dims)
def test_random_orthogonal_matrix(float64_tols, n):
    """ Tests whether random orthogonal matrices are indeed orthogonal enough.
    """

    rot = choreo.scipy_plus.linalg.random_orthogonal_matrix(n)

    assert np.allclose(np.matmul(rot  , rot.T), np.identity(n), rtol = float64_tols.rtol, atol = float64_tols.atol) 
    assert np.allclose(np.matmul(rot.T, rot  ), np.identity(n), rtol = float64_tols.rtol, atol = float64_tols.atol) 

@ParametrizeDocstrings
@ProbabilisticTest()
@pytest.mark.parametrize("m", Dense_linalg_dims)
@pytest.mark.parametrize("n", Dense_linalg_dims)
def test_nullspace(float64_tols, n, m):
    """ Tests properties of nullspace computation.
    """

    P = choreo.scipy_plus.linalg.random_orthogonal_matrix(n)

    if (n == m):

        Z = choreo.scipy_plus.linalg.null_space(P)

        assert Z.shape[0] == m
        assert Z.shape[1] == 0

    for rank in range(min(n,m)+1):

        nullspace_dim = m - rank

        Q = choreo.scipy_plus.linalg.random_orthogonal_matrix(m)

        diag = np.random.rand(rank)
        diagmat = np.zeros((n,m))

        for i in range(rank):
            diagmat[i,i] = 2 + diag[i]

        A = np.matmul(P, np.matmul(diagmat, Q))

        Z = choreo.scipy_plus.linalg.null_space(A)

        # Dimensions
        assert Z.shape[0] == m
        assert Z.shape[1] == nullspace_dim
        
        # Nullspace property
        assert np.allclose(np.matmul(A,Z), 0, rtol = float64_tols.rtol, atol = float64_tols.atol) 

        # Orthogonality
        assert np.allclose(np.matmul(Z.T,Z), np.identity(nullspace_dim), rtol = float64_tols.rtol, atol = float64_tols.atol) 

@ParametrizeDocstrings
@ProbabilisticTest()
@pytest.mark.parametrize("m", Dense_linalg_dims)
@pytest.mark.parametrize("n", Dense_linalg_dims)
@pytest.mark.parametrize("k", Dense_linalg_dims)
def test_blas_matmul(float64_tols, m, n, k):
    """ Tests calling dgemm directly from blas for regular matmul.
    """
    
    A = np.random.random((m,k))
    B = np.random.random((k,n))

    AB_np = np.matmul(A,B)
    AB_blas = choreo.scipy_plus.cython.blas_cheatsheet.blas_matmul(A,B)
    
    print(np.linalg.norm(AB_np - AB_blas))
    assert np.allclose(AB_np, AB_blas, rtol = float64_tols.rtol, atol = float64_tols.atol) 
    
@ParametrizeDocstrings
@ProbabilisticTest()
@pytest.mark.parametrize("m", Dense_linalg_dims)
@pytest.mark.parametrize("n", Dense_linalg_dims)
def test_blas_matmul(float64_tols, m, n):
    """ Tests calling dgemv directly from blas for regular matmul.
    """
    
    A = np.random.random((m,n))
    v = np.random.random((n))

    Av_np = np.matmul(A,v)
    Av_blas = choreo.scipy_plus.cython.blas_cheatsheet.blas_matvecmul(A,v)
    
    print(np.linalg.norm(Av_np - Av_blas))
    assert np.allclose(Av_np, Av_blas, rtol = float64_tols.rtol, atol = float64_tols.atol) 
            
@ParametrizeDocstrings
def test_kepler(float64_tols_strict, float64_tols_loose):
    """ Tests Kepler solver.
    """
    
    for M in [0., 0.5, 10., 100.]:
        for ecc in [0., 0.5, 0.8, 0.95, 0.99, 0.99999]:
    
            E = choreo.scipy_plus.cython.kepler.solve(M, ecc)

            print(abs(E - ecc * np.sin(E) - M))
            assert abs(E - ecc * np.sin(E) - M) <= float64_tols_strict.atol + 2 * float64_tols_strict.rtol * M
                
            def f(x):
                Ep, cosfp, sinfp, dcosfp, dsinfp = choreo.scipy_plus.cython.kepler.kepler(x[0], ecc)
                return np.array([cosfp, sinfp])
                
            def grad_f(x, dx):
                Ep, cosfp, sinfp, dcosfp, dsinfp = choreo.scipy_plus.cython.kepler.kepler(x[0], ecc)
                return np.array([dcosfp * dx[0], dsinfp * dx[0]])
                    
            err = compare_FD_and_exact_grad(
                f               ,
                grad_f          ,
                np.array([M])   ,
                order=2         ,
                vectorize=False ,
                relative = True ,
            )
            
            print(err.min())
            assert err.min() < 100 * float64_tols_loose.rtol # precision is not great when e is near 1
            
    imax = 2**12
    for i in range(imax):
        
        M = i*2*np.pi / imax
        
        for ecc in [0., 0.5, 0.8, 0.95, 0.99, 0.99999]:
            
            E = choreo.scipy_plus.cython.kepler.solve(M, ecc)

            print(M, ecc, abs(E - ecc * np.sin(E) - M))
            assert abs(E - ecc * np.sin(E) - M) <= float64_tols_strict.atol + 2 * float64_tols_strict.rtol * M
            
