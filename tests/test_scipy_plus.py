""" Tests for a few numerical methods that could be a part of scipy.

.. autosummary::
    :toctree: _generated/
    :template: tests-formatting/base.rst
    :nosignatures:

    test_random_orthogonal_matrix
    test_nullspace
    test_blas_matmul
    test_blas_matvecmul
    test_kepler
    test_cs_angle
    test_DecomposeRotation_nD

"""

import pytest
from .test_config import *
import numpy as np
import choreo
import random

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
def test_blas_matvecmul(float64_tols, m, n):
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
            
@ParametrizeDocstrings
@RepeatTest(100)
def test_cs_angle(float64_tols_strict):
    """ Tests cos sin to angle.
    """
    
    tol = 3 * float64_tols_strict.atol
    
    angle_exact = 2*np.pi * np.random.random()
    c = np.cos(angle_exact)
    s = np.sin(angle_exact)   
    
    angle_res = choreo.scipy_plus.cs_to_angle(c,s)
    
    assert abs(c - np.cos(angle_res)) < tol
    assert abs(s - np.sin(angle_res)) < tol
    assert abs(angle_exact - angle_res) < tol
            
@ParametrizeDocstrings
@RepeatTest(100)
@pytest.mark.parametrize("n", Dense_linalg_dims)
def test_DecomposeRotation_nD(float64_tols, n):
    """ Tests decomposition of rotation in nD
    """
    
    q = n // 2
    
    P = choreo.scipy_plus.random_orthogonal_matrix(n)
    
    D_init = np.zeros((n,n), dtype=np.float64)
    
    n2DBlocks = random.randint(0, q)
    n1DBlocks = n - 2*n2DBlocks
    angles = 100* np.random.random(n2DBlocks) # angles == 0 mod pi will fail
    
    print(n2DBlocks, n1DBlocks)
    
    angles_norm = angles - (2*np.pi) * (angles // (2*np.pi))
    for i in range(n2DBlocks):
        if angles_norm[i] > np.pi:
            angles_norm[i] = 2*np.pi - angles_norm[i]
    
    angles_norm = np.sort(angles_norm)
    
    for i in range(n2DBlocks):
        D_init[2*i:2*(i+1),2*i:2*(i+1)] = choreo.scipy_plus.angle_to_2D_rot(angles[i])
 
    for i in range(n1DBlocks):
        D_init[n-1-i,n-1-i] = 1
 
    if np.random.random() < 0.5:
        refl = 1.
    else:
        refl = -1.
    
    if n1DBlocks > 0:
        irefl = random.randint(0, n1DBlocks-1)
        D_init[n-1-irefl,n-1-irefl] *= refl
    
    Mat = P @ D_init @ P.T
    
    cs_angles, subspace_dim, vr = choreo.scipy_plus.DecomposeRotation(Mat, eps=1e-12)
    
    print(np.linalg.norm(np.matmul(vr, vr.T) - np.identity(n)))
    assert np.allclose(np.matmul(vr, vr.T), np.identity(n), rtol = 10*float64_tols.rtol, atol = 10*float64_tols.atol) 
    
    det = np.linalg.det(vr)
    
    assert abs(abs(det) - 1) < float64_tols.atol
    assert abs(det - 1) < float64_tols.atol

    angles_res = []
    D_res = np.zeros((n,n), dtype=np.float64)
    i = 0
    n_2D = 0
    for d in subspace_dim:
        
        if d == 2:
            
            angle_res = choreo.scipy_plus.cs_to_angle(cs_angles[i], cs_angles[i+1])
            angles_res.append(angle_res)
            
            D_res[i:i+2,i:i+2] = choreo.scipy_plus.angle_to_2D_rot(angle_res)

            n_2D += 1
        
        elif d == 1:
            refl_res = cs_angles[i]
            D_res[i,i] = refl_res
            
        else:
            raise ValueError("This should never happen")
        
        i += d

    print(np.linalg.norm(np.matmul(Mat, vr) - np.matmul(vr, D_res)))
    assert np.allclose(np.matmul(Mat, vr), np.matmul(vr, D_res), rtol = float64_tols.rtol, atol = float64_tols.atol) 

    if len(angles_res) > 0:
        if angles_res[0] > np.pi:
            angles_res[0] = 2*np.pi - angles_res[0]
    
    angles_res = np.sort(np.array(angles_res))

    print(np.linalg.norm(angles_norm - angles_res))    
    assert np.allclose(angles_norm, angles_res, rtol = float64_tols.rtol, atol = float64_tols.atol) 
    assert n_2D == n2DBlocks
    
    