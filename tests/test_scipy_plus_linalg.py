import os
import sys

__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(__PROJECT_ROOT__)

import pytest
from test_config import *

import numpy as np
import scipy
import choreo



@ProbabilisticTest()
def test_random_orthogonal_matrix(float64_tols, Dense_linalg_dims):

    print("Generation of random orthogonal matrices.")

    for n in Dense_linalg_dims.all_geodims:

        print(f"Dimension: {n}")

        rot = choreo.scipy_plus.linalg.random_orthogonal_matrix(n)

        assert np.allclose(np.matmul(rot  , rot.T), np.identity(n), rtol = float64_tols.rtol, atol = float64_tols.atol) 
        assert np.allclose(np.matmul(rot.T, rot  ), np.identity(n), rtol = float64_tols.rtol, atol = float64_tols.atol) 


@ProbabilisticTest()
def test_nullspace(float64_tols, Dense_linalg_dims):

    print("Testing nullspace of a dense matrix.")

    for n in Dense_linalg_dims.all_geodims:

        for m in Dense_linalg_dims.all_geodims:

            print(f"Dimensions: {n}, {m}")

            P = choreo.scipy_plus.linalg.random_orthogonal_matrix(n)

            if (n == m):

                Z = choreo.scipy_plus.linalg.null_space(P)

                assert Z.shape[0] == m
                assert Z.shape[1] == 0

            for rank in range( min(n,m)+1):

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




@ProbabilisticTest()
def test_matmul(float64_tols, Dense_linalg_dims):
    
    print("Testing matrix multiplication")

    for n in Dense_linalg_dims.all_geodims:
        for k in Dense_linalg_dims.all_geodims:
            for m in Dense_linalg_dims.all_geodims:

                print(f"Dimensions: {n}, {k}, {m}")
                
                A = np.random.random((n,k))
                B = np.random.random((k,m))
                
                AB_np = np.zeros((n,m),dtype=np.float64)
                np.matmul(A,B,out=AB_np)     
                    
                AB_blas = np.zeros((n,m),dtype=np.float64)
                choreo.cython.test_blis.blas_matmul_contiguous(A,B,AB_blas)
                assert np.allclose(AB_np, AB_blas, rtol = float64_tols.rtol, atol = float64_tols.atol)     
                                
                AB_blis = np.zeros((n,m),dtype=np.float64)
                choreo.cython.test_blis.blis_matmul_contiguous(A,B,AB_blis)
                assert np.allclose(AB_np, AB_blis, rtol = float64_tols.rtol, atol = float64_tols.atol) 
            
            
@ProbabilisticTest()
def test_matmulTT(float64_tols, Dense_linalg_dims):
    
    print("Testing matrix multiplication")

    for n in Dense_linalg_dims.all_geodims:
        for k in Dense_linalg_dims.all_geodims:
            for m in Dense_linalg_dims.all_geodims:

                print(f"Dimensions: {n}, {k}, {m}")
                
                A = np.random.random((n,k))
                B = np.random.random((k,m))
                
                BTAT_np = np.zeros((m,n),dtype=np.float64)
                np.matmul(B.T,A.T,out=BTAT_np)     
                    
                BTAT_blas = np.zeros((m,n),dtype=np.float64)
                choreo.cython.test_blis.blas_matmulTT_contiguous(B,A,BTAT_blas)
                assert np.allclose(BTAT_np, BTAT_blas, rtol = float64_tols.rtol, atol = float64_tols.atol)     
                                            
@ProbabilisticTest()
def test_matmulNT(float64_tols, Dense_linalg_dims):
    
    print("Testing matrix multiplication")

    for n in Dense_linalg_dims.all_geodims:
        for k in Dense_linalg_dims.all_geodims:
            for m in Dense_linalg_dims.all_geodims:

                print(f"Dimensions: {m}, {k}, {n}")
                
                A = np.random.random((m,k))
                B = np.random.random((n,k))
                
                ABT_np = np.zeros((m,n),dtype=np.float64)
                np.matmul(A,B.T,out=ABT_np)     

                ABT_blas = np.zeros((m,n),dtype=np.float64)
                choreo.cython.test_blis.blas_matmulNT_contiguous(A,B,ABT_blas)
                
                assert np.allclose(ABT_np, ABT_blas, rtol = float64_tols.rtol, atol = float64_tols.atol)     
                                



@ProbabilisticTest()
def test_matmul_realpart(float64_tols, Dense_linalg_dims):
    
    print("Testing matrix multiplication")

    for n in Dense_linalg_dims.all_geodims:
        for k in Dense_linalg_dims.all_geodims:
            for m in Dense_linalg_dims.all_geodims:

                print(f"Dimensions: {n}, {k}, {m}")
                
                A = np.random.random((n,k)) + 1j*np.random.random((n,k))
                B = np.random.random((k,m)) + 1j*np.random.random((k,m))
                
                AB_np = np.zeros((n,m),dtype=np.complex128)
                np.matmul(A,B,out=AB_np)     
                    
                AB_blis = np.zeros((n,m),dtype=np.float64)
                choreo.cython.test_blis.blis_matmul_real(A,B,AB_blis)
                assert np.allclose(AB_np.real, AB_blis, rtol = float64_tols.rtol, atol = float64_tols.atol)     
                                
                AB_blas = np.zeros((n,m),dtype=np.float64)
                choreo.cython.test_blis.blas_matmul_real(A,B,AB_blas)
                assert np.allclose(AB_np.real, AB_blas, rtol = float64_tols.rtol, atol = float64_tols.atol) 
                                            
                AB_blas = np.zeros((n,m),dtype=np.float64)
                choreo.cython.test_blis.blas_matmul_real_copy(A,B,AB_blas)
                assert np.allclose(AB_np.real, AB_blas, rtol = float64_tols.rtol, atol = float64_tols.atol) 
            
            

