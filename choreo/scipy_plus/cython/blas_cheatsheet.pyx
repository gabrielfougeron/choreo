'''
blas_cheatsheet.pyx This file is the answer to the questions I always aks myself calling blas/lapack functions in cython: WTF am I doing ?
'''

import os
cimport scipy.linalg.cython_blas
from choreo.scipy_plus.cython.blas_consts cimport *

import numpy as np
cimport numpy as np
np.import_array()

cimport cython

cpdef np.ndarray[double, ndim=2, mode="c"] blas_matmul(
    double[:,::1] amat  ,
    double[:,::1] bmat  ,
):

    assert amat.shape[1] == bmat.shape[0]

    cdef int m = bmat.shape[1]
    cdef int n = amat.shape[0]
    cdef int k = amat.shape[1]

    cdef np.ndarray[double, ndim=2, mode="c"] res = np.empty((n,m) ,dtype=np.float64)

    # res = amat . bmat
    scipy.linalg.cython_blas.dgemm(transn,transn,&m,&n,&k,&one_double,&bmat[0,0],&m,&amat[0,0],&k,&zero_double,&res[0,0],&m)

    return res