'''
ODE.pyx : Defines ODE-related things I designed I feel ought to be in scipy ... but faster !


'''

import os
cimport scipy.linalg.cython_blas
cimport scipy.linalg.cython_lapack
from libc.stdlib cimport malloc, free

import numpy as np
cimport numpy as np
np.import_array()

cimport cython

from libc.math cimport pow as cpow
from libc.math cimport fabs as cfabs
from libc.math cimport cos as ccos
from libc.math cimport sin as csin
from libc.math cimport sqrt as csqrt
from libc.math cimport isnan as cisnan
from libc.math cimport isinf as cisinf



cdef inline void single_cy_fun_pointer(
    const double x,
    double *res,
) nogil noexcept:

    cdef Py_ssize_t i
    cdef Py_ssize_t size = 1
    cdef double val

    for i in range(size):
        
        val = (i+1) * x
        res[i] = csin(val)

cdef inline void single_cy_fun_memoryview(
    const double x,
    double[::1] res,
) nogil noexcept:

    single_cy_fun_pointer(x,&res[0])

cdef inline double single_cy_fun_oneval(
    const double x,
) nogil noexcept:

    cdef double res
    single_cy_fun_pointer(x,&res)
    return res

cpdef double[::1] single_py_fun(const double x):

    cdef double[::1] res = np.empty((1),dtype=np.float64)
    single_cy_fun_memoryview(x, res)
    return res


cdef int mul_size = 10
mul_size_py = mul_size

cdef inline void mul_cy_fun_pointer(
    const double x,
    double *res,
) nogil noexcept:

    cdef Py_ssize_t i
    cdef double val

    for i in range(mul_size):
        
        val = (i+1) * x
        res[i] = csin(val)

cdef inline void mul_cy_fun_memoryview(
    const double x,
    double[::1] res,
) nogil noexcept:

    mul_cy_fun_pointer(x,&res[0])

cpdef double[::1] mul_py_fun(const double x):

    cdef double[::1] res = np.empty((mul_size),dtype=np.float64)
    mul_cy_fun_memoryview(x, res)
    return res
