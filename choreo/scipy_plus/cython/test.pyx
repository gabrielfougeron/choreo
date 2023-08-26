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



cdef inline void cy_fun_pointer(
    const double x,
    double *res,
) nogil noexcept:

    cdef Py_ssize_t i
    cdef Py_ssize_t size = 1
    cdef double val

    for i in range(size):
        
        val = (i+1) * x
        res[i] = csin(val)

cdef inline void cy_fun_memoryview(
    const double x,
    double[::1] res,
) nogil noexcept:

    cy_fun_pointer(x,&res[0])

cdef inline double cy_fun_oneval(
    const double x,
) nogil noexcept:

    cdef double res
    cy_fun_pointer(x,&res)
    return res

cpdef double[::1] py_fun(const double x):

    cdef double[::1] res = np.empty((10),dtype=np.float64)
    cy_fun_memoryview(x, res)
    return res


