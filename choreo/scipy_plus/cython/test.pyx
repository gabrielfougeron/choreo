'''
ODE.pyx : Defines ODE-related things I designed I feel ought to be in scipy ... but faster !


'''

import os
cimport scipy.linalg.cython_blas
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
    double x,
    double *res,
) noexcept nogil:

    cdef Py_ssize_t i
    cdef Py_ssize_t size = 1
    cdef double val

    for i in range(size):
        
        val = (i+1) * x
        res[i] = csin(val)

cdef inline void single_cy_fun_memoryview(
    double x,
    double[::1] res,
) noexcept nogil:

    single_cy_fun_pointer(x,&res[0])

cdef inline double single_cy_fun_oneval(
    double x,
) noexcept nogil:

    cdef double res
    single_cy_fun_pointer(x,&res)
    return res

def single_py_fun(double x):

    cdef double[::1] res = np.empty((1),dtype=np.float64)
    single_cy_fun_memoryview(x, res)
    return res

cdef int mul_size = 10
mul_size_py = mul_size

cdef inline void mul_cy_fun_pointer(
    double x,
    double *res,
) noexcept nogil:

    cdef Py_ssize_t i
    cdef double val

    for i in range(mul_size):
        
        val = (i+1) * x
        res[i] = csin(val)

cdef inline void mul_cy_fun_memoryview(
    double x,
    double[::1] res,
) noexcept nogil:

    mul_cy_fun_pointer(x,&res[0])


def mul_py_fun(double x):

    cdef double[::1] res = np.empty((mul_size),dtype=np.float64)
    mul_cy_fun_memoryview(x, res)
    return np.asarray(res)


# cdef inline void single_cy_fun_pointer_tx(
#     double t,
#     double x,
#     double *res,
# ) noexcept nogil:
# 
#     cdef Py_ssize_t i
#     cdef Py_ssize_t size = 1
#     cdef double val
# 
#     for i in range(size):
#         
#         val = (i+1) * x
#         res[i] = csin(t*val)
# 
# cdef inline void single_cy_fun_memoryview_tx(
#     double t,
#     double x,
#     double[::1] res,
# ) noexcept nogil:
# 
#     single_cy_fun_pointer_tx(t,x,&res[0])
# 
# cdef inline double single_cy_fun_oneval_tx(
#     double t,
#     double x,
# ) noexcept nogil:
# 
#     cdef double res
#     single_cy_fun_pointer_tx(t,x,&res)
#     return res
# 
# def single_py_fun_tx(double t, double x):
# 
#     cdef double[::1] res = np.empty((1),dtype=np.float64)
#     single_cy_fun_memoryview_tx(t,x, res)
#     return res

cdef inline void mul_cy_fun_pointer_tx(
    double t    ,
    double *x   ,
    double *res ,
) noexcept nogil:

    cdef Py_ssize_t i
    cdef double val

    for i in range(mul_size):

        res[i] = csin(t*(i+1) * x[i])

cdef inline void mul_cy_fun_memoryview_tx(
    double t        ,
    double[::1] x   ,
    double[::1] res ,
) noexcept nogil:

    mul_cy_fun_pointer_tx(t, &x[0], &res[0])


def mul_py_fun_tx(double t, double[::1] x):

    cdef double[::1] res = np.empty((mul_size),dtype=np.float64)
    mul_cy_fun_memoryview_tx(t,x, res)
    return np.asarray(res)

@cython.cdivision(True)
cpdef inplace_taylor_poly(double[:] v, x):

    cdef Py_ssize_t i
    cdef double cur_term = 1.

    v[0] = cur_term

    for i in range(1,v.shape[0]):

        cur_term = cur_term * (x / i)

        v[i] = cur_term

        