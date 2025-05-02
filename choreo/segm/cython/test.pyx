'''
test functions


'''

import os
cimport scipy.linalg.cython_blas
from choreo.scipy_plus.cython.blas_consts cimport *
from libc.stdlib cimport malloc, free, rand
cdef extern from "limits.h":
    int RAND_MAX

from libc.math cimport sin as csin
from libc.math cimport exp as cexp
from libc.math cimport log2 as clog2
from libc.math cimport roundf as cround
from libc.math cimport pow as cpow

import numpy as np
cimport numpy as np
np.import_array()

cimport cython

cpdef void AssertFalse():
    assert False

# y'' = -y
cdef void ypp_eq_my_c_fun_memoryview(
    double t        ,
    double[::1] x   ,
    double[::1] res ,
) noexcept nogil:

    cdef int n = res.shape[0]

    scipy.linalg.cython_blas.dcopy(&n,&x[0],&int_one,&res[0],&int_one)

cdef void ypp_eq_my_c_gun_memoryview(
    double t        ,
    double[::1] v   ,
    double[::1] res ,
) noexcept nogil:

    cdef int n = res.shape[0]

    scipy.linalg.cython_blas.dcopy(&n,&v[0],&int_one,&res[0],&int_one)
    scipy.linalg.cython_blas.dscal(&n,&minusone_double,&res[0],&int_one)

cdef void ypp_eq_my_c_fun_memoryview_vec(
    double[::1] t       ,
    double[:,::1] x     ,
    double[:,::1] res   ,
) noexcept nogil:

    cdef int n = res.shape[0] * res.shape[1]

    scipy.linalg.cython_blas.dcopy(&n,&x[0,0],&int_one,&res[0,0],&int_one)

cdef void ypp_eq_my_c_gun_memoryview_vec(
    double[::1] t       ,
    double[:,::1] v     ,
    double[:,::1] res   ,
) noexcept nogil:

    cdef int n = res.shape[0] * res.shape[1]

    scipy.linalg.cython_blas.dcopy(&n,&v[0,0],&int_one,&res[0,0],&int_one)
    scipy.linalg.cython_blas.dscal(&n,&minusone_double,&res[0,0],&int_one)

# y'' = t*y
cdef void ypp_eq_ty_c_fun_memoryview(
    double t        ,
    double[::1] x   ,
    double[::1] res ,
) noexcept nogil:

    cdef int n = res.shape[0]

    scipy.linalg.cython_blas.dcopy(&n,&x[0],&int_one,&res[0],&int_one)

cdef void ypp_eq_ty_c_gun_memoryview(
    double t        ,
    double[::1] v   ,
    double[::1] res ,
) noexcept nogil:

    cdef int n = res.shape[0]

    scipy.linalg.cython_blas.dcopy(&n,&v[0],&int_one,&res[0],&int_one)
    scipy.linalg.cython_blas.dscal(&n,&t,&res[0],&int_one)

cdef void ypp_eq_ty_c_fun_memoryview_vec(
    double[::1] t            ,
    double[:,::1] x     ,
    double[:,::1] res   ,
) noexcept nogil:

    cdef int n = res.shape[0] * res.shape[1]

    scipy.linalg.cython_blas.dcopy(&n,&x[0,0],&int_one,&res[0,0],&int_one)

cdef void ypp_eq_ty_c_gun_memoryview_vec(
    double[::1] t            ,
    double[:,::1] v     ,
    double[:,::1] res   ,
) noexcept nogil:

    cdef int n = res.shape[0] * res.shape[1]
    cdef Py_ssize_t i

    scipy.linalg.cython_blas.dcopy(&n,&v[0,0],&int_one,&res[0,0],&int_one)

    n = res.shape[1]

    for i in range(res.shape[0]):
        scipy.linalg.cython_blas.dscal(&n,&t[i],&res[i,0],&int_one)

cdef void Wallis_c_fun_memoryview(
    double x        ,
    double[::1] res ,
) noexcept nogil:

    cdef Py_ssize_t i
    res[0] = 1.
    cdef double s = csin(x)
    for i in range(1,res.shape[0]):
        res[i] = res[i-1] * s



cdef void Wallis7_c_inplace_array_cython(
    double x,
    double *res,
) noexcept nogil:

    res[0] = csin(x)

cdef void Wallis7_c_inplace_memoryview_cython(
    double x        ,
    double[::1] res ,
) noexcept nogil:

    res[0] = csin(x)






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
cpdef inplace_taylor_poly(double[::1] v, double x):

    cdef Py_ssize_t i
    cdef double cur_term = 1.

    v[0] = cur_term

    for i in range(1,v.shape[0]):

        cur_term = cur_term * (x / i)

        v[i] = cur_term

@cython.cdivision(True)
cpdef inplace_taylor_poly_perm(double[::1] v, double x, Py_ssize_t[::1] perm):

    cdef Py_ssize_t i
    cdef double cur_term = 1.

    v[perm[0]] = cur_term

    for i in range(1,v.shape[0]):

        cur_term = cur_term * (x / i)

        v[perm[i]] = cur_term


@cython.cdivision(True)
cdef void exp_fun(double x, double* res, void* param):

    cdef Py_ssize_t test_ndim = <Py_ssize_t> (<Py_ssize_t*>param)[0]
    cdef Py_ssize_t i

    for i in range(test_ndim):
        res[i] = i * cexp(i*x)
