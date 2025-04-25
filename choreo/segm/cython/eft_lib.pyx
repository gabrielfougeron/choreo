'''
eft_lib.ptx : Error-free transformations
Warning: this file needs to be compiled without unsafe optimizations. No -Ofast. No -ffast-math.

Essentially follows the implementation described in [1]

[1] Ogita, T., Rump, S. M., & Oishi, S. I. (2005). Accurate sum and dot product. SIAM Journal on Scientific Computing, 26(6), 1955-1988.
'''

from libc.stdlib cimport malloc, free

import numpy as np
cimport numpy as np
np.import_array()

cimport cython

cimport scipy.linalg.cython_blas
from choreo.scipy_plus.cython.blas_consts cimport *

from libc.math cimport pow as cpow

cdef Py_ssize_t s_double = 27
cdef double split_factor = cpow(2., s_double) + 1 # = 134217729

cdef inline (double, double) Fast2Sum(double a, double b) noexcept nogil: 

    cdef double x = a + b
    cdef double y = (a-x) + b
    
    return (x,y)

cdef inline (double, double) TwoSum(double a, double b) noexcept nogil: 

    cdef double x = a+b
    cdef double z = x-a
    cdef double y = (a-(x-z))+(b-z)

    return (x,y)

def TwoSum_py(a, b):
    return TwoSum(a, b)

cdef inline (double, double) Split(double a) noexcept nogil: 

    # cdef double c = split_factor * a
    cdef double c = 134217729 * a

    cdef double x = c-(c-a)
    cdef double y = a-x

    return (x,y)

def Split_py(a):
    return Split(a)

cdef inline (double, double) TwoProduct(double a, double b) noexcept nogil: 

    cdef double x = a*b
    cdef double a1, a2, b1, b2

    a1, a2 = Split(a)
    b1, b2 = Split(b)

    cdef double y = a2*b2 - (((x-a1*b1)-a2*b1)-a1*b2)

    return (x,y)

def TwoProduct_py(a, b):
    return TwoProduct(a, b)

cdef inline void TwoSum_incr(double *y, double *d, double *e, int n) noexcept nogil: 
    
    cdef Py_ssize_t j
    cdef double a

    for j in range(n):

        a = y[j]
        e[j] = e[j] + d[j]
        y[j] = a + e[j]
        e[j] = e[j] + (a - y[j])

cdef inline void TwoSumScal_incr(double *y, double *d, double s, double *e, int n) noexcept nogil: 
    
    cdef Py_ssize_t j
    cdef double a

    for j in range(n):

        a = y[j]
        e[j] = e[j] + s*d[j]
        y[j] = a + e[j]
        e[j] = e[j] + (a - y[j])

cdef inline double SumXBLAS(double* p, Py_ssize_t n) noexcept nogil:

    cdef Py_ssize_t i
    cdef double s = 0
    cdef double t = 0

    cdef double t1, t2

    for i in range(n):
        t1, t2 = TwoSum(s, p[i])
        t2 += t
        s, t = Fast2Sum(t1, t2)

    return s

cdef inline void FastVecSum(double* p, double* q, Py_ssize_t n) noexcept nogil:

    cdef Py_ssize_t i

    q[0] = p[0]
    for i in range(1, n):
        q[i], q[i-1] = Fast2Sum(p[i], q[i-1])

cdef inline void VecSum(double* p, double* q, Py_ssize_t n) noexcept nogil:

    cdef Py_ssize_t i

    q[0] = p[0]
    for i in range(1, n):

        q[i], q[i-1] = TwoSum(p[i], q[i-1])

cpdef double SumK(
    double[::1] v       ,
    Py_ssize_t k = 0    ,
) noexcept:
 
    cdef Py_ssize_t i

    cdef Py_ssize_t n = v.shape[0]
    cdef double[::1] cp 
    cdef double *p
    cdef double *q
    cdef double *r

    cdef double res = 0
    
    if k > 0:

        if k > 1:
            cp = v.copy()
        else:
            cp = v

        with nogil:

            p = &cp[0]
            q = <double*> malloc(sizeof(double) * n)

            for i in range(k):

                VecSum(p, q, n)

                r = p
                p = q
                q = r

            for i in range(n):
                res += p[i]

            if (k % 2) == 0:

                free(q)

            else:

                free(p)

    else:

        for i in range(n):
            res += v[i]

    return res

cpdef double FastSumK(
    double[::1] v       ,
    Py_ssize_t k = 0    ,
):
 
    cdef Py_ssize_t i

    cdef Py_ssize_t n = v.shape[0]
    cdef double[::1] cp 
    cdef double *arr
    cdef double *p
    cdef double *q
    cdef double *r

    cdef double res = 0
    
    if k > 0:

        if k > 1:
            cp = v.copy()
        else:
            cp = v

        with nogil:

            p = &cp[0]
            arr = <double*> malloc(sizeof(double) * n)
            q = arr

            for i in range(k):

                FastVecSum(p, q, n)

                r = p
                p = q
                q = r

            for i in range(n):
                res += p[i]

            free(arr)

    else:

        for i in range(n):
            res += v[i]

    return res

cpdef void compute_r_vec(double[::1] v, double[::1] w, double[::1] r) noexcept nogil:
    
    cdef Py_ssize_t i,j
    cdef double h, p

    p, r[0] = TwoProduct(v[0], w[0])

    j = v.shape[0]
    for i in range(1, v.shape[0]):
        h, r[i] = TwoProduct(v[i], w[i])
        p, r[j] = TwoSum(p, h)

        j += 1

    r[j] = p

cpdef double naive_dot(double[::1] v, double[::1] w):

    assert v.shape[0] == w.shape[0]

    cdef double res = 0.
    cdef Py_ssize_t i

    for i in range(v.shape[0]):
        res += v[i]*w[i]

    return res

cpdef double DotK(double[::1] v, double[::1] w, Py_ssize_t k = 0):

    assert v.shape[0] == w.shape[0]

    cdef double res
    cdef double[::1] r
    cdef int n = v.shape[0]

    if k == 0:

        res = scipy.linalg.cython_blas.ddot(&n, &v[0], &int_one, &w[0], &int_one)

    else:

        r = np.empty(2*v.shape[0], dtype=np.float64)

        compute_r_vec(v, w, r)
        res = SumK(r, k-1)

    return res
