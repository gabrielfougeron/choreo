'''
eft_lib.ptx : Error-free transformations
Warning: this file needs to be compile without unsafe optimizations. No -Ofast. No -ffast-math.
'''

from libc.stdlib cimport malloc, free
cimport cython

cdef (double, double) Fast2Sum(double a, double b) noexcept nogil: 

    cdef double x = a + b
    cdef double y = (a-x) + b
    
    return (x,y)

cdef (double, double) TwoSum(double a, double b) noexcept nogil: 

    cdef double x = a+b
    cdef double z = x-a
    cdef double y = (a-(x-z))+(b-z)

    return (x,y)

cdef void TwoSum_incr(double *y, double *d, double *e, int n) noexcept nogil: 
    
    cdef Py_ssize_t j
    cdef double a

    for j in range(n):

        a = y[j]
        e[j] = e[j] + d[j]
        y[j] = a + e[j]
        e[j] = e[j] + (a - y[j])

cdef void FastVecSum(double* p, double* q, Py_ssize_t n) noexcept nogil:

    cdef Py_ssize_t i

    q[0] = p[0]
    for i in range(1, n):

        q[i], q[i-1] = Fast2Sum(p[i], q[i-1])

cdef void VecSum(double* p, double* q, Py_ssize_t n) noexcept nogil:

    cdef Py_ssize_t i

    q[0] = p[0]
    for i in range(1, n):

        q[i], q[i-1] = TwoSum(p[i], q[i-1])

cpdef double SumK(
    double[::1] v       ,
    Py_ssize_t k = 0          ,
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
    Py_ssize_t k = 0          ,
):
 
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

                FastVecSum(p, q, n)

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

cpdef double naive_sum_vect(double[:] v) noexcept nogil:

    cdef Py_ssize_t i
    cdef double res = 0.

    for i in range(v.shape[0]):
        res += v[i]

    return res 
