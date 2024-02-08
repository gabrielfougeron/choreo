import numpy as np
cimport numpy as np
np.import_array()
cimport cython

from libc.math cimport fabs as cfabs
cimport scipy.linalg.cython_blas
cimport blis.cy
from libc.stdlib cimport abort, malloc, free

cpdef void blis_matmul_contiguous(
    double[:,::1] a,
    double[:,::1] b,
    double[:,::1] c
) noexcept nogil:

    cdef int m = a.shape[0]
    cdef int k = a.shape[1]
    cdef int n = b.shape[1]

    blis.cy.gemm(
        blis.cy.NO_TRANSPOSE, blis.cy.NO_TRANSPOSE,
        m, n, k,
        1.0, &a[0,0], k, 1,
        &b[0,0], n, 1,
        0.0, &c[0,0], n, 1
    )


# Computes the real part of a @ b
cpdef void blis_matmul_real(
    double complex[:,::1] a ,
    double complex[:,::1] b ,
    double[:,::1] c         ,
) noexcept nogil:

    cdef int m = a.shape[0]
    cdef int k = a.shape[1]
    cdef int n = b.shape[1]

    cdef double* a_real = <double*> &a[0,0]
    cdef double* b_real = <double*> &b[0,0]
    cdef double* res = &c[0,0]

    cdef int lda = 2 * k
    cdef int ldb = 2 * n

    blis.cy.gemm(
        blis.cy.NO_TRANSPOSE, blis.cy.NO_TRANSPOSE,
        m, n, k,
        1.0, a_real, lda, 2,
        b_real, ldb, 2 ,
        0.0, res, n, 1
    )

    # Pointer addition
    a_real += 1
    b_real += 1

    blis.cy.gemm(
        blis.cy.NO_TRANSPOSE, blis.cy.NO_TRANSPOSE,
        m, n, k,
        -1.0, a_real, lda, 2,
        b_real, ldb, 2 ,
        1.0, res, n, 1
    )

cdef int int_one = 1
cdef int int_two = 2
cdef double minus_one_double = -1.
cdef double one_double = 1.
cdef double zero_double = 0.
cdef double complex one_double_complex = 1.
cdef double complex zero_double_complex = 0.
cdef char *transn = 'n'
cdef char *transt = 't'

cpdef void blas_matmul_real(
    double complex[:,::1] a ,
    double complex[:,::1] b ,
    double[:,::1] c         ,
) noexcept nogil:

    cdef int n = a.shape[0]
    cdef int k = a.shape[1]
    cdef int m = b.shape[1]

    cdef int mn = m*n
    cdef double complex* res_complex = <double complex*> malloc(sizeof(double complex) * mn)

    cdef int int_two = 2

    scipy.linalg.cython_blas.zgemm(transn, transn, &m, &n, &k, &one_double_complex, &b[0,0], &m, &a[0,0], &k, &zero_double_complex, res_complex, &m)

    cdef double* res_double = <double*> res_complex

    scipy.linalg.cython_blas.dcopy(&mn,res_double,&int_two,&c[0,0],&int_one)

    free(res_complex)


cpdef void blas_matmul_real_copy(
    double complex[:,::1] a ,
    double complex[:,::1] b ,
    double[:,::1] c         ,
) noexcept nogil:

    cdef int n = a.shape[0]
    cdef int k = a.shape[1]
    cdef int m = b.shape[1]

    cdef int nk = n*k
    cdef int km = k*m
    cdef int mn = m*n
    cdef double* a_double = <double*> malloc(sizeof(double) * nk)
    cdef double* b_double = <double*> malloc(sizeof(double) * km)

    cdef double* a_start = <double*> &a[0,0]
    cdef double* b_start = <double*> &b[0,0]
    cdef double* c_start = <double*> &c[0,0]

    scipy.linalg.cython_blas.dcopy(&nk,a_start,&int_two,a_double,&int_one)
    scipy.linalg.cython_blas.dcopy(&km,b_start,&int_two,b_double,&int_one)
    scipy.linalg.cython_blas.dgemm(transn, transn, &m, &n, &k, &one_double, b_double, &m, a_double, &k, &zero_double, c_start, &m)

    #  Pointer arithmetic
    a_start +=1
    b_start +=1

    scipy.linalg.cython_blas.dcopy(&nk,a_start,&int_two,a_double,&int_one)
    scipy.linalg.cython_blas.dcopy(&km,b_start,&int_two,b_double,&int_one)
    scipy.linalg.cython_blas.dgemm(transn, transn, &m, &n, &k, &minus_one_double, b_double, &m, a_double, &k, &one_double, c_start, &m)

    free(a_double)
    free(b_double)