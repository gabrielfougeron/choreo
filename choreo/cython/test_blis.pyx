import numpy as np
cimport numpy as np
np.import_array()
cimport cython

from libc.math cimport fabs as cfabs
cimport scipy.linalg.cython_blas
cimport blis.cy

cpdef void blis_matmul_contiguous(
    double[:,::1] a,
    double[:,::1] b,
    double[:,::1] c
) nogil noexcept:

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
