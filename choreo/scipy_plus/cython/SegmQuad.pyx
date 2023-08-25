'''
ODE.pyx : Defines ODE-related things I designed I feel ought to be in scipy ... but faster !


'''


import os
cimport scipy.linalg.cython_blas
cimport scipy.linalg.cython_lapack

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

cdef class QuadFormula:
    
    cdef double[::1] w
    cdef double[::1] x
    cdef Py_ssize_t nsteps

    def __init__(self, w, x):

        self.w = w
        self.x = x
        self.nsteps = w.shape[0]

        assert w.shape[0] == x.shape[0]

@cython.cdivision(True)
cpdef double[::1] IntegrateOnSegment(
    object fun,
    long ndim,
    (double, double) x_span,
    long nint,
    QuadFormula quad
):

    cdef Py_ssize_t istep
    cdef long iint,idim
    cdef double xbeg, dx
    cdef double xi
    cdef double[::1] cdx = np.empty((quad.nsteps),dtype=np.float64)
    cdef double[::1] res
    cdef double[::1] f_int = np.zeros((ndim),dtype=np.float64)

    dx = (x_span[1] - x_span[0]) / nint

    for istep in range(quad.nsteps):
        cdx[istep] = quad.x[istep] * dx

    for iint in range(nint):

        xbeg = x_span[0] + iint * dx

        for istep in range(quad.nsteps):

            xi = xbeg + cdx[istep]
            res = fun(xi)

            for idim in range(ndim):

                f_int[idim] = f_int[idim] + quad.w[istep] * res[idim]

    for idim in range(ndim):

        f_int[idim] = f_int[idim] * dx

    return f_int
