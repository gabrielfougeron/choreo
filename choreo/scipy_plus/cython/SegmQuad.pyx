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

from .ccallback cimport (ccallback_t, ccallback_prepare, ccallback_release, CCALLBACK_DEFAULTS,
                         ccallback_signature_t)

sigs = [
    (b"void (double, __Pyx_memviewslice)", 0),
]

cdef ccallback_signature_t signatures[2]

for idx, sig in enumerate(sigs):
    signatures[idx].signature = sig[0]
    signatures[idx].value = sig[1]

signatures[idx + 1].signature = NULL



ctypedef void (*cy_fun_type)(const double, double[::1]) nogil noexcept
cdef inline void cy_fun(
    double x,
    double[::1] res,
) nogil noexcept:

    cdef Py_ssize_t i
    cdef Py_ssize_t size = 10
    cdef double val

    for i in range(size):
        
        val = i * x
        res[i] = csin(val)

cpdef py_fun(const double x):

    cdef double[::1] res = np.empty((10),dtype=np.float64)
    
    cy_fun(x, res)
    
    return res


cdef class QuadFormula:
    
    cdef double[::1] w
    cdef double[::1] x

    def __init__(self, w, x):

        self.w = w
        self.x = x

        assert w.shape[0] == x.shape[0]

    @property
    def nsteps(self):
        return self.w.shape[0]
    

cpdef np.ndarray[double, ndim=1, mode="c"] IntegrateOnSegment(
    object fun              ,
    long ndim               ,
    (double, double) x_span ,
    QuadFormula quad        ,
    long nint               ,
):

    cdef ccallback_t callback
    ccallback_prepare(&callback, signatures, fun, CCALLBACK_DEFAULTS)

    cdef cy_fun_type c_fun = <cy_fun_type> callback.c_function
    cdef bint UseLowLevel = (c_fun != NULL)

    cdef object py_fun
    if UseLowLevel:
        py_fun = None
    else:
        py_fun = <object> callback.py_function

    cdef double[::1] f_res = np.empty((ndim),dtype=np.float64)
    cdef np.ndarray[double, ndim=1, mode="c"] f_int_np = np.zeros((ndim),dtype=np.float64)
    cdef double[::1] f_int = f_int_np

    with nogil:
        IntegrateOnSegment_ann(
            c_fun       ,
            py_fun      ,
            UseLowLevel ,
            ndim        ,
            x_span      ,
            nint        ,
            quad        ,
            f_res       ,
            f_int       ,
        )

    ccallback_release(&callback)

    return f_int_np

@cython.cdivision(True)
cdef void IntegrateOnSegment_ann(
    cy_fun_type c_fun           ,
    object py_fun               ,
    bint UseLowLevel            ,
    long ndim                   ,
    (double, double) x_span     ,
    long nint                   ,
    QuadFormula quad            ,
    double[::1] f_res           ,
    double[::1] f_int           ,
) nogil noexcept:

    cdef Py_ssize_t istep
    cdef long iint
    cdef Py_ssize_t idim
    cdef double xbeg, dx
    cdef double xi
    cdef double *cdx = <double*> malloc(sizeof(double)*quad.w.shape[0])

    dx = (x_span[1] - x_span[0]) / nint

    for istep in range(quad.w.shape[0]):
        cdx[istep] = quad.x[istep] * dx

    for iint in range(nint):
        xbeg = x_span[0] + iint * dx

        for istep in range(quad.w.shape[0]):

            xi = xbeg + cdx[istep]

            if UseLowLevel:
                c_fun(xi, f_res)
            else:
                with gil:
                    f_res = py_fun(xi)

            for idim in range(ndim):

                f_int[idim] = f_int[idim] + quad.w[istep] * f_res[idim]

    for idim in range(ndim):
        f_int[idim] = f_int[idim] * dx

    free(cdx)
