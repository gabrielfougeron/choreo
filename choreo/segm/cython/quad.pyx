'''
quad.pyx : Defines segment quadrature related things.

'''

__all__ = [
    'QuadFormula',
    'IntegrateOnSegment',
]

from choreo.segm.cython.eft_lib cimport TwoSum_incr

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

from choreo.scipy_plus.cython.ccallback cimport ccallback_t, ccallback_prepare, ccallback_release, CCALLBACK_DEFAULTS, ccallback_signature_t

from choreo.scipy_plus.cython.blas_consts cimport *

cdef int PY_FUN = -1
cdef int C_FUN_MEMORYVIEW = 0
cdef int C_FUN_MEMORYVIEW_DATA = 1
cdef int C_FUN_POINTER = 2
cdef int C_FUN_POINTER_DATA = 3
cdef int C_FUN_ONEVAL = 4
cdef int C_FUN_ONEVAL_DATA = 5
cdef int N_SIGNATURES = 6
cdef ccallback_signature_t signatures[7]

ctypedef void (*c_fun_type_memoryview)(double, double[::1]) noexcept nogil 
signatures[C_FUN_MEMORYVIEW].signature = b"void (double, __Pyx_memviewslice)"
signatures[C_FUN_MEMORYVIEW].value = C_FUN_MEMORYVIEW

ctypedef void (*c_fun_type_memoryview_data)(double, double[::1], void*) noexcept nogil 
signatures[C_FUN_MEMORYVIEW_DATA].signature = b"void (double, __Pyx_memviewslice, void *)"
signatures[C_FUN_MEMORYVIEW_DATA].value = C_FUN_MEMORYVIEW_DATA

ctypedef void (*c_fun_type_pointer)(double, double*) noexcept nogil 
signatures[C_FUN_POINTER].signature = b"void (double, double *)"
signatures[C_FUN_POINTER].value = C_FUN_POINTER

ctypedef void (*c_fun_type_pointer_data)(double, double*, void*) noexcept nogil 
signatures[C_FUN_POINTER_DATA].signature = b"void (double, double *, void *)"
signatures[C_FUN_POINTER_DATA].value = C_FUN_POINTER_DATA

ctypedef double (*c_fun_type_oneval)(double) noexcept nogil 
signatures[C_FUN_ONEVAL].signature = b"double (double)"
signatures[C_FUN_ONEVAL].value = C_FUN_ONEVAL

ctypedef double (*c_fun_type_oneval_data)(double, void*) noexcept nogil 
signatures[C_FUN_ONEVAL_DATA].signature = b"double (double, void *)"
signatures[C_FUN_ONEVAL_DATA].value = C_FUN_ONEVAL_DATA

signatures[N_SIGNATURES].signature = NULL

cdef inline void LowLevelFun_apply(
    ccallback_t callback    ,
    double x                ,
    double[::1] res         ,
) noexcept nogil:

    if (callback.py_function == NULL):

        if (callback.user_data == NULL):

            if callback.signature.value == C_FUN_MEMORYVIEW:
                (<c_fun_type_memoryview> callback.c_function)(x, res)

            elif callback.signature.value == C_FUN_POINTER:
                (<c_fun_type_pointer> callback.c_function)(x, &res[0])

            elif callback.signature.value == C_FUN_ONEVAL:
                res[0] = (<c_fun_type_oneval> callback.c_function)(x)

            else:
                with gil:
                    raise ValueError("Incompatible function signature.")

        else:

            if callback.signature.value == C_FUN_MEMORYVIEW_DATA:
                (<c_fun_type_memoryview_data> callback.c_function)(x, res, callback.user_data)

            elif callback.signature.value == C_FUN_POINTER_DATA:
                (<c_fun_type_pointer_data> callback.c_function)(x, &res[0], callback.user_data)

            elif callback.signature.value == C_FUN_ONEVAL_DATA:
                res[0] = (<c_fun_type_oneval_data> callback.c_function)(x, callback.user_data)

            else:
                with gil:
                    raise ValueError("Incompatible function signature.")

    else:

        with gil:

            PyFun_apply(callback, x, res)

cdef void PyFun_apply(
    ccallback_t callback    ,
    double x                ,
    double[::1] res         ,
):

    cdef int n = res.shape[0]
    cdef double[::1] res_1D = (<object> callback.py_function)(x)

    scipy.linalg.cython_blas.dcopy(&n,&res_1D[0],&int_one,&res[0],&int_one)

@cython.final
cdef class QuadFormula:

    """

    integral( f ) ~ sum w_i f( x_i )
    
    """
    
    cdef double[::1] _w             # Integration weights on [0,1]
    cdef double[::1] _x             # Integration nodes on [0,1]
    cdef double[::1] _wlag          # Barycentric Lagrange interpolation weights
    cdef Py_ssize_t _th_cvg_rate    # Self-reported convergence rate on smooth functions

    def __init__(
        self                ,
        w                   ,
        x                   ,
        wlag                ,
        th_cvg_rate = None  ,
    ):

        self._w = w.copy()
        self._x = x.copy()
        self._wlag = wlag.copy()

        assert self._w.shape[0] == self._x.shape[0]

        if th_cvg_rate is None:
            self._th_cvg_rate = -1
        else:
            self._th_cvg_rate = th_cvg_rate

    def __repr__(self):

        res = f'QuadFormula object with {self._w.shape[0]} nodes\n'
        res += f'Nodes: {self.x}\n'
        res += f'Weights: {self.w}\n'

        return res

    @property
    def nsteps(self):
        return self._w.shape[0]

    @property
    def x(self):
        return np.asarray(self._x)
    
    @property
    def w(self):
        return np.asarray(self._w)      

    @property
    def wlag(self):
        return np.asarray(self._wlag)    

    @property
    def th_cvg_rate(self):
        return self._th_cvg_rate

cpdef np.ndarray[double, ndim=1, mode="c"] IntegrateOnSegment(
    object fun              ,
    int ndim                ,
    (double, double) x_span ,
    QuadFormula quad        ,
    Py_ssize_t nint = 1     ,
    bint DoEFT = True       ,
):

    cdef ccallback_t callback
    ccallback_prepare(&callback, signatures, fun, CCALLBACK_DEFAULTS)

    cdef double[::1] f_res = np.empty((ndim),dtype=np.float64)
    cdef np.ndarray[double, ndim=1, mode="c"] f_int_np = np.zeros((ndim),dtype=np.float64)
    cdef double[::1] f_int = f_int_np

    with nogil:
        IntegrateOnSegment_ann(
            callback    ,
            ndim        ,
            x_span      ,
            nint        ,
            DoEFT       ,
            quad._w     ,
            quad._x     ,
            f_res       ,
            f_int       ,
        )

    ccallback_release(&callback)

    return f_int_np

@cython.cdivision(True)
cdef void IntegrateOnSegment_ann(
    ccallback_t callback    ,
    int ndim                ,
    (double, double) x_span ,
    Py_ssize_t nint         ,
    bint DoEFT              ,
    double[::1] w           ,
    double[::1] x           ,
    double[::1] f_res       ,
    double[::1] f_int       ,
) noexcept nogil:

    cdef Py_ssize_t istep
    cdef Py_ssize_t iint
    cdef Py_ssize_t idim
    cdef double xbeg, dx
    cdef double xi

    cdef double* f_eft_comp

    if DoEFT:

        f_eft_comp = <double *> malloc(sizeof(double) * ndim)
        for istep in range(ndim):
            f_eft_comp[istep] = 0.

    cdef double *cdx = <double*> malloc(sizeof(double)*w.shape[0])

    dx = (x_span[1] - x_span[0]) / nint

    for istep in range(w.shape[0]):
        cdx[istep] = x[istep] * dx

    for iint in range(nint):
        xbeg = x_span[0] + iint * dx

        for istep in range(w.shape[0]):

            xi = xbeg + cdx[istep]

            # f_res = f(xi)
            LowLevelFun_apply(callback, xi, f_res)

            # f_int = f_int + w * f_res
            if DoEFT:
                scipy.linalg.cython_blas.dscal(&ndim,&w[istep],&f_res[0],&int_one)
                TwoSum_incr(&f_int[0],&f_res[0],f_eft_comp,ndim)

            else:
                scipy.linalg.cython_blas.daxpy(&ndim,&w[istep],&f_res[0],&int_one,&f_int[0],&int_one)

    scipy.linalg.cython_blas.dscal(&ndim,&dx,&f_int[0],&int_one)

    free(cdx)

    if DoEFT:
        free(f_eft_comp)
