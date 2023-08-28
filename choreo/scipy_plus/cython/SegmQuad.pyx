'''
ODE.pyx : Defines ODE-related things I designed I feel ought to be in scipy ... but faster !


'''

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

from .ccallback cimport ccallback_t, ccallback_prepare, ccallback_release, CCALLBACK_DEFAULTS, ccallback_signature_t

cdef int PY_FUN = -1
cdef int C_FUN_MEMORYVIEW = 0
cdef int C_FUN_POINTER = 1
cdef int C_FUN_ONEVAL = 2

cdef ccallback_signature_t signatures[4]

ctypedef void (*c_fun_type_memoryview)(double, double[::1]) noexcept nogil 
signatures[C_FUN_MEMORYVIEW].signature = b"void (double, __Pyx_memviewslice)"
signatures[C_FUN_MEMORYVIEW].value = C_FUN_MEMORYVIEW

ctypedef void (*c_fun_type_pointer)(double, double*) noexcept nogil 
signatures[C_FUN_POINTER].signature = b"void (double, double *)"
signatures[C_FUN_POINTER].value = C_FUN_POINTER

ctypedef double (*c_fun_type_oneval)(double) noexcept nogil 
signatures[C_FUN_ONEVAL].signature = b"double (double)"
signatures[C_FUN_ONEVAL].value = C_FUN_ONEVAL

signatures[3].signature = NULL

cdef struct s_LowLevelFun:
    int fun_type
    void *py_fun
    c_fun_type_memoryview c_fun_memoryview
    c_fun_type_pointer c_fun_pointer
    c_fun_type_oneval c_fun_oneval

ctypedef s_LowLevelFun LowLevelFun

cdef LowLevelFun LowLevelFun_init(
    ccallback_t callback
):

    cdef LowLevelFun fun

    fun.py_fun = NULL
    fun.c_fun_memoryview = NULL
    fun.c_fun_pointer = NULL
    fun.c_fun_oneval = NULL

    if (callback.py_function == NULL):
        fun.fun_type = callback.signature.value
    else:
        fun.fun_type = PY_FUN

    if fun.fun_type == PY_FUN:
        fun.py_fun = callback.py_function

    elif fun.fun_type == C_FUN_MEMORYVIEW:
        fun.c_fun_memoryview = <c_fun_type_memoryview> callback.c_function

    elif fun.fun_type == C_FUN_POINTER:
        fun.c_fun_pointer = <c_fun_type_pointer> callback.c_function

    elif fun.fun_type == C_FUN_ONEVAL:
        fun.c_fun_oneval = <c_fun_type_oneval> callback.c_function

    return fun

cdef inline void LowLevelFun_apply(
    const LowLevelFun fun ,
    const double x        ,
    double[::1] res ,
) noexcept nogil:

    if fun.fun_type == C_FUN_MEMORYVIEW:
        fun.c_fun_memoryview(x, res)

    elif fun.fun_type == C_FUN_POINTER:
        fun.c_fun_pointer(x, &res[0])

    elif fun.fun_type == C_FUN_ONEVAL:
        res[0] = fun.c_fun_oneval(x)


cdef inline void PyFun_apply(
    object fun      ,
    const double x  ,
    double[::1] res ,
    Py_ssize_t ndim ,  
) noexcept:

    cdef object f_res_raw

    cdef Py_ssize_t i

    f_res_raw = fun(x)

    if isinstance(f_res_raw, float):   # idim is expected to be 1
        res[0] = f_res_raw
    else:
        for i in range(ndim):
            res[i] = f_res_raw[i]

cdef class QuadFormula:

    """

    integral( f ) ~ sum w_i f( x_i )
    
    """
    
    cdef double[::1] _w
    cdef double[::1] _x
    cdef long _th_cvg_rate

    def __init__(
        self                ,
        w                   ,
        x                   ,
        th_cvg_rate = None  ,
    ):

        self._w = w
        self._x = x

        assert self._w.shape[0] == self._x.shape[0]

        if th_cvg_rate is None:
            self._th_cvg_rate = -1
        else:
            self._th_cvg_rate = th_cvg_rate

    @property
    def nsteps(self):
        return self._w.shape[0]

    @property
    def x(self):
        return np.array(self._x)
    
    @property
    def w(self):
        return np.array(self._w)    

    @property
    def th_cvg_rate(self):
        return self._th_cvg_rate
    

cpdef np.ndarray[double, ndim=1, mode="c"] IntegrateOnSegment(
    object fun              ,
    long ndim               ,
    (double, double) x_span ,
    QuadFormula quad        ,
    long nint               ,
):

    cdef ccallback_t callback
    ccallback_prepare(&callback, signatures, fun, CCALLBACK_DEFAULTS)
    cdef LowLevelFun lowlevelfun = LowLevelFun_init(callback)
    cdef object py_fun = None

    cdef double[::1] f_res = np.empty((ndim),dtype=np.float64)
    cdef np.ndarray[double, ndim=1, mode="c"] f_int_np = np.zeros((ndim),dtype=np.float64)
    cdef double[::1] f_int = f_int_np

    if lowlevelfun.fun_type == PY_FUN:

        py_fun = <object> lowlevelfun.py_fun

        IntegrateOnSegment_ann_python(
            py_fun      ,
            ndim        ,
            x_span      ,
            nint        ,
            quad        ,
            f_res       ,
            f_int       ,
        )

    else:
        
        with nogil:
            IntegrateOnSegment_ann_lowlevel(
                lowlevelfun ,
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
cdef void IntegrateOnSegment_ann_lowlevel(
    LowLevelFun fun                 ,
    const long ndim                 ,
    const (double, double) x_span   ,
    const long nint                 ,
    QuadFormula quad                ,
    double[::1] f_res               ,
    double[::1] f_int               ,
) noexcept nogil:

    cdef Py_ssize_t istep
    cdef long iint
    cdef Py_ssize_t idim
    cdef double xbeg, dx
    cdef double xi
    cdef double *cdx = <double*> malloc(sizeof(double)*quad._w.shape[0])

    dx = (x_span[1] - x_span[0]) / nint

    for istep in range(quad._w.shape[0]):
        cdx[istep] = quad._x[istep] * dx

    for iint in range(nint):
        xbeg = x_span[0] + iint * dx

        for istep in range(quad._w.shape[0]):

            xi = xbeg + cdx[istep]

            LowLevelFun_apply(fun, xi, f_res)

            for idim in range(ndim):

                f_int[idim] = f_int[idim] + quad._w[istep] * f_res[idim]

    for idim in range(ndim):
        f_int[idim] = f_int[idim] * dx

    free(cdx)



@cython.cdivision(True)
cdef void IntegrateOnSegment_ann_python(
    object fun                      ,
    const long ndim                 ,
    const (double, double) x_span   ,
    const long nint                 ,
    QuadFormula quad                ,
    double[::1] f_res               ,
    double[::1] f_int               ,
) noexcept:

    cdef Py_ssize_t istep
    cdef long iint
    cdef Py_ssize_t idim
    cdef double xbeg, dx
    cdef double xi
    cdef double *cdx = <double*> malloc(sizeof(double)*quad._w.shape[0])

    cdef object f_res_raw

    dx = (x_span[1] - x_span[0]) / nint

    for istep in range(quad._w.shape[0]):
        cdx[istep] = quad._x[istep] * dx

    for iint in range(nint):
        xbeg = x_span[0] + iint * dx

        for istep in range(quad._w.shape[0]):

            xi = xbeg + cdx[istep]

            PyFun_apply(fun, xi, f_res, ndim)

            for idim in range(ndim):

                f_int[idim] = f_int[idim] + quad._w[istep] * f_res[idim]

    for idim in range(ndim):
        f_int[idim] = f_int[idim] * dx

    free(cdx)
