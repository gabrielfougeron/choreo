'''
SegmQuad.pyx : Defines segment quadrature related things I designed I feel ought to be in scipy ... but faster !

'''

__all__ = [
    'QuadFormula',
    'IntegrateOnSegment',
]

from choreo.scipy_plus.cython.eft_lib cimport TwoSum_incr

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

cdef int PY_FUN_FLOAT = 0
cdef int PY_FUN_NDARRAY = 1 

cdef inline void PyFun_apply(
    object fun          ,
    const int res_type  ,
    const double x      ,
    double[::1] res     ,
    int ndim            ,  
):

    cdef Py_ssize_t i
    cdef np.ndarray[double, ndim=1, mode="c"] f_res_np

    if (res_type == PY_FUN_FLOAT):   
        res[0] = fun(x)
    else:
        f_res_np = fun(x)
        
        scipy.linalg.cython_blas.dcopy(&ndim,&f_res_np[0],&int_one,&res[0],&int_one)
        # for i in range(ndim):
            # res[i] = f_res_np[i]
            
@cython.final
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

    def __repr__(self):

        res = f'QuadFormula object with {self.nsteps} nodes\n'
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
    def th_cvg_rate(self):
        return self._th_cvg_rate

cpdef np.ndarray[double, ndim=1, mode="c"] IntegrateOnSegment(
    object fun              ,
    int ndim                ,
    (double, double) x_span ,
    QuadFormula quad        ,
    long nint = 1           ,
    bint DoEFT = True       ,
):

    cdef ccallback_t callback
    ccallback_prepare(&callback, signatures, fun, CCALLBACK_DEFAULTS)
    cdef LowLevelFun lowlevelfun = LowLevelFun_init(callback)
    
    cdef object py_fun_res = None
    cdef object py_fun
    cdef int py_fun_type
    if lowlevelfun.fun_type == PY_FUN:
        py_fun = <object> lowlevelfun.py_fun
        
        py_fun_res = py_fun(x_span[0])

        if isinstance(py_fun_res, float):
            py_fun_type = PY_FUN_FLOAT
        elif isinstance(py_fun_res, np.ndarray):
            py_fun_type = PY_FUN_NDARRAY
        else:
            raise ValueError(f"Could not recognize return type of python callable. Found {type(py_fun_res)}.")

    else:

        py_fun = None
        py_fun_type = -1

    cdef double[::1] f_res = np.empty((ndim),dtype=np.float64)
    cdef np.ndarray[double, ndim=1, mode="c"] f_int_np = np.zeros((ndim),dtype=np.float64)
    cdef double[::1] f_int = f_int_np

    with nogil:
        IntegrateOnSegment_ann(
            lowlevelfun ,
            py_fun      ,
            py_fun_type ,
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
    LowLevelFun lowlevelfun         ,
    object py_fun                   ,
    const int py_fun_type           ,
    const int ndim                  ,
    const (double, double) x_span   ,
    const long nint                 ,
    const bint DoEFT                ,
    const double[::1] w             ,
    const double[::1] x             ,
    double[::1] f_res               ,
    double[::1] f_int               ,
) noexcept nogil:

    cdef Py_ssize_t istep
    cdef long iint
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
            if py_fun_type > 0:
                with gil:
                    PyFun_apply(py_fun, py_fun_type, xi, f_res, ndim)
            else:
                LowLevelFun_apply(lowlevelfun, xi, f_res)

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
