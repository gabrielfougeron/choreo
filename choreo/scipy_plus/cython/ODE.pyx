'''
ODE.pyx : Defines ODE-related things I designed I feel ought to be in scipy ... but faster !

'''

__all__ = [
    'ExplicitSymplecticRKTable' ,
    'ImplicitRKTable'           ,
    'ExplicitSymplecticIVP'     ,
    'ImplicitSymplecticIVP'     ,
]

from choreo.scipy_plus.cython.eft_lib cimport TwoSum_incr

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

cdef double minus_one = -1.
cdef double one = 1.
cdef double zero = 0.
cdef char *transn = 'n'
cdef int int_one = 1

cdef int PY_FUN = -1
cdef int C_FUN_MEMORYVIEW = 0
cdef int C_FUN_POINTER = 1
cdef int C_GRAD_FUN_MEMORYVIEW = 2
cdef int C_GRAD_FUN_POINTER = 3

cdef int PY_FUN_FLOAT = 0
cdef int PY_FUN_NDARRAY = 1 

cdef ccallback_signature_t signatures[5]

ctypedef void (*c_fun_type_memoryview)(double, double[::1], double[::1]) noexcept nogil 
signatures[C_FUN_MEMORYVIEW].signature = b"void (double, __Pyx_memviewslice, __Pyx_memviewslice)"
signatures[C_FUN_MEMORYVIEW].value = C_FUN_MEMORYVIEW

ctypedef void (*c_fun_type_pointer)(double, double*, double*) noexcept nogil 
signatures[C_FUN_POINTER].signature = b"void (double, double *, double *)"
signatures[C_FUN_POINTER].value = C_FUN_POINTER

ctypedef void (*c_grad_fun_type_memoryview)(double, double[::1], double[:,::1], double[:,::1]) noexcept nogil 
signatures[C_GRAD_FUN_MEMORYVIEW].signature = b"void (double, __Pyx_memviewslice, __Pyx_memviewslice, __Pyx_memviewslice)"
signatures[C_GRAD_FUN_MEMORYVIEW].value = C_GRAD_FUN_MEMORYVIEW

ctypedef void (*c_grad_fun_type_pointer)(double, double*, double*, double*) noexcept nogil 
signatures[C_GRAD_FUN_POINTER].signature = b"void (double, double *, double *, double *)"
signatures[C_GRAD_FUN_POINTER].value = C_GRAD_FUN_POINTER

signatures[4].signature = NULL

cdef struct s_LowLevelFun:
    int fun_type
    void *py_fun
    c_fun_type_memoryview c_fun_memoryview
    c_fun_type_pointer c_fun_pointer
    c_grad_fun_type_memoryview c_grad_fun_memoryview
    c_grad_fun_type_pointer c_grad_fun_pointer

ctypedef s_LowLevelFun LowLevelFun

cdef LowLevelFun LowLevelFun_init(
    ccallback_t callback
):

    cdef LowLevelFun fun

    fun.py_fun = NULL
    fun.c_fun_memoryview = NULL
    fun.c_fun_pointer = NULL
    fun.c_grad_fun_memoryview = NULL
    fun.c_grad_fun_pointer = NULL

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

    elif fun.fun_type == C_GRAD_FUN_MEMORYVIEW:
        fun.c_grad_fun_memoryview = <c_grad_fun_type_memoryview> callback.c_function

    elif fun.fun_type == C_GRAD_FUN_POINTER:
        fun.c_grad_fun_pointer = <c_grad_fun_type_pointer> callback.c_function

    return fun

cdef inline void LowLevelFun_apply(
    const LowLevelFun fun   ,
    const double t          ,
    double[::1] x           ,
    double[::1] res         ,
) noexcept nogil:

    if fun.fun_type == C_FUN_MEMORYVIEW:
        fun.c_fun_memoryview(t, x, res)

    elif fun.fun_type == C_FUN_POINTER:
        fun.c_fun_pointer(t, &x[0], &res[0])

cdef inline void PyFun_apply(
    object fun          ,
    const int res_type  ,
    const double t      ,
    double[::1] x       ,
    double[::1] res     ,
):

    cdef int n
    cdef np.ndarray[double, ndim=1, mode="c"] f_res_np

    if (res_type == PY_FUN_FLOAT):   
        res[0] = fun(t, x)
    else:
        f_res_np = fun(t, x)

        n = x.shape[0]
        scipy.linalg.cython_blas.dcopy(&n,&f_res_np[0],&int_one,&res[0],&int_one)

cdef inline void LowLevelFun_grad_apply(
    const LowLevelFun grad_fun  ,
    const double t              ,
    double[::1] x               ,
    double[:,::1] grad_x        ,
    double[:,::1] res           ,
) noexcept nogil:

    if grad_fun.fun_type == C_GRAD_FUN_MEMORYVIEW:
        grad_fun.c_grad_fun_memoryview(t, x, grad_x, res)

    elif grad_fun.fun_type == C_GRAD_FUN_POINTER:
        grad_fun.c_grad_fun_pointer(t, &x[0], &grad_x[0,0], &res[0,0])

cdef inline void PyFun_grad_apply(
    object grad_fun         ,
    const int res_type      ,
    const double t          ,
    double[::1] x           ,
    double[:,::1] grad_x    ,
    double[:,::1] res       ,
):

    cdef int n
    cdef np.ndarray[double, ndim=2, mode="c"] f_res_np

    if (res_type == PY_FUN_FLOAT):   
        res[0,:] = grad_fun(t, x, grad_x)
    else:
        f_res_np = grad_fun(t, x, grad_x)

        n = grad_x.shape[0]*grad_x.shape[1]
        scipy.linalg.cython_blas.dcopy(&n,&f_res_np[0,0],&int_one,&res[0,0],&int_one)

cdef inline void LowLevelFun_apply_vectorized(
    const LowLevelFun fun   ,
    double[::1] all_t       ,
    double[:,::1] all_x     ,
    double[:,::1] all_res   ,
) noexcept nogil:

    cdef Py_ssize_t i

    if fun.fun_type == C_FUN_MEMORYVIEW:

        for i in range(all_t.shape[0]):
            fun.c_fun_memoryview(all_t[i], all_x[i,:], all_res[i,:])

    elif fun.fun_type == C_FUN_POINTER:

        for i in range(all_t.shape[0]):
            fun.c_fun_pointer(all_t[i], &all_x[i,0], &all_res[i,0])

cdef inline void PyFun_apply_vectorized(
    object fun              ,
    const int res_type      ,
    double[::1] all_t       ,
    double[:,::1] all_x     ,
    double[:,::1] all_res   ,
):

    cdef Py_ssize_t i
    cdef int n 
    cdef np.ndarray[double, ndim=1, mode="c"] f_res_np

    if (res_type == PY_FUN_FLOAT): 
        for i in range(all_t.shape[0]):  
            all_res[i,0] = fun(all_t[i], all_x[i,:])

    else:
        
        n = all_x.shape[1]
        
        for i in range(all_t.shape[0]):  
            f_res_np = fun(all_t[i], all_x[i,:])

            scipy.linalg.cython_blas.dcopy(&n,&f_res_np[0],&int_one,&all_res[i,0],&int_one)

cdef inline void LowLevelFun_apply_grad_vectorized(
    const LowLevelFun grad_fun  ,
    double[::1] all_t           ,
    double[:,::1] all_x         ,
    double[:,:,::1] all_grad_x  ,
    double[:,:,::1] all_res     ,
) noexcept nogil:

    cdef Py_ssize_t i

    if grad_fun.fun_type == C_GRAD_FUN_MEMORYVIEW:

        for i in range(all_t.shape[0]):
            grad_fun.c_grad_fun_memoryview(all_t[i], all_x[i,:], all_grad_x[i,:,:], all_res[i,:,:])

    elif grad_fun.fun_type == C_GRAD_FUN_POINTER:

        for i in range(all_t.shape[0]):
            grad_fun.c_grad_fun_pointer(all_t[i], &all_x[i,0], &all_grad_x[i,0,0], &all_res[i,0,0])

cdef inline void PyFun_apply_grad_vectorized(
    object fun                  ,
    const int res_type          ,
    double[::1] all_t           ,
    double[:,::1] all_x         ,
    double[:,:,::1] all_grad_x  ,
    double[:,:,::1] all_res     ,
):

    cdef Py_ssize_t i
    cdef int n 
    cdef np.ndarray[double, ndim=2, mode="c"] f_res_np

    if (res_type == PY_FUN_FLOAT): 
        for i in range(all_t.shape[0]):  
            all_res[i,0,:] = fun(all_t[i], all_x[i,:], all_grad_x[i,:,:])

    else:
        
        n = all_grad_x.shape[1] * all_grad_x.shape[2]
        
        for i in range(all_t.shape[0]):  
            f_res_np = fun(all_t[i], all_x[i,:], all_grad_x[i,:,:])

            scipy.linalg.cython_blas.dcopy(&n,&f_res_np[0,0],&int_one,&all_res[i,0,0],&int_one)

cdef class ExplicitSymplecticRKTable:
    
    cdef double[::1] _c_table
    cdef double[::1] _d_table
    cdef long _th_cvg_rate

    def __init__(
        self                ,
        c_table     = None  ,
        d_table     = None  ,
        th_cvg_rate = None  ,
    ):

        self._c_table = c_table.copy()
        self._d_table = d_table.copy()

        assert c_table.shape[0] == d_table.shape[0]

        if th_cvg_rate is None:
            self._th_cvg_rate = -1
        else:
            self._th_cvg_rate = th_cvg_rate

    @property
    def nsteps(self):
        return self._c_table.shape[0]

    @property
    def c_table(self):
        return np.asarray(self._c_table)
    
    @property
    def d_table(self):
        return np.asarray(self._d_table)    

    @property
    def th_cvg_rate(self):
        return self._th_cvg_rate

    def symmetric_adjoint(self):

        cdef Py_ssize_t nsteps = self.nsteps
        cdef Py_ssize_t i

        cdef double[::1] c_table_reversed = np.empty((nsteps), dtype=np.float64)
        cdef double[::1] d_table_reversed = np.empty((nsteps), dtype=np.float64)

        for i in range(nsteps):
            c_table_reversed[i] = self._d_table[nsteps-i-1]
            d_table_reversed[i] = self._c_table[nsteps-i-1]

        return ExplicitSymplecticRKTable(
            c_table = c_table_reversed      ,
            d_table = d_table_reversed      ,
            th_cvg_rate = self._th_cvg_rate ,
        )

@cython.cdivision(True)
cpdef ExplicitSymplecticIVP(
    object fun                      ,
    object gun                      ,
    (double, double) t_span         ,
    double[::1] x0                  ,
    double[::1] v0                  ,
    ExplicitSymplecticRKTable rk    ,
    object grad_fun = None          ,
    object grad_gun = None          ,
    double[:,::1] grad_x0 = None    ,
    double[:,::1] grad_v0 = None    ,
    object mode = "VX"              ,
    long nint = 1                   ,
    long keep_freq = -1             ,
    bint DoEFT = True               ,
): 

    if (x0.shape[0] != v0.shape[0]):
        raise ValueError("x0 and v0 must have the same shape")

    cdef ccallback_t callback_fun
    ccallback_prepare(&callback_fun, signatures, fun, CCALLBACK_DEFAULTS)
    cdef LowLevelFun lowlevelfun = LowLevelFun_init(callback_fun)

    cdef ccallback_t callback_gun
    ccallback_prepare(&callback_gun, signatures, gun, CCALLBACK_DEFAULTS)
    cdef LowLevelFun lowlevelgun = LowLevelFun_init(callback_gun)

    if (lowlevelfun.fun_type == PY_FUN) != (lowlevelgun.fun_type == PY_FUN):
        raise ValueError("fun and gun must both be python functions or LowLevelCallables")

    cdef object py_fun_res = None
    cdef object py_gun_res = None
    cdef object py_fun, py_gun
    cdef int py_fun_type

    if lowlevelfun.fun_type == PY_FUN:
        py_fun = <object> lowlevelfun.py_fun
        py_gun = <object> lowlevelgun.py_fun
        
        py_fun_res = py_fun(t_span[0], v0)
        py_gun_res = py_gun(t_span[0], x0)

        if isinstance(py_fun_res, float) and isinstance(py_gun_res, float):
            py_fun_type = PY_FUN_FLOAT
        elif isinstance(py_fun_res, np.ndarray) and isinstance(py_gun_res, np.ndarray):
            py_fun_type = PY_FUN_NDARRAY
        else:
            raise ValueError(f"Could not recognize return type of python callable. Found {type(py_fun_res)} and {type(py_gun_res)}.")

    else:
        py_fun = None
        py_gun = None
        py_fun_type = -1

    cdef bint DoTanIntegration = not(grad_fun is None) or not(grad_gun is None) or not(grad_x0 is None) or not(grad_v0 is None)

    cdef ccallback_t callback_grad_fun
    cdef LowLevelFun lowlevelgrad_fun

    cdef ccallback_t callback_grad_gun
    cdef LowLevelFun lowlevelgrad_gun

    cdef object py_grad_fun_res = None
    cdef object py_grad_gun_res = None
    cdef object py_grad_fun, py_grad_gun

    if (keep_freq < 0):
        keep_freq = nint

    cdef long ndof = x0.shape[0]
    cdef long nint_keep = nint // keep_freq

    cdef double[::1] x = x0.copy()
    cdef double[::1] v = v0.copy()

    cdef double[::1] res = np.empty((ndof), dtype=np.float64)

    cdef np.ndarray[double, ndim=2, mode="c"] x_keep_np = np.empty((nint_keep, ndof), dtype=np.float64)
    cdef np.ndarray[double, ndim=2, mode="c"] v_keep_np = np.empty((nint_keep, ndof), dtype=np.float64)
    cdef double[:,::1] x_keep = x_keep_np
    cdef double[:,::1] v_keep = v_keep_np

    cdef double[:,::1] grad_x
    cdef double[:,::1] grad_v
    cdef double[:,::1] grad_res

    cdef np.ndarray[double, ndim=3, mode="c"] grad_x_keep_np
    cdef np.ndarray[double, ndim=3, mode="c"] grad_v_keep_np
    cdef double[:,:,::1] grad_x_keep
    cdef double[:,:,::1] grad_v_keep
    cdef int grad_ndof
    cdef Py_ssize_t i,j

    if DoTanIntegration:

        if (grad_x0 is None) and (grad_v0 is None):

            grad_ndof = 2*ndof

            grad_x = np.zeros((ndof, grad_ndof), dtype=np.float64)
            grad_v = np.zeros((ndof, grad_ndof), dtype=np.float64)

            for i in range(ndof):

                grad_x[i,i] = 1.
                
                j = ndof+i
                grad_v[i,j] = 1.

        elif not(grad_x0 is None) and not(grad_v0 is None):

            grad_x = grad_x0.copy()
            grad_v = grad_v0.copy()

            assert grad_x.shape[0] == ndof
            assert grad_v.shape[0] == ndof
            
            grad_ndof = grad_x.shape[1]
            assert grad_v.shape[1] == grad_ndof

        else:
            raise ValueError('Wrong values for grad_x0 and/or grad_v0')

        grad_x_keep_np = np.empty((nint_keep, ndof, grad_ndof), dtype=np.float64)
        grad_v_keep_np = np.empty((nint_keep, ndof, grad_ndof), dtype=np.float64)

        grad_x_keep = grad_x_keep_np
        grad_v_keep = grad_v_keep_np

        grad_res = np.empty((ndof, grad_ndof), dtype=np.float64)

        if grad_fun is None:
            raise ValueError(f"grad_fun was not provided")
        if grad_gun is None:
            raise ValueError(f"grad_gun was not provided")

        ccallback_prepare(&callback_grad_fun, signatures, grad_fun, CCALLBACK_DEFAULTS)
        lowlevelgrad_fun = LowLevelFun_init(callback_grad_fun)

        ccallback_prepare(&callback_grad_gun, signatures, grad_gun, CCALLBACK_DEFAULTS)
        lowlevelgrad_gun = LowLevelFun_init(callback_grad_gun)

        if (lowlevelgrad_fun.fun_type == PY_FUN) != (lowlevelgrad_gun.fun_type == PY_FUN):
            raise ValueError("fun and gun must both be python functions or LowLevelCallables")

        if lowlevelgrad_fun.fun_type == PY_FUN:
            py_grad_fun = <object> lowlevelgrad_fun.py_fun
            py_grad_gun = <object> lowlevelgrad_gun.py_fun
            
            py_grad_fun_res = py_grad_fun(t_span[0], v0, grad_v)
            py_grad_gun_res = py_grad_gun(t_span[0], x0, grad_x)

            if isinstance(py_grad_fun_res, float) and isinstance(py_grad_gun_res, float):
                py_grad_fun_type = PY_FUN_FLOAT
            elif isinstance(py_fun_res, np.ndarray) and isinstance(py_grad_gun_res, np.ndarray):
                py_grad_fun_type = PY_FUN_NDARRAY
            else:
                raise ValueError(f"Could not recognize return type of python callable. Found {type(py_grad_fun_res)} and {type(py_grad_gun_res)}.")

        else:
            py_grad_fun = None
            py_grad_gun = None

    else:

        grad_x = np.zeros((0, 0), dtype=np.float64)
        grad_v = np.zeros((0, 0), dtype=np.float64)

        grad_x_keep_np = np.empty((0, 0, 0), dtype=np.float64)
        grad_v_keep_np = np.empty((0, 0, 0), dtype=np.float64)

        grad_x_keep = grad_x_keep_np
        grad_v_keep = grad_v_keep_np

        grad_res = np.empty((0, 0), dtype=np.float64)

        py_grad_fun = None
        py_grad_gun = None        

    if mode == 'VX':

        with nogil:
        
            ExplicitSymplecticIVP_ann(
                lowlevelfun         ,
                lowlevelgun         ,
                lowlevelgrad_fun    ,
                lowlevelgrad_gun    ,
                py_fun              ,
                py_gun              ,
                py_grad_fun         ,
                py_grad_gun         ,
                py_fun_type         ,
                t_span              ,
                x                   ,
                v                   ,
                grad_x              ,
                grad_v              ,
                res                 ,
                grad_res            ,
                rk                  ,
                nint                ,
                keep_freq           ,
                DoEFT               ,
                DoTanIntegration    ,
                x_keep              ,
                v_keep              ,
                grad_x_keep         ,
                grad_v_keep         ,
            )

    elif mode == 'XV':

        with nogil:
            ExplicitSymplecticIVP_ann(
                lowlevelgun         ,
                lowlevelfun         ,
                lowlevelgrad_gun    ,
                lowlevelgrad_fun    ,
                py_gun              ,
                py_fun              ,
                py_grad_gun         ,
                py_grad_fun         ,
                py_fun_type         ,
                t_span              ,
                v                   ,
                x                   ,
                grad_v              ,
                grad_x              ,
                res                 ,
                grad_res            ,
                rk                  ,
                nint                ,
                keep_freq           ,
                DoEFT               ,
                DoTanIntegration    ,
                v_keep              ,
                x_keep              ,
                grad_v_keep         ,
                grad_x_keep         ,
            )

    else:
        raise ValueError(f"Unknown mode {mode}. Possible options are 'VX' and 'XV'.")

    ccallback_release(&callback_fun)
    ccallback_release(&callback_gun)

    if DoTanIntegration:

        ccallback_release(&callback_grad_fun)
        ccallback_release(&callback_grad_gun)

        return x_keep_np, v_keep_np, grad_x_keep_np, grad_v_keep_np
    
    else:
        
        return x_keep_np, v_keep_np

@cython.cdivision(True)
cdef void ExplicitSymplecticIVP_ann(
    const LowLevelFun lowlevelfun       ,
    const LowLevelFun lowlevelgun       ,
    const LowLevelFun lowlevelgrad_fun  ,
    const LowLevelFun lowlevelgrad_gun  ,
          object py_fun                 ,
          object py_gun                 ,
          object py_grad_fun            ,
          object py_grad_gun            ,
    const int py_fun_type               ,
    const (double, double) t_span       ,
          double[::1]     x             ,
          double[::1]     v             ,
          double[:,::1]   grad_x        ,
          double[:,::1]   grad_v        ,
          double[::1]     res           ,
          double[:,::1]   grad_res      ,
          ExplicitSymplecticRKTable rk  ,
    const long nint                     ,
    const long keep_freq                ,
    const bint DoEFT                    ,
    const bint DoTanIntegration         ,
          double[:,::1]   x_keep        ,
          double[:,::1]   v_keep        ,
          double[:,:,::1] grad_x_keep   ,
          double[:,:,::1] grad_v_keep   ,
) noexcept nogil:

    cdef double tx = t_span[0]
    cdef double tv = t_span[0]
    cdef double dt = (t_span[1] - t_span[0]) / nint

    cdef int ndof = x.shape[0]
    cdef long nint_keep = nint // keep_freq
    cdef long nsteps = rk._c_table.shape[0]

    cdef int grad_nvar
    if DoTanIntegration:
        grad_nvar = ndof * grad_x.shape[1]

    cdef double *cdt = <double *> malloc(sizeof(double) * nsteps)
    cdef double *ddt = <double *> malloc(sizeof(double) * nsteps)

    cdef double *x_eft_comp
    cdef double *v_eft_comp
    cdef double *grad_x_eft_comp
    cdef double *grad_v_eft_comp
    cdef double tx_comp = 0.
    cdef double tv_comp = 0.

    cdef Py_ssize_t istep, iint_keep

    if DoEFT:

        x_eft_comp = <double *> malloc(sizeof(double) * ndof)
        for istep in range(ndof):
            x_eft_comp[istep] = 0.

        v_eft_comp = <double *> malloc(sizeof(double) * ndof)
        for istep in range(ndof):
            v_eft_comp[istep] = 0.

        if DoTanIntegration:

            grad_x_eft_comp = <double *> malloc(sizeof(double) * grad_nvar)
            for istep in range(grad_nvar):
                grad_x_eft_comp[istep] = 0.

            grad_v_eft_comp = <double *> malloc(sizeof(double) * grad_nvar)
            for istep in range(grad_nvar):
                grad_v_eft_comp[istep] = 0.

    for istep in range(nsteps):
        cdt[istep] = rk._c_table[istep] * dt
        ddt[istep] = rk._d_table[istep] * dt

    for iint_keep in range(nint_keep):

        for ifreq in range(keep_freq):

            for istep in range(nsteps):
                
                # res = f(t,v)
                if py_fun_type > 0:
                    with gil:
                        PyFun_apply(py_fun, py_fun_type, tv, v, res)
                else:
                    LowLevelFun_apply(lowlevelfun, tv, v, res)
                
                if DoTanIntegration:
                    # grad_res = grad_f(t,v,grad_v)
                    if py_fun_type > 0:
                        with gil:
                            PyFun_grad_apply(py_grad_fun, py_fun_type, tv, v, grad_v, grad_res)
                    else:
                        LowLevelFun_grad_apply(lowlevelgrad_fun, tv, v, grad_v, grad_res)

                # x = x + cdt * res
                if DoEFT:
                    scipy.linalg.cython_blas.dscal(&ndof,&cdt[istep],&res[0],&int_one)
                    TwoSum_incr(&x[0],&res[0],x_eft_comp,ndof)
                    TwoSum_incr(&tx,&cdt[istep],&tx_comp,1)

                    if DoTanIntegration:
                        scipy.linalg.cython_blas.dscal(&grad_nvar,&cdt[istep],&grad_res[0,0],&int_one)
                        TwoSum_incr(&grad_x[0,0],&grad_res[0,0],grad_x_eft_comp,grad_nvar)

                else:
                    scipy.linalg.cython_blas.daxpy(&ndof,&cdt[istep],&res[0],&int_one,&x[0],&int_one)
                    tx += cdt[istep]

                    if DoTanIntegration:
                        scipy.linalg.cython_blas.daxpy(&grad_nvar,&cdt[istep],&grad_res[0,0],&int_one,&grad_x[0,0],&int_one)

                # res = g(t,x)
                if py_fun_type > 0:
                    with gil:
                        PyFun_apply(py_gun, py_fun_type, tx, x, res)
                else:
                    LowLevelFun_apply(lowlevelgun, tx, x, res)

                if DoTanIntegration:
                    # grad_res = grad_g(t,x,grad_x)
                    if py_fun_type > 0:
                        with gil:
                            PyFun_grad_apply(py_grad_gun, py_fun_type, tx, x, grad_x, grad_res)
                    else:
                        LowLevelFun_grad_apply(lowlevelgrad_gun, tx, x, grad_x, grad_res)

                # v = v + ddt * res
                if DoEFT:
                    scipy.linalg.cython_blas.dscal(&ndof,&ddt[istep],&res[0],&int_one)
                    TwoSum_incr(&v[0],&res[0],v_eft_comp,ndof)
                    TwoSum_incr(&tv,&ddt[istep],&tv_comp,1)

                    if DoTanIntegration:
                        scipy.linalg.cython_blas.dscal(&grad_nvar,&ddt[istep],&grad_res[0,0],&int_one)
                        TwoSum_incr(&grad_v[0,0],&grad_res[0,0],grad_v_eft_comp,grad_nvar)

                else:
                    scipy.linalg.cython_blas.daxpy(&ndof,&ddt[istep],&res[0],&int_one,&v[0],&int_one)
                    tv += ddt[istep]

                    if DoTanIntegration:
                        scipy.linalg.cython_blas.daxpy(&grad_nvar,&ddt[istep],&grad_res[0,0],&int_one,&grad_v[0,0],&int_one)

        scipy.linalg.cython_blas.dcopy(&ndof,&x[0],&int_one,&x_keep[iint_keep,0],&int_one)
        scipy.linalg.cython_blas.dcopy(&ndof,&v[0],&int_one,&v_keep[iint_keep,0],&int_one)

        if DoTanIntegration:

            scipy.linalg.cython_blas.dcopy(&grad_nvar,&grad_x[0,0],&int_one,&grad_x_keep[iint_keep,0,0],&int_one)
            scipy.linalg.cython_blas.dcopy(&grad_nvar,&grad_v[0,0],&int_one,&grad_v_keep[iint_keep,0,0],&int_one)

    free(cdt)
    free(ddt)

    if DoEFT:
        free(x_eft_comp)
        free(v_eft_comp)

        if DoTanIntegration:

            free(grad_x_eft_comp)
            free(grad_v_eft_comp)

cdef class ImplicitRKTable:
    
    cdef double[:,::1] _a_table
    cdef double[::1] _b_table
    cdef double[::1] _c_table
    cdef double[:,::1] _beta_table
    cdef double[:,::1] _gamma_table
    cdef long _th_cvg_rate

    def __init__(
        self                ,
        a_table     = None  ,
        b_table     = None  ,
        c_table     = None  ,
        beta_table  = None  ,
        gamma_table = None  ,
        th_cvg_rate = None  ,
    ):

        self._a_table = a_table.copy()
        self._b_table = b_table.copy()
        self._c_table = c_table.copy()
        self._beta_table = beta_table.copy()
        self._gamma_table = gamma_table.copy()

        assert self._a_table.shape[0] == self._a_table.shape[1]
        assert self._a_table.shape[0] == self._b_table.shape[0]
        assert self._a_table.shape[0] == self._c_table.shape[0]
        assert self._a_table.shape[0] == self._beta_table.shape[0]
        assert self._a_table.shape[0] == self._beta_table.shape[1]
        assert self._a_table.shape[0] == self._gamma_table.shape[0]
        assert self._a_table.shape[0] == self._gamma_table.shape[1]

        if th_cvg_rate is None:
            self._th_cvg_rate = -1
        else:
            self._th_cvg_rate = th_cvg_rate

    @property
    def nsteps(self):
        return self._a_table.shape[0]

    @property
    def a_table(self):
        return np.asarray(self._a_table)

    @property
    def b_table(self):
        return np.asarray(self._b_table)

    @property
    def c_table(self):
        return np.asarray(self._c_table)
    
    @property
    def beta_table(self):
        return np.asarray(self._beta_table)    

    @property
    def gamma_table(self):
        return np.asarray(self._gamma_table)    

    @property
    def th_cvg_rate(self):
        return self._th_cvg_rate

    @property
    def stability_cst(self):
        return np.linalg.norm(self.a_table, np.inf)

    cpdef ImplicitRKTable symmetric_adjoint(self):

        cdef Py_ssize_t n = self._a_table.shape[0]
        cdef Py_ssize_t i, j

        cdef double[:,::1] a_table_sym = np.empty((n,n), dtype=np.float64)
        cdef double[::1] b_table_sym = np.empty((n), dtype=np.float64)
        cdef double[::1] c_table_sym = np.empty((n), dtype=np.float64)
        cdef double[:,::1] beta_table_sym = np.empty((n,n), dtype=np.float64)
        cdef double[:,::1] gamma_table_sym = np.empty((n,n), dtype=np.float64)

        for i in range(n):

            b_table_sym[i] = self._b_table[n-1-i]
            c_table_sym[i] = 1. - self._c_table[n-1-i]

        for i in range(n):
            for j in range(n):
                
                a_table_sym[i,j] = self._b_table[n-1-j] - self._a_table[n-1-i,n-1-j]
                beta_table_sym[i,j]  = self._gamma_table[n-1-i,n-1-j]
                gamma_table_sym[i,j] = self._beta_table[n-1-i,n-1-j]

        return ImplicitRKTable(
            a_table     = a_table_sym       ,
            b_table     = b_table_sym       ,
            c_table     = c_table_sym       ,
            beta_table  = beta_table_sym    ,
            gamma_table = gamma_table_sym   ,
            th_cvg_rate = self._th_cvg_rate ,
        )

    cpdef double _symmetry_default(
        self                    ,
        ImplicitRKTable other   ,
    ) noexcept:

        cdef Py_ssize_t nsteps = self._a_table.shape[0]
        cdef Py_ssize_t i,j
        cdef double maxi = -1
        cdef double val

        for i in range(nsteps):

            val = self._b_table[i] - other._b_table[nsteps-1-i] 
            maxi = max(maxi, cfabs(val))

            val = self._c_table[i] + other._c_table[nsteps-1-i] - 1
            maxi = max(maxi, cfabs(val))

            for j in range(self._a_table.shape[0]):
                val = self._a_table[i,j] - self._b_table[j] + other._a_table[nsteps-1-i,nsteps-1-j]
                maxi = max(maxi, cfabs(val))
                
                val = self._beta_table[i,j] - other._gamma_table[nsteps-1-i,nsteps-1-j]
                maxi = max(maxi, cfabs(val))

                val = self._gamma_table[i,j] - other._beta_table[nsteps-1-i,nsteps-1-j]
                maxi = max(maxi, cfabs(val))
                
        return maxi

    def symmetry_default(
        self        ,
        other = None,
    ):
        if other is None:
            return self._symmetry_default(self)
        else:
            return self._symmetry_default(other)
     
    cpdef bint _is_symmetric_pair(self, ImplicitRKTable other, double tol):
        return (self._symmetry_default(other) < tol)

    def is_symmetric_pair(self, ImplicitRKTable other, double tol = 1e-12):
        return self._is_symmetric_pair(other, tol)

    def is_symmetric(self, double tol = 1e-12):
        return self._is_symmetric_pair(self, tol)

    @cython.cdivision(True)
    cpdef ImplicitRKTable symplectic_adjoint(self):

        cdef Py_ssize_t nsteps = self._a_table.shape[0]
        cdef Py_ssize_t i, j

        cdef double[:,::1] a_table_sym = np.empty((nsteps,nsteps), dtype=np.float64)

        for i in range(nsteps):
            for j in range(nsteps):
                
                a_table_sym[i,j] = self._b_table[j] * (1. - self._a_table[j,i] / self._b_table[i])

        return ImplicitRKTable(
            a_table     = a_table_sym       ,
            b_table     = self._b_table     ,
            c_table     = self._c_table     ,
            beta_table  = self._beta_table  ,
            gamma_table = self._gamma_table ,
            th_cvg_rate = self._th_cvg_rate ,
        )

    cpdef double _symplectic_default(
        self                    ,
        ImplicitRKTable other   ,
    ) noexcept:

        cdef Py_ssize_t nsteps = self._a_table.shape[0]
        cdef Py_ssize_t i,j
        cdef double maxi = -1
        cdef double val

        for i in range(nsteps):

            val = self._b_table[i] - other._b_table[i] 
            maxi = max(maxi, cfabs(val))

            val = self._c_table[i] - other._c_table[i] 
            maxi = max(maxi, cfabs(val))

            for j in range(nsteps):
                val = self._b_table[i] * other._a_table[i,j] + other._b_table[j] * self._a_table[j,i] - self._b_table[i] * other._b_table[j] 
                maxi = max(maxi, cfabs(val))

        return maxi

    def symplectic_default(
        self        ,
        other = None,
    ):
        if other is None:
            return self._symplectic_default(self)
        else:
            return self._symplectic_default(other)
     
    cpdef bint _is_symplectic_pair(self, ImplicitRKTable other, double tol):
        return (self._symplectic_default(other) < tol)

    def is_symplectic_pair(self, ImplicitRKTable other, double tol = 1e-12):
        return self._is_symplectic_pair(other, tol)

    def is_symplectic(self, double tol = 1e-12):
        return self._is_symplectic_pair(self, tol)

@cython.cdivision(True)
cpdef ImplicitSymplecticIVP(
    object fun                              ,
    object gun                              ,
    (double, double) t_span                 ,
    double[::1] x0                          ,
    double[::1] v0                          ,
    ImplicitRKTable rk_x                    ,
    ImplicitRKTable rk_v                    ,
    object grad_fun = None                  ,
    object grad_gun = None                  ,
    double[:,::1] grad_x0 = None            ,
    double[:,::1] grad_v0 = None            ,
    long nint = 1                           ,
    long keep_freq = -1                     ,
    # bint DoEFT = True                       ,
    bint DoEFT = False                       ,
    double eps = np.finfo(np.float64).eps   ,
    long maxiter = 50                       ,
):

    cdef long nsteps = rk_x._a_table.shape[0]

    if (rk_v._a_table.shape[0] != nsteps):
        raise ValueError("rk_x and rk_v must have the same shape")

    if (x0.shape[0] != v0.shape[0]):
        raise ValueError("x0 and v0 must have the same shape")

    cdef ccallback_t callback_fun
    ccallback_prepare(&callback_fun, signatures, fun, CCALLBACK_DEFAULTS)
    cdef LowLevelFun lowlevelfun = LowLevelFun_init(callback_fun)

    cdef ccallback_t callback_gun
    ccallback_prepare(&callback_gun, signatures, gun, CCALLBACK_DEFAULTS)
    cdef LowLevelFun lowlevelgun = LowLevelFun_init(callback_gun)

    if (lowlevelfun.fun_type == PY_FUN) != (lowlevelgun.fun_type == PY_FUN):
        raise ValueError("fun and gun must both be python functions or LowLevelCallables")

    cdef object py_fun_res = None
    cdef object py_gun_res = None
    cdef object py_fun, py_gun
    cdef int py_fun_type

    if lowlevelfun.fun_type == PY_FUN:
        py_fun = <object> lowlevelfun.py_fun
        py_gun = <object> lowlevelgun.py_fun
        
        py_fun_res = py_fun(t_span[0], v0)
        py_gun_res = py_gun(t_span[0], x0)

        if isinstance(py_fun_res, float) and isinstance(py_gun_res, float):
            py_fun_type = PY_FUN_FLOAT
        elif isinstance(py_fun_res, np.ndarray) and isinstance(py_gun_res, np.ndarray):
            py_fun_type = PY_FUN_NDARRAY
            
            assert py_fun_res.shape[0] == x0.shape[0]
            assert py_gun_res.shape[0] == x0.shape[0]

        else:
            raise ValueError(f"Could not recognize return type of python callable. Found {type(py_fun_res)} and {type(py_gun_res)}.")

    else:
        py_fun = None
        py_gun = None
        py_fun_type = -1

    cdef bint DoTanIntegration = not(grad_fun is None) or not(grad_gun is None) or not(grad_x0 is None) or not(grad_v0 is None)

    cdef ccallback_t callback_grad_fun
    cdef LowLevelFun lowlevelgrad_fun

    cdef ccallback_t callback_grad_gun
    cdef LowLevelFun lowlevelgrad_gun

    cdef object py_grad_fun_res = None
    cdef object py_grad_gun_res = None
    cdef object py_grad_fun, py_grad_gun

    if (keep_freq < 0):
        keep_freq = nint

    cdef long ndof = x0.shape[0]
    cdef long nint_keep = nint // keep_freq

    cdef double[::1] x = x0.copy()
    cdef double[::1] v = v0.copy()
    cdef double[:,::1] K_fun = np.zeros((nsteps, ndof), dtype=np.float64)
    cdef double[:,::1] K_gun = np.zeros((nsteps, ndof), dtype=np.float64)
    cdef double[:,::1] dX = np.empty((nsteps, ndof), dtype=np.float64)
    cdef double[:,::1] dV = np.empty((nsteps, ndof), dtype=np.float64) 
    cdef double[:,::1] dX_prev = np.empty((nsteps, ndof), dtype=np.float64)
    cdef double[:,::1] dV_prev = np.empty((nsteps, ndof), dtype=np.float64) 

    cdef double[::1] all_t_x = np.empty((nsteps), dtype=np.float64) 
    cdef double[::1] all_t_v = np.empty((nsteps), dtype=np.float64) 

    cdef np.ndarray[double, ndim=2, mode="c"] x_keep_np = np.empty((nint_keep, ndof), dtype=np.float64)
    cdef np.ndarray[double, ndim=2, mode="c"] v_keep_np = np.empty((nint_keep, ndof), dtype=np.float64)
    cdef double[:,::1] x_keep = x_keep_np
    cdef double[:,::1] v_keep = v_keep_np

    cdef double[:,::1] grad_x
    cdef double[:,::1] grad_v
    cdef double[:,:,::1] grad_K_fun
    cdef double[:,:,::1] grad_K_gun
    cdef double[:,:,::1] grad_dX
    cdef double[:,:,::1] grad_dV 
    cdef double[:,:,::1] grad_dX_prev
    cdef double[:,:,::1] grad_dV_prev 

    cdef np.ndarray[double, ndim=3, mode="c"] grad_x_keep_np
    cdef np.ndarray[double, ndim=3, mode="c"] grad_v_keep_np
    cdef double[:,:,::1] grad_x_keep
    cdef double[:,:,::1] grad_v_keep
    cdef int grad_ndof
    cdef Py_ssize_t i,j

    if DoTanIntegration:

        if (grad_x0 is None) and (grad_v0 is None):

            grad_ndof = 2*ndof

            grad_x = np.zeros((ndof, grad_ndof), dtype=np.float64)
            grad_v = np.zeros((ndof, grad_ndof), dtype=np.float64)

            for i in range(ndof):

                grad_x[i,i] = 1.
                
                j = ndof+i
                grad_v[i,j] = 1.

        elif not(grad_x0 is None) and not(grad_v0 is None):

            grad_x = grad_x0.copy()
            grad_v = grad_v0.copy()

            assert grad_x.shape[0] == ndof
            assert grad_v.shape[0] == ndof
            
            grad_ndof = grad_x.shape[1]
            assert grad_v.shape[1] == grad_ndof

        else:
            raise ValueError('Wrong values for grad_x0 and/or grad_v0')

        grad_x_keep_np = np.empty((nint_keep, ndof, grad_ndof), dtype=np.float64)
        grad_v_keep_np = np.empty((nint_keep, ndof, grad_ndof), dtype=np.float64)

        grad_x_keep = grad_x_keep_np
        grad_v_keep = grad_v_keep_np

        grad_K_fun = np.zeros((nsteps, ndof, grad_ndof), dtype=np.float64)
        grad_K_gun = np.zeros((nsteps, ndof, grad_ndof), dtype=np.float64)
        grad_dX = np.empty((nsteps, ndof, grad_ndof), dtype=np.float64)
        grad_dV = np.empty((nsteps, ndof, grad_ndof), dtype=np.float64) 
        grad_dX_prev = np.empty((nsteps, ndof, grad_ndof), dtype=np.float64)
        grad_dV_prev = np.empty((nsteps, ndof, grad_ndof), dtype=np.float64) 

        if grad_fun is None:
            raise ValueError(f"grad_fun was not provided")
        if grad_gun is None:
            raise ValueError(f"grad_gun was not provided")

        ccallback_prepare(&callback_grad_fun, signatures, grad_fun, CCALLBACK_DEFAULTS)
        lowlevelgrad_fun = LowLevelFun_init(callback_grad_fun)

        ccallback_prepare(&callback_grad_gun, signatures, grad_gun, CCALLBACK_DEFAULTS)
        lowlevelgrad_gun = LowLevelFun_init(callback_grad_gun)

        if (lowlevelgrad_fun.fun_type == PY_FUN) != (lowlevelgrad_gun.fun_type == PY_FUN):
            raise ValueError("fun and gun must both be python functions or LowLevelCallables")

        if lowlevelgrad_fun.fun_type == PY_FUN:
            py_grad_fun = <object> lowlevelgrad_fun.py_fun
            py_grad_gun = <object> lowlevelgrad_gun.py_fun
            
            py_grad_fun_res = py_grad_fun(t_span[0], v0, grad_v)
            py_grad_gun_res = py_grad_gun(t_span[0], x0, grad_x)

            if isinstance(py_grad_fun_res, float) and isinstance(py_grad_gun_res, float):
                py_grad_fun_type = PY_FUN_FLOAT
            elif isinstance(py_fun_res, np.ndarray) and isinstance(py_grad_gun_res, np.ndarray):
                py_grad_fun_type = PY_FUN_NDARRAY
            else:
                raise ValueError(f"Could not recognize return type of python callable. Found {type(py_grad_fun_res)} and {type(py_grad_gun_res)}.")

        else:
            py_grad_fun = None
            py_grad_gun = None

    else:

        grad_x = np.zeros((0, 0), dtype=np.float64)
        grad_v = np.zeros((0, 0), dtype=np.float64)

        grad_x_keep_np = np.empty((0, 0, 0), dtype=np.float64)
        grad_v_keep_np = np.empty((0, 0, 0), dtype=np.float64)

        grad_x_keep = grad_x_keep_np
        grad_v_keep = grad_v_keep_np

        grad_K_fun = np.zeros((0, 0, 0), dtype=np.float64)
        grad_K_gun = np.zeros((0, 0, 0), dtype=np.float64)
        grad_dX = np.empty((0, 0, 0), dtype=np.float64)
        grad_dV = np.empty((0, 0, 0), dtype=np.float64) 
        grad_dX_prev = np.empty((0, 0, 0), dtype=np.float64)
        grad_dV_prev = np.empty((0, 0, 0), dtype=np.float64) 

        py_grad_fun = None
        py_grad_gun = None        

    with nogil:
    
        ImplicitSymplecticIVP_ann(
            lowlevelfun     ,
            lowlevelgun     ,
            lowlevelgrad_fun,
            lowlevelgrad_gun,
            py_fun          ,
            py_gun          ,
            py_grad_fun     ,
            py_grad_gun     ,
            py_fun_type     ,
            t_span          ,
            x               ,
            v               ,
            grad_x          ,
            grad_v          ,
            K_fun           ,
            K_gun           ,
            grad_K_fun      ,
            grad_K_gun      ,
            dX              ,
            dV              ,
            grad_dX         ,
            grad_dV         ,
            dX_prev         ,
            dV_prev         ,
            grad_dX_prev    ,
            grad_dV_prev    ,
            all_t_x         ,
            all_t_v         ,
            rk_x._a_table   ,
            rk_x._b_table   ,
            rk_x._c_table   ,
            rk_x._beta_table,
            rk_v._a_table   ,
            rk_v._b_table   ,
            rk_v._c_table   ,
            rk_v._beta_table,
            nint            ,
            keep_freq       ,
            DoEFT           ,
            DoTanIntegration,
            eps             ,
            maxiter         ,
            x_keep          ,
            v_keep          ,
            grad_x_keep     ,
            grad_v_keep     ,
        )

    ccallback_release(&callback_fun)
    ccallback_release(&callback_gun)

    if DoTanIntegration:

        ccallback_release(&callback_grad_fun)
        ccallback_release(&callback_grad_gun)

        return x_keep_np, v_keep_np, grad_x_keep_np, grad_v_keep_np
    
    else:
        
        return x_keep_np, v_keep_np

@cython.cdivision(True)
cdef void ImplicitSymplecticIVP_ann(
    const LowLevelFun lowlevelfun       ,
    const LowLevelFun lowlevelgun       ,
    const LowLevelFun lowlevelgrad_fun  ,
    const LowLevelFun lowlevelgrad_gun  ,
          object py_fun                 ,
          object py_gun                 ,
          object py_grad_fun            ,
          object py_grad_gun            ,
    const int py_fun_type               ,
    const (double, double) t_span       ,
          double[::1]     x             ,
          double[::1]     v             ,
          double[:,::1]   grad_x        ,
          double[:,::1]   grad_v        ,
          double[:,::1]   K_fun         ,
          double[:,::1]   K_gun         ,
          double[:,:,::1] grad_K_fun    ,
          double[:,:,::1] grad_K_gun    ,
          double[:,::1]   dX            ,
          double[:,::1]   dV            ,
          double[:,:,::1] grad_dX       ,
          double[:,:,::1] grad_dV       ,
          double[:,::1]   dX_prev       ,
          double[:,::1]   dV_prev       ,
          double[:,:,::1] grad_dX_prev  ,
          double[:,:,::1] grad_dV_prev  ,
          double[::1]     all_t_x       ,
          double[::1]     all_t_v       ,
    const double[:,::1]   a_table_x     ,
    const double[::1]     b_table_x     ,
    const double[::1]     c_table_x     ,
    const double[:,::1]   beta_table_x  ,
    const double[:,::1]   a_table_v     ,
    const double[::1]     b_table_v     ,
    const double[::1]     c_table_v     ,
    const double[:,::1]   beta_table_v  ,
    const long nint                     ,
    const long keep_freq                ,
    const bint DoEFT                    ,
    const bint DoTanIntegration         ,
    const double eps                    ,
    const long maxiter                  ,
          double[:,::1]   x_keep        ,
          double[:,::1]   v_keep        ,
          double[:,:,::1] grad_x_keep   ,
          double[:,:,::1] grad_v_keep   ,
) noexcept nogil:

    cdef int ndof = x.shape[0]
    cdef int grad_ndof
    cdef long iGS
    cdef Py_ssize_t istep, jdof
    cdef Py_ssize_t iint_keep, ifreq
    cdef long iint
    cdef long tot_niter = 0
    cdef long grad_tot_niter = 0
    cdef long nint_keep = nint // keep_freq

    cdef bint GoOnGS

    cdef double dXV_err, dX_err, dV_err, diff
    cdef double tbeg
    cdef double dt = (t_span[1] - t_span[0]) / nint
    cdef int nsteps = a_table_x.shape[0]
    cdef int dX_size = nsteps*ndof
    cdef double eps_mul = eps * dX_size * dt
    cdef double grad_eps_mul
    cdef int grad_nvar
    cdef int grad_dX_size

    if DoTanIntegration:
        grad_ndof = grad_x.shape[1]
        grad_nvar = ndof * grad_ndof
        grad_dX_size = nsteps * grad_nvar
        grad_eps_mul = eps * grad_dX_size * dt

    cdef double *dxv
    cdef double *x_eft_comp
    cdef double *v_eft_comp
    cdef double *grad_dxv
    cdef double *grad_x_eft_comp
    cdef double *grad_v_eft_comp
    cdef double tx_comp = 0.
    cdef double tv_comp = 0.

    if DoEFT:

        dxv = <double *> malloc(sizeof(double) * ndof)

        x_eft_comp = <double *> malloc(sizeof(double) * ndof)
        for istep in range(ndof):
            x_eft_comp[istep] = 0.

        v_eft_comp = <double *> malloc(sizeof(double) * ndof)
        for istep in range(ndof):
            v_eft_comp[istep] = 0.

        if DoTanIntegration:

            grad_dxv = <double *> malloc(sizeof(double) * grad_nvar)

            grad_x_eft_comp = <double *> malloc(sizeof(double) * grad_nvar)
            for istep in range(grad_nvar):
                grad_x_eft_comp[istep] = 0.

            grad_v_eft_comp = <double *> malloc(sizeof(double) * grad_nvar)
            for istep in range(grad_nvar):
                grad_v_eft_comp[istep] = 0.

    cdef double *cdt_x = <double *> malloc(sizeof(double) * nsteps)
    cdef double *cdt_v = <double *> malloc(sizeof(double) * nsteps)

    for istep in range(nsteps):
        cdt_v[istep] = c_table_v[istep]*dt

    for istep in range(nsteps):
        cdt_x[istep] = c_table_x[istep]*dt

    for iint_keep in range(nint_keep):

        for ifreq in range(keep_freq):

            iint = iint_keep * keep_freq + ifreq

            tbeg = t_span[0] + iint * dt    
            for istep in range(nsteps):
                all_t_v[istep] = tbeg + cdt_v[istep]

            for istep in range(nsteps):
                all_t_x[istep] = tbeg + cdt_x[istep]

            # dV = beta_table_v . K_gun
            scipy.linalg.cython_blas.dgemm(transn,transn,&ndof,&nsteps,&nsteps,&one,&K_gun[0,0],&ndof,&beta_table_v[0,0],&nsteps,&zero,&dV[0,0],&ndof)

            # dX = beta_table_x . K_fun
            scipy.linalg.cython_blas.dgemm(transn,transn,&ndof,&nsteps,&nsteps,&one,&K_fun[0,0],&ndof,&beta_table_x[0,0],&nsteps,&zero,&dX[0,0],&ndof)

            iGS = 0
            GoOnGS = True

            while GoOnGS:

                # dV_prev = dV
                scipy.linalg.cython_blas.dcopy(&dX_size,&dV[0,0],&int_one,&dV_prev[0,0],&int_one)

                # dX_prev = dX
                scipy.linalg.cython_blas.dcopy(&dX_size,&dX[0,0],&int_one,&dX_prev[0,0],&int_one)

                # K_fun = dt * fun(t,v+dV)
                for istep in range(nsteps):
                    scipy.linalg.cython_blas.daxpy(&ndof,&one,&v[0],&int_one,&dV[istep,0],&int_one)

                if py_fun_type > 0:
                    with gil:
                        PyFun_apply_vectorized(py_fun, py_fun_type, all_t_v, dV, K_fun)
                else:
                    LowLevelFun_apply_vectorized(lowlevelfun, all_t_v, dV, K_fun)

                scipy.linalg.cython_blas.dscal(&dX_size,&dt,&K_fun[0,0],&int_one)

                # dX = a_table_x .K_fun
                scipy.linalg.cython_blas.dgemm(transn,transn,&ndof,&nsteps,&nsteps,&one,&K_fun[0,0],&ndof,&a_table_x[0,0],&nsteps,&zero,&dX[0,0],&ndof)

                # K_gun = dt * gun(t,x+dX)
                for istep in range(nsteps):
                    scipy.linalg.cython_blas.daxpy(&ndof,&one,&x[0],&int_one,&dX[istep,0],&int_one)

                if py_fun_type > 0:
                    with gil:
                        PyFun_apply_vectorized(py_gun, py_fun_type, all_t_x, dX, K_gun)
                else:
                    LowLevelFun_apply_vectorized(lowlevelgun, all_t_x, dX, K_gun)

                scipy.linalg.cython_blas.dscal(&dX_size,&dt,&K_gun[0,0],&int_one)

                # dV = a_table_v . K_gun
                scipy.linalg.cython_blas.dgemm(transn,transn,&ndof,&nsteps,&nsteps,&one,&K_gun[0,0],&ndof,&a_table_v[0,0],&nsteps,&zero,&dV[0,0],&ndof)

                # dX_prev = dX_prev - dX
                scipy.linalg.cython_blas.daxpy(&dX_size,&minus_one,&dX[0,0],&int_one,&dX_prev[0,0],&int_one)

                # dV_prev = dV_prev - dV
                scipy.linalg.cython_blas.daxpy(&dX_size,&minus_one,&dV[0,0],&int_one,&dV_prev[0,0],&int_one)

                dX_err = scipy.linalg.cython_blas.dasum(&dX_size,&dX_prev[0,0],&int_one)
                dV_err = scipy.linalg.cython_blas.dasum(&dX_size,&dV_prev[0,0],&int_one)
                dXV_err = dX_err + dV_err  

                iGS += 1

                GoOnGS = (iGS < maxiter) and (dXV_err > eps_mul)

            # if (iGS >= maxiter):
                # print("Max iter exceeded. Rel error : ",dX_err/eps_mul,dV_err/eps_mul)

            tot_niter += iGS

            if DoTanIntegration:

                iGS = 0
                GoOnGS = True

                # ONLY dV here! dX was updated before !!

                # dV = v + dV
                for istep in range(nsteps):
                    scipy.linalg.cython_blas.daxpy(&ndof,&one,&v[0],&int_one,&dV[istep,0],&int_one)

                # grad_dV = beta_table_v . grad_K_gun
                scipy.linalg.cython_blas.dgemm(transn,transn,&grad_nvar,&nsteps,&nsteps,&one,&grad_K_gun[0,0,0],&grad_nvar,&beta_table_v[0,0],&nsteps,&zero,&grad_dV[0,0,0],&grad_nvar)

                # grad_dX = beta_table_x . grad_K_fun
                scipy.linalg.cython_blas.dgemm(transn,transn,&grad_nvar,&nsteps,&nsteps,&one,&grad_K_fun[0,0,0],&grad_nvar,&beta_table_x[0,0],&nsteps,&zero,&grad_dX[0,0,0],&grad_nvar)

                while GoOnGS:

                    # grad_dV_prev = grad_dV
                    scipy.linalg.cython_blas.dcopy(&grad_dX_size,&grad_dV[0,0,0],&int_one,&grad_dV_prev[0,0,0],&int_one)

                    # grad_dX_prev = grad_dX
                    scipy.linalg.cython_blas.dcopy(&grad_dX_size,&grad_dX[0,0,0],&int_one,&grad_dX_prev[0,0,0],&int_one)

                    # grad_K_fun = dt * grad_fun(t,grad_v+grad_dV)
                    for istep in range(nsteps):
                        scipy.linalg.cython_blas.daxpy(&grad_nvar,&one,&grad_v[0,0],&int_one,&grad_dV[istep,0,0],&int_one)

                    if py_fun_type > 0:
                        with gil:
                            PyFun_apply_grad_vectorized(py_grad_fun, py_fun_type, all_t_v, dV, grad_dV, grad_K_fun)
                    else:
                        LowLevelFun_apply_grad_vectorized(lowlevelgrad_fun, all_t_v, dV, grad_dV, grad_K_fun)

                    scipy.linalg.cython_blas.dscal(&grad_dX_size,&dt,&grad_K_fun[0,0,0],&int_one)

                    # grad_dX = a_table_x . grad_K_fun
                    scipy.linalg.cython_blas.dgemm(transn,transn,&grad_nvar,&nsteps,&nsteps,&one,&grad_K_fun[0,0,0],&grad_nvar,&a_table_x[0,0],&nsteps,&zero,&grad_dX[0,0,0],&grad_nvar)

                    # grad_K_gun = dt * grad_gun(t,grad_x+grad_dX)
                    for istep in range(nsteps):
                        scipy.linalg.cython_blas.daxpy(&grad_nvar,&one,&grad_x[0,0],&int_one,&grad_dX[istep,0,0],&int_one)

                    if py_fun_type > 0:
                        with gil:
                            PyFun_apply_grad_vectorized(py_grad_gun, py_fun_type, all_t_x, dX, grad_dX, grad_K_gun)
                    else:
                        LowLevelFun_apply_grad_vectorized(lowlevelgrad_gun, all_t_x, dX, grad_dX, grad_K_gun)

                    scipy.linalg.cython_blas.dscal(&grad_dX_size,&dt,&grad_K_gun[0,0,0],&int_one)

                    # grad_dV = a_table_v . grad_K_gun
                    scipy.linalg.cython_blas.dgemm(transn,transn,&grad_nvar,&nsteps,&nsteps,&one,&grad_K_gun[0,0,0],&grad_nvar,&a_table_v[0,0],&nsteps,&zero,&grad_dV[0,0,0],&grad_nvar)

                    # grad_dX_prev = grad_dX_prev - grad_dX
                    scipy.linalg.cython_blas.daxpy(&grad_dX_size,&minus_one,&grad_dX[0,0,0],&int_one,&grad_dX_prev[0,0,0],&int_one)

                    # grad_dV_prev = grad_dV_prev - grad_dV
                    scipy.linalg.cython_blas.daxpy(&grad_dX_size,&minus_one,&grad_dV[0,0,0],&int_one,&grad_dV_prev[0,0,0],&int_one)

                    dX_err = scipy.linalg.cython_blas.dasum(&grad_dX_size,&grad_dX_prev[0,0,0],&int_one)
                    dV_err = scipy.linalg.cython_blas.dasum(&grad_dX_size,&grad_dV_prev[0,0,0],&int_one)
                    dXV_err = dX_err + dV_err  

                    iGS += 1

                    GoOnGS = (iGS < maxiter) and (dXV_err > grad_eps_mul)

                # if (iGS >= maxiter):
                #     with gil:
                #         print("Tangent Max iter exceeded. Rel error : ",dXV_err/grad_eps_mul,iint)

                grad_tot_niter += iGS

            if DoEFT:

                # dxv = b_table_x^T . K_fun
                scipy.linalg.cython_blas.dgemv(transn,&ndof,&nsteps,&one,&K_fun[0,0],&ndof,&b_table_x[0],&int_one,&zero,dxv,&int_one)
                # x = x + dxv
                TwoSum_incr(&x[0],dxv,x_eft_comp,ndof)

                # dxv = b_table_v^T . K_gun
                scipy.linalg.cython_blas.dgemv(transn,&ndof,&nsteps,&one,&K_gun[0,0],&ndof,&b_table_v[0],&int_one,&zero,dxv,&int_one)
                # v = v + dxv
                TwoSum_incr(&v[0],dxv,v_eft_comp,ndof)

                if DoTanIntegration:

                    # grad_dxv = b_table_x^T . grad_K_fun
                    scipy.linalg.cython_blas.dgemv(transn,&grad_nvar,&nsteps,&one,&grad_K_fun[0,0,0],&grad_nvar,&b_table_x[0],&int_one,&zero,grad_dxv,&int_one)
                    # grad_x = grad_x + grad_dxv
                    TwoSum_incr(&grad_x[0,0],grad_dxv,grad_x_eft_comp,grad_nvar)

                    # grad_dxv = b_table_v^T . grad_K_gun
                    scipy.linalg.cython_blas.dgemv(transn,&grad_nvar,&nsteps,&one,&grad_K_gun[0,0,0],&grad_nvar,&b_table_v[0],&int_one,&zero,grad_dxv,&int_one)
                    # grad_v = grad_v + grad_dxv
                    TwoSum_incr(&grad_v[0,0],grad_dxv,grad_v_eft_comp,grad_nvar)

            else:

                # x = x + b_table_x^T . K_fun
                scipy.linalg.cython_blas.dgemv(transn,&ndof,&nsteps,&one,&K_fun[0,0],&ndof,&b_table_x[0],&int_one,&one,&x[0],&int_one)

                # v = v + b_table_v^T . K_gun
                scipy.linalg.cython_blas.dgemv(transn,&ndof,&nsteps,&one,&K_gun[0,0],&ndof,&b_table_v[0],&int_one,&one,&v[0],&int_one)

                if DoTanIntegration:

                    # grad_x = grad_x + b_table_x^T . grad_K_fun
                    scipy.linalg.cython_blas.dgemv(transn,&grad_nvar,&nsteps,&one,&grad_K_fun[0,0,0],&grad_nvar,&b_table_x[0],&int_one,&one,&grad_x[0,0],&int_one)

                    # grad_v = grad_v + b_table_v^T . grad_K_gun
                    scipy.linalg.cython_blas.dgemv(transn,&grad_nvar,&nsteps,&one,&grad_K_gun[0,0,0],&grad_nvar,&b_table_v[0],&int_one,&one,&grad_v[0,0],&int_one)


        scipy.linalg.cython_blas.dcopy(&ndof,&x[0],&int_one,&x_keep[iint_keep,0],&int_one)
        scipy.linalg.cython_blas.dcopy(&ndof,&v[0],&int_one,&v_keep[iint_keep,0],&int_one)

        if DoTanIntegration:

            scipy.linalg.cython_blas.dcopy(&grad_nvar,&grad_x[0,0],&int_one,&grad_x_keep[iint_keep,0,0],&int_one)
            scipy.linalg.cython_blas.dcopy(&grad_nvar,&grad_v[0,0],&int_one,&grad_v_keep[iint_keep,0,0],&int_one)

    # print(tot_niter / nint)
    # print(1+nsteps*tot_niter)

    free(cdt_x)
    free(cdt_v)

    if DoEFT:

        free(dxv)
        free(x_eft_comp)
        free(v_eft_comp)

        if DoTanIntegration:

            free(grad_dxv)
            free(grad_x_eft_comp)
            free(grad_v_eft_comp)
