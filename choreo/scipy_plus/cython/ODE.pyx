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
cdef int C_FUN_MEMORYVIEW_VEC = 1
cdef int C_FUN_POINTER = 2
cdef int C_FUN_POINTER_VEC = 3
cdef int C_FUN_MEMORYVIEW_DATA = 4
cdef int C_FUN_MEMORYVIEW_VEC_DATA = 5
cdef int C_FUN_POINTER_DATA = 6
cdef int C_FUN_POINTER_VEC_DATA = 7
cdef int C_GRAD_FUN_MEMORYVIEW = 8
cdef int C_GRAD_FUN_MEMORYVIEW_VEC = 9
cdef int C_GRAD_FUN_POINTER = 10
cdef int C_GRAD_FUN_POINTER_VEC = 11
cdef int C_GRAD_FUN_MEMORYVIEW_DATA = 12
cdef int C_GRAD_FUN_MEMORYVIEW_VEC_DATA = 13
cdef int C_GRAD_FUN_POINTER_DATA = 14
cdef int C_GRAD_FUN_POINTER_VEC_DATA = 15
cdef int N_SIGNATURES = 16
cdef ccallback_signature_t signatures[17]

ctypedef void (*c_fun_type_memoryview)(double, double[::1], double[::1]) noexcept nogil 
signatures[C_FUN_MEMORYVIEW].signature = b"void (double, __Pyx_memviewslice, __Pyx_memviewslice)"
signatures[C_FUN_MEMORYVIEW].value = C_FUN_MEMORYVIEW

ctypedef void (*c_fun_type_memoryview_vec)(double[::1], double[:,::1], double[:,::1]) noexcept nogil 
signatures[C_FUN_MEMORYVIEW_VEC].signature = b"void (__Pyx_memviewslice, __Pyx_memviewslice, __Pyx_memviewslice)"
signatures[C_FUN_MEMORYVIEW_VEC].value = C_FUN_MEMORYVIEW_VEC

ctypedef void (*c_fun_type_pointer)(double, double*, double*) noexcept nogil 
signatures[C_FUN_POINTER].signature = b"void (double, double *, double *)"
signatures[C_FUN_POINTER].value = C_FUN_POINTER

ctypedef void (*c_fun_type_pointer_vec)(double*, double*, double*) noexcept nogil 
signatures[C_FUN_POINTER_VEC].signature = b"void (double *, double *, double *)"
signatures[C_FUN_POINTER_VEC].value = C_FUN_POINTER_VEC

ctypedef void (*c_fun_type_memoryview_data)(double, double[::1], double[::1], void*) noexcept nogil 
signatures[C_FUN_MEMORYVIEW_DATA].signature = b"void (double, __Pyx_memviewslice, __Pyx_memviewslice, void *)"
signatures[C_FUN_MEMORYVIEW_DATA].value = C_FUN_MEMORYVIEW_DATA

ctypedef void (*c_fun_type_memoryview_vec_data)(double[::1], double[:,::1], double[:,::1], void*) noexcept nogil 
signatures[C_FUN_MEMORYVIEW_VEC_DATA].signature = b"void (__Pyx_memviewslice, __Pyx_memviewslice, __Pyx_memviewslice, void *)"
signatures[C_FUN_MEMORYVIEW_VEC_DATA].value = C_FUN_MEMORYVIEW_VEC_DATA

ctypedef void (*c_fun_type_pointer_data)(double, double*, double*, void*) noexcept nogil 
signatures[C_FUN_POINTER_DATA].signature = b"void (double, double *, double *, void *)"
signatures[C_FUN_POINTER_DATA].value = C_FUN_POINTER_DATA

ctypedef void (*c_fun_type_pointer_vec_data)(double*, double*, double*, void*) noexcept nogil 
signatures[C_FUN_POINTER_VEC_DATA].signature = b"void (double *, double *, double *, void *)"
signatures[C_FUN_POINTER_VEC_DATA].value = C_FUN_POINTER_VEC_DATA

ctypedef void (*c_grad_fun_type_memoryview)(double, double[::1], double[:,::1], double[:,::1]) noexcept nogil 
signatures[C_GRAD_FUN_MEMORYVIEW].signature = b"void (double, __Pyx_memviewslice, __Pyx_memviewslice, __Pyx_memviewslice)"
signatures[C_GRAD_FUN_MEMORYVIEW].value = C_GRAD_FUN_MEMORYVIEW

ctypedef void (*c_grad_fun_type_memoryview_vec)(double[::1], double[:,::1], double[:,:,::1], double[:,:,::1]) noexcept nogil 
signatures[C_GRAD_FUN_MEMORYVIEW_VEC].signature = b"void (__Pyx_memviewslice, __Pyx_memviewslice, __Pyx_memviewslice, __Pyx_memviewslice)"
signatures[C_GRAD_FUN_MEMORYVIEW_VEC].value = C_GRAD_FUN_MEMORYVIEW_VEC

ctypedef void (*c_grad_fun_type_pointer)(double, double*, double*, double*) noexcept nogil 
signatures[C_GRAD_FUN_POINTER].signature = b"void (double, double *, double *, double *)"
signatures[C_GRAD_FUN_POINTER].value = C_GRAD_FUN_POINTER

ctypedef void (*c_grad_fun_type_pointer_vec)(double*, double*, double*, double*) noexcept nogil 
signatures[C_GRAD_FUN_POINTER_VEC].signature = b"void (double *, double *, double *, double *)"
signatures[C_GRAD_FUN_POINTER_VEC].value = C_GRAD_FUN_POINTER_VEC

ctypedef void (*c_grad_fun_type_memoryview_data)(double, double[::1], double[:,::1], double[:,::1], void*) noexcept nogil 
signatures[C_GRAD_FUN_MEMORYVIEW_DATA].signature = b"void (double, __Pyx_memviewslice, __Pyx_memviewslice, __Pyx_memviewslice, void *)"
signatures[C_GRAD_FUN_MEMORYVIEW_DATA].value = C_GRAD_FUN_MEMORYVIEW_DATA

ctypedef void (*c_grad_fun_type_memoryview_vec_data)(double[::1], double[:,::1], double[:,:,::1], double[:,:,::1], void*) noexcept nogil 
signatures[C_GRAD_FUN_MEMORYVIEW_VEC_DATA].signature = b"void (__Pyx_memviewslice, __Pyx_memviewslice, __Pyx_memviewslice, __Pyx_memviewslice, void *)"
signatures[C_GRAD_FUN_MEMORYVIEW_VEC_DATA].value = C_GRAD_FUN_MEMORYVIEW_VEC_DATA

ctypedef void (*c_grad_fun_type_pointer_data)(double, double*, double*, double*, void*) noexcept nogil 
signatures[C_GRAD_FUN_POINTER_DATA].signature = b"void (double, double *, double *, double *, void *)"
signatures[C_GRAD_FUN_POINTER_DATA].value = C_GRAD_FUN_POINTER_DATA

ctypedef void (*c_grad_fun_type_pointer_vec_data)(double*, double*, double*, double*, void*) noexcept nogil 
signatures[C_GRAD_FUN_POINTER_VEC_DATA].signature = b"void (double *, double *, double *, double *, void *)"
signatures[C_GRAD_FUN_POINTER_VEC_DATA].value = C_GRAD_FUN_POINTER_VEC_DATA

signatures[N_SIGNATURES].signature = NULL

cdef inline void LowLevelFun_apply(
    ccallback_t callback    ,
    double t                ,
    double[::1] x           ,
    double[::1] res         ,
) noexcept nogil:

    cdef int n
    cdef double[::1] res_1D

    if (callback.py_function == NULL):

        if (callback.user_data == NULL):
            
            if callback.signature.value == C_FUN_MEMORYVIEW:

                (<c_fun_type_memoryview> callback.c_function)(t, x, res)

            elif callback.signature.value == C_FUN_POINTER:

                (<c_fun_type_pointer> callback.c_function)(t, &x[0], &res[0])

            else:
                with gil:
                    raise ValueError("Incompatible function signature.")

        else:

            if callback.signature.value == C_FUN_MEMORYVIEW_DATA:

                (<c_fun_type_memoryview_data> callback.c_function)(t, x, res, callback.user_data)

            elif callback.signature.value == C_FUN_POINTER_DATA:

                (<c_fun_type_pointer_data> callback.c_function)(t, &x[0], &res[0], callback.user_data)

            else:
                with gil:
                    raise ValueError("Incompatible function signature.")
    else:

        with gil:
            res_1D =  (<object> callback.py_function)(t, x)

            n = res.shape[0]
            scipy.linalg.cython_blas.dcopy(&n,&res_1D[0],&int_one,&res[0],&int_one)

cdef inline void LowLevelFun_grad_apply(
    ccallback_t callback    ,
    double t                ,
    double[::1] x           ,
    double[:,::1] grad_x    ,
    double[:,::1] res       ,
) noexcept nogil:

    cdef int n
    cdef double[:,::1] res_2D

    if (callback.py_function == NULL):

        if (callback.user_data == NULL):
            
            if callback.signature.value == C_GRAD_FUN_MEMORYVIEW:

                (<c_grad_fun_type_memoryview> callback.c_function)(t, x, grad_x, res)

            elif callback.signature.value == C_GRAD_FUN_POINTER:

                (<c_grad_fun_type_pointer> callback.c_function)(t, &x[0], &grad_x[0,0], &res[0,0])

            else:
                with gil:
                    raise ValueError("Incompatible function signature.")
        else:

            if callback.signature.value == C_GRAD_FUN_MEMORYVIEW_DATA:

                (<c_grad_fun_type_memoryview_data> callback.c_function)(t, x, grad_x, res, callback.user_data)

            elif callback.signature.value == C_GRAD_FUN_POINTER_DATA:

                (<c_grad_fun_type_pointer_data> callback.c_function)(t, &x[0], &grad_x[0,0], &res[0,0], callback.user_data)

            else:
                with gil:
                    raise ValueError("Incompatible function signature.")
    else:

        with gil:
            res_2D = (<object> callback.py_function)(t, x, grad_x)

            n = res.shape[0] * res.shape[1]
            scipy.linalg.cython_blas.dcopy(&n,&res_2D[0,0],&int_one,&res[0,0],&int_one)

cdef inline void LowLevelFun_apply_vectorized(
    bint vector_calls       ,
    ccallback_t callback    ,
    double[::1] all_t       ,
    double[:,::1] all_x     ,
    double[:,::1] all_res   ,
) noexcept nogil:

    cdef Py_ssize_t i
    cdef int n

    cdef double[::1] res_1D
    cdef double[:,::1] res_2D

    if (callback.py_function == NULL):

        if (callback.user_data == NULL):
                
            if vector_calls:

                if callback.signature.value == C_FUN_MEMORYVIEW_VEC:

                    (<c_fun_type_memoryview_vec> callback.c_function)(all_t, all_x, all_res)

                elif callback.signature.value == C_FUN_POINTER_VEC:

                    (<c_fun_type_pointer_vec> callback.c_function)(&all_t[0], &all_x[0,0], &all_res[0,0])

                else:
                    with gil:
                        raise ValueError("Incompatible function signature.")

            else:

                if callback.signature.value == C_FUN_MEMORYVIEW:

                    for i in range(all_t.shape[0]):
                        (<c_fun_type_memoryview> callback.c_function)(all_t[i], all_x[i,:], all_res[i,:])

                elif callback.signature.value == C_FUN_POINTER:

                    for i in range(all_t.shape[0]):
                        (<c_fun_type_pointer> callback.c_function)(all_t[i], &all_x[i,0], &all_res[i,0])

                else:
                    with gil:
                        raise ValueError("Incompatible function signature.")

        else:
                
            if vector_calls:

                if callback.signature.value == C_FUN_MEMORYVIEW_VEC_DATA:

                    (<c_fun_type_memoryview_vec_data> callback.c_function)(all_t, all_x, all_res, callback.user_data)

                elif callback.signature.value == C_FUN_POINTER_VEC_DATA:

                    (<c_fun_type_pointer_vec_data> callback.c_function)(&all_t[0], &all_x[0,0], &all_res[0,0], callback.user_data)

                else:
                    with gil:
                        raise ValueError("Incompatible function signature.")

            else:

                if callback.signature.value == C_FUN_MEMORYVIEW_DATA:

                    for i in range(all_t.shape[0]):
                        (<c_fun_type_memoryview_data> callback.c_function)(all_t[i], all_x[i,:], all_res[i,:], callback.user_data)

                elif callback.signature.value == C_FUN_POINTER_DATA:

                    for i in range(all_t.shape[0]):
                        (<c_fun_type_pointer_data> callback.c_function)(all_t[i], &all_x[i,0], &all_res[i,0], callback.user_data)

                else:
                    with gil:
                        raise ValueError("Incompatible function signature.")

    else:

        with gil:

            if vector_calls:

                res_2D = (<object> callback.py_function)(all_t, all_x)

                n = all_res.shape[0] * all_res.shape[1]
                scipy.linalg.cython_blas.dcopy(&n,&res_2D[0,0],&int_one,&all_res[0,0],&int_one)

            else:

                n = all_res.shape[1]

                for i in range(all_t.shape[0]):  

                    res_1D = (<object> callback.py_function)(all_t[i], all_x[i,:])

                    scipy.linalg.cython_blas.dcopy(&n,&res_1D[0],&int_one,&all_res[i,0],&int_one)                    

cdef inline void LowLevelFun_apply_grad_vectorized(
    bint vector_calls           ,
    ccallback_t callback        ,
    double[::1] all_t           ,
    double[:,::1] all_x         ,
    double[:,:,::1] all_grad_x  ,
    double[:,:,::1] all_res     ,
) noexcept nogil:

    cdef Py_ssize_t i
    cdef int n

    cdef double[:,::1] res_2D
    cdef double[:,:,::1] res_3D

    if (callback.py_function == NULL):
            
        if (callback.user_data == NULL):
                    
            if vector_calls:

                if callback.signature.value == C_GRAD_FUN_MEMORYVIEW_VEC:

                    (<c_grad_fun_type_memoryview_vec> callback.c_function)(all_t, all_x, all_grad_x, all_res)

                elif callback.signature.value == C_GRAD_FUN_POINTER_VEC:

                    (<c_grad_fun_type_pointer_vec> callback.c_function)(&all_t[0], &all_x[0,0], &all_grad_x[0,0,0], &all_res[0,0,0])

                else:
                    with gil:
                        raise ValueError("Incompatible function signature.")
            else:

                if callback.signature.value == C_GRAD_FUN_MEMORYVIEW:

                    for i in range(all_t.shape[0]):
                        (<c_grad_fun_type_memoryview> callback.c_function)(all_t[i], all_x[i,:], all_grad_x[i,:,:], all_res[i,:,:])

                elif callback.signature.value == C_GRAD_FUN_POINTER:

                    for i in range(all_t.shape[0]):
                        (<c_grad_fun_type_pointer> callback.c_function)(all_t[i], &all_x[i,0], &all_grad_x[i,0,0], &all_res[i,0,0])

                else:
                    with gil:
                        raise ValueError("Incompatible function signature.")
        
        else:

            if vector_calls:

                if callback.signature.value == C_GRAD_FUN_MEMORYVIEW_VEC_DATA:

                    (<c_grad_fun_type_memoryview_vec_data> callback.c_function)(all_t, all_x, all_grad_x, all_res, callback.user_data)

                elif callback.signature.value == C_GRAD_FUN_POINTER_VEC_DATA:

                    (<c_grad_fun_type_pointer_vec_data> callback.c_function)(&all_t[0], &all_x[0,0], &all_grad_x[0,0,0], &all_res[0,0,0], callback.user_data)

                else:
                    with gil:
                        raise ValueError("Incompatible function signature.")
            else:

                if callback.signature.value == C_GRAD_FUN_MEMORYVIEW_DATA:

                    for i in range(all_t.shape[0]):
                        (<c_grad_fun_type_memoryview_data> callback.c_function)(all_t[i], all_x[i,:], all_grad_x[i,:,:], all_res[i,:,:], callback.user_data)

                elif callback.signature.value == C_GRAD_FUN_POINTER_DATA:

                    for i in range(all_t.shape[0]):
                        (<c_grad_fun_type_pointer_data> callback.c_function)(all_t[i], &all_x[i,0], &all_grad_x[i,0,0], &all_res[i,0,0], callback.user_data)

                else:
                    with gil:
                        raise ValueError("Incompatible function signature.")
                        
    else:

        with gil:

            if vector_calls:

                res_3D = (<object> callback.py_function)(all_t, all_x, all_grad_x)

                n = all_res.shape[0] * all_res.shape[1] * all_res.shape[2]
                scipy.linalg.cython_blas.dcopy(&n,&res_3D[0,0,0],&int_one,&all_res[0,0,0],&int_one)

            else:

                n = all_res.shape[1] * all_res.shape[2]

                for i in range(all_t.shape[0]):  

                    res_2D = (<object> callback.py_function)(all_t[i], all_x[i,:], all_grad_x[i,:,:])

                    scipy.linalg.cython_blas.dcopy(&n,&res_2D[0,0],&int_one,&all_res[i,0,0],&int_one)

@cython.final
cdef class ExplicitSymplecticRKTable:
    
    cdef double[::1] _c_table
    cdef double[::1] _d_table
    cdef Py_ssize_t _th_cvg_rate

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
    
    @cython.final
    @property
    def nsteps(self):
        return self._c_table.shape[0]

    @cython.final
    @property
    def c_table(self):
        return np.asarray(self._c_table)
    
    @cython.final
    @property
    def d_table(self):
        return np.asarray(self._d_table)    

    @cython.final
    @property
    def th_cvg_rate(self):
        if self._th_cvg_rate > 0:
            return self._th_cvg_rate
        else:
            return None

    @cython.final
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
    Py_ssize_t nint = 1             ,
    Py_ssize_t keep_freq = -1       ,
    bint keep_init = False          ,
    bint DoEFT = True               ,
): 

    cdef Py_ssize_t keep_start

    if (x0.shape[0] != v0.shape[0]):
        raise ValueError("x0 and v0 must have the same shape")

    cdef ccallback_t callback_fun
    ccallback_prepare(&callback_fun, signatures, fun, CCALLBACK_DEFAULTS)

    cdef ccallback_t callback_gun
    ccallback_prepare(&callback_gun, signatures, gun, CCALLBACK_DEFAULTS)

    cdef bint DoTanIntegration = not(grad_fun is None) or not(grad_gun is None) or not(grad_x0 is None) or not(grad_v0 is None)

    cdef ccallback_t callback_grad_fun
    cdef ccallback_t callback_grad_gun

    if (keep_freq < 0):
        keep_freq = nint

    cdef Py_ssize_t ndof = x0.shape[0]
    cdef Py_ssize_t nint_keep = nint // keep_freq

    if keep_init:
        nint_keep += 1
        keep_start = 1
    else:
        keep_start = 0

    cdef double[::1] x = x0.copy()
    cdef double[::1] v = v0.copy()

    cdef double[::1] res = np.empty((ndof), dtype=np.float64)

    cdef np.ndarray[double, ndim=2, mode="c"] x_keep_np = np.empty((nint_keep, ndof), dtype=np.float64)
    cdef np.ndarray[double, ndim=2, mode="c"] v_keep_np = np.empty((nint_keep, ndof), dtype=np.float64)

    if keep_init:
      x_keep_np[0,:] = x[:]  
      v_keep_np[0,:] = v[:]  

    cdef double[:,::1] x_keep = <double[:(nint_keep-keep_start),:ndof:1]> &x_keep_np[keep_start,0]
    cdef double[:,::1] v_keep = <double[:(nint_keep-keep_start),:ndof:1]> &v_keep_np[keep_start,0]

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

        if keep_init:
            grad_x_keep_np[0,:] = grad_x[:]  
            grad_v_keep_np[0,:] = grad_v[:]  

        grad_x_keep = <double[:(nint_keep-keep_start),:ndof,:grad_ndof:1]> &grad_x_keep_np[keep_start,0,0]
        grad_v_keep = <double[:(nint_keep-keep_start),:ndof,:grad_ndof:1]> &grad_v_keep_np[keep_start,0,0]

        grad_res = np.empty((ndof, grad_ndof), dtype=np.float64)

        ccallback_prepare(&callback_grad_fun, signatures, grad_fun, CCALLBACK_DEFAULTS)
        ccallback_prepare(&callback_grad_gun, signatures, grad_gun, CCALLBACK_DEFAULTS)

    else:

        grad_x = np.zeros((0, 0), dtype=np.float64)
        grad_v = np.zeros((0, 0), dtype=np.float64)

        grad_x_keep_np = np.empty((0, 0, 0), dtype=np.float64)
        grad_v_keep_np = np.empty((0, 0, 0), dtype=np.float64)

        grad_x_keep = grad_x_keep_np
        grad_v_keep = grad_v_keep_np

        grad_res = np.empty((0, 0), dtype=np.float64)

    if mode == 'VX':

        with nogil:
        
            ExplicitSymplecticIVP_ann(
                callback_fun        ,
                callback_gun        ,
                callback_grad_fun   ,
                callback_grad_gun   ,
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
                callback_gun        ,
                callback_fun        ,
                callback_grad_gun   ,
                callback_grad_fun   ,
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
    ccallback_t callback_fun        ,
    ccallback_t callback_gun        ,
    ccallback_t callback_grad_fun   ,
    ccallback_t callback_grad_gun   ,
    (double, double) t_span         ,
    double[::1]     x               ,
    double[::1]     v               ,
    double[:,::1]   grad_x          ,
    double[:,::1]   grad_v          ,
    double[::1]     res             ,
    double[:,::1]   grad_res        ,
    ExplicitSymplecticRKTable rk    ,
    Py_ssize_t nint                 ,
    Py_ssize_t keep_freq            ,
    bint DoEFT                      ,
    bint DoTanIntegration           ,
    double[:,::1]   x_keep          ,
    double[:,::1]   v_keep          ,
    double[:,:,::1] grad_x_keep     ,
    double[:,:,::1] grad_v_keep     ,
) noexcept nogil:

    cdef double tx = t_span[0]
    cdef double tv = t_span[0]
    cdef double dt = (t_span[1] - t_span[0]) / nint

    cdef int ndof = x.shape[0]
    cdef Py_ssize_t nint_keep = nint // keep_freq
    cdef Py_ssize_t nsteps = rk._c_table.shape[0]

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
                LowLevelFun_apply(callback_fun, tv, v, res)
                
                if DoTanIntegration:
                    # grad_res = grad_f(t,v,grad_v)
                    LowLevelFun_grad_apply(callback_grad_fun, tv, v, grad_v, grad_res)

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
                LowLevelFun_apply(callback_gun, tx, x, res)

                if DoTanIntegration:
                    # grad_res = grad_g(t,x,grad_x)
                    LowLevelFun_grad_apply(callback_grad_gun, tx, x, grad_x, grad_res)

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

@cython.final
cdef class ImplicitRKTable:
    
    cdef double[:,::1] _a_table             # A Butcher table.
    cdef double[::1] _b_table               # b Butcher table. Integration weights on [0,1]
    cdef double[::1] _c_table               # c Butcher table. Integration nodes on [0,1]
    cdef double[:,::1] _beta_table          # Beta Butcher table for initial guess in convergence loop. 
    cdef double[:,::1] _gamma_table         # Beta Butcher table of the symmetric adjoint.
    cdef Py_ssize_t _th_cvg_rate                  # Theoretical convergence rate of the method.

    @cython.final
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

    @cython.final
    @property
    def nsteps(self):
        return self._a_table.shape[0]

    @cython.final
    @property
    def a_table(self):
        return np.asarray(self._a_table)

    @cython.final
    @property
    def b_table(self):
        return np.asarray(self._b_table)

    @cython.final
    @property
    def c_table(self):
        return np.asarray(self._c_table)
    
    @cython.final
    @property
    def beta_table(self):
        return np.asarray(self._beta_table)    

    @cython.final
    @property
    def gamma_table(self):
        return np.asarray(self._gamma_table)    

    @cython.final
    @property
    def th_cvg_rate(self):
        return self._th_cvg_rate

    @cython.final
    @property
    def stability_cst(self):
        return np.linalg.norm(self.a_table, np.inf)

    @cython.final
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

    @cython.final
    cdef double _symmetry_default(
        self                    ,
        ImplicitRKTable other   ,
    ) noexcept nogil:

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

    @cython.final
    def symmetry_default(
        self        ,
        other = None,
    ):
        if other is None:
            return self._symmetry_default(self)
        else:
            return self._symmetry_default(other)
    
    @cython.final
    cdef bint _is_symmetric_pair(self, ImplicitRKTable other, double tol) noexcept nogil:
        return (self._symmetry_default(other) < tol)

    @cython.final
    def is_symmetric_pair(self, ImplicitRKTable other, double tol = 1e-12):
        return self._is_symmetric_pair(other, tol)

    @cython.final
    def is_symmetric(self, double tol = 1e-12):
        return self._is_symmetric_pair(self, tol)

    @cython.final
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

    @cython.final
    cdef double _symplectic_default(
        self                    ,
        ImplicitRKTable other   ,
    ) noexcept nogil:

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

    @cython.final
    def symplectic_default(
        self        ,
        other = None,
    ):
        if other is None:
            return self._symplectic_default(self)
        else:
            return self._symplectic_default(other)
    
    @cython.final
    cpdef bint _is_symplectic_pair(self, ImplicitRKTable other, double tol):
        return (self._symplectic_default(other) < tol)

    @cython.final
    def is_symplectic_pair(self, ImplicitRKTable other, double tol = 1e-12):
        return self._is_symplectic_pair(other, tol)

    @cython.final
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
    bint vector_calls = False               ,
    object grad_fun = None                  ,
    object grad_gun = None                  ,
    double[:,::1] grad_x0 = None            ,
    double[:,::1] grad_v0 = None            ,
    Py_ssize_t nint = 1                     ,
    Py_ssize_t keep_freq = -1               ,
    bint keep_init = False                  ,
    bint DoEFT = True                       ,
    double eps = np.finfo(np.float64).eps   ,
    Py_ssize_t maxiter = 50                 ,
):

    cdef Py_ssize_t nsteps = rk_x._a_table.shape[0]
    cdef Py_ssize_t keep_start
    cdef bint correct_shapes
    cdef Py_ssize_t istep
    cdef Py_ssize_t i,j

    if (rk_v._a_table.shape[0] != nsteps):
        raise ValueError("rk_x and rk_v must have the same shape")

    if (x0.shape[0] != v0.shape[0]):
        raise ValueError("x0 and v0 must have the same shape")

    cdef Py_ssize_t ndof = x0.shape[0]

    cdef ccallback_t callback_fun
    ccallback_prepare(&callback_fun, signatures, fun, CCALLBACK_DEFAULTS)

    cdef ccallback_t callback_gun
    ccallback_prepare(&callback_gun, signatures, gun, CCALLBACK_DEFAULTS)

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

    if (keep_freq < 0):
        keep_freq = nint

    cdef Py_ssize_t nint_keep = nint // keep_freq

    if keep_init:
        nint_keep += 1
        keep_start = 1
    else:
        keep_start = 0

    cdef np.ndarray[double, ndim=2, mode="c"] x_keep_np = np.empty((nint_keep, ndof), dtype=np.float64)
    cdef np.ndarray[double, ndim=2, mode="c"] v_keep_np = np.empty((nint_keep, ndof), dtype=np.float64)

    istep = nint_keep-keep_start

    cdef double[:,::1] x_keep = <double[:istep,:ndof:1]> &x_keep_np[keep_start,0]
    cdef double[:,::1] v_keep = <double[:istep,:ndof:1]> &v_keep_np[keep_start,0]

    if keep_init:
        x_keep_np[0,:] = x[:]  
        v_keep_np[0,:] = v[:]  

    cdef bint DoTanIntegration = not(grad_fun is None) or not(grad_gun is None) or not(grad_x0 is None) or not(grad_v0 is None)

    cdef ccallback_t callback_grad_fun
    cdef ccallback_t callback_grad_gun

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

        if keep_init:
            grad_x_keep_np[0,:] = grad_x[:]  
            grad_v_keep_np[0,:] = grad_v[:]  

        grad_x_keep = <double[:(nint_keep-keep_start),:ndof,:grad_ndof:1]> &grad_x_keep_np[keep_start,0,0]
        grad_v_keep = <double[:(nint_keep-keep_start),:ndof,:grad_ndof:1]> &grad_v_keep_np[keep_start,0,0]

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
        ccallback_prepare(&callback_grad_gun, signatures, grad_gun, CCALLBACK_DEFAULTS)

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

    with nogil:
    
        ImplicitSymplecticIVP_ann(
            callback_fun        ,
            callback_gun        ,
            callback_grad_fun   ,
            callback_grad_gun   ,
            vector_calls        ,
            t_span              ,
            x                   ,
            v                   ,
            grad_x              ,
            grad_v              ,
            K_fun               ,
            K_gun               ,
            grad_K_fun          ,
            grad_K_gun          ,
            dX                  ,
            dV                  ,
            grad_dX             ,
            grad_dV             ,
            dX_prev             ,
            dV_prev             ,
            grad_dX_prev        ,
            grad_dV_prev        ,
            all_t_x             ,
            all_t_v             ,
            rk_x._a_table       ,
            rk_x._b_table       ,
            rk_x._c_table       ,
            rk_x._beta_table    ,
            rk_v._a_table       ,
            rk_v._b_table       ,
            rk_v._c_table       ,
            rk_v._beta_table    ,
            nint                ,
            keep_freq           ,
            DoEFT               ,
            DoTanIntegration    ,
            eps                 ,
            maxiter             ,
            x_keep              ,
            v_keep              ,
            grad_x_keep         ,
            grad_v_keep         ,
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
    ccallback_t callback_fun        ,
    ccallback_t callback_gun        ,
    ccallback_t callback_grad_fun   ,
    ccallback_t callback_grad_gun   ,
    bint vector_calls               ,
    (double, double) t_span         ,
    double[::1]     x               ,
    double[::1]     v               ,
    double[:,::1]   grad_x          ,
    double[:,::1]   grad_v          ,
    double[:,::1]   K_fun           ,
    double[:,::1]   K_gun           ,
    double[:,:,::1] grad_K_fun      ,
    double[:,:,::1] grad_K_gun      ,
    double[:,::1]   dX              ,
    double[:,::1]   dV              ,
    double[:,:,::1] grad_dX         ,
    double[:,:,::1] grad_dV         ,
    double[:,::1]   dX_prev         ,
    double[:,::1]   dV_prev         ,
    double[:,:,::1] grad_dX_prev    ,
    double[:,:,::1] grad_dV_prev    ,
    double[::1]     all_t_x         ,
    double[::1]     all_t_v         ,
    double[:,::1]   a_table_x       ,
    double[::1]     b_table_x       ,
    double[::1]     c_table_x       ,
    double[:,::1]   beta_table_x    ,
    double[:,::1]   a_table_v       ,
    double[::1]     b_table_v       ,
    double[::1]     c_table_v       ,
    double[:,::1]   beta_table_v    ,
    Py_ssize_t nint                 ,
    Py_ssize_t keep_freq            ,
    bint DoEFT                      ,
    bint DoTanIntegration           ,
    double eps                      ,
    Py_ssize_t maxiter              ,
    double[:,::1]   x_keep          ,
    double[:,::1]   v_keep          ,
    double[:,:,::1] grad_x_keep     ,
    double[:,:,::1] grad_v_keep     ,
) noexcept nogil:

    cdef int ndof = x.shape[0]
    cdef int grad_ndof
    cdef Py_ssize_t iGS
    cdef Py_ssize_t istep, jdof
    cdef Py_ssize_t iint_keep, ifreq
    cdef Py_ssize_t iint
    cdef Py_ssize_t tot_niter = 0
    cdef Py_ssize_t grad_tot_niter = 0
    cdef Py_ssize_t nint_keep = nint // keep_freq

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
            scipy.linalg.cython_blas.dgemm(transn,transn,&ndof,&nsteps,&nsteps,&one_double,&K_gun[0,0],&ndof,&beta_table_v[0,0],&nsteps,&zero_double,&dV[0,0],&ndof)

            # dX = beta_table_x . K_fun
            scipy.linalg.cython_blas.dgemm(transn,transn,&ndof,&nsteps,&nsteps,&one_double,&K_fun[0,0],&ndof,&beta_table_x[0,0],&nsteps,&zero_double,&dX[0,0],&ndof)

            # dX_prev = dX
            scipy.linalg.cython_blas.dcopy(&dX_size,&dX[0,0],&int_one,&dX_prev[0,0],&int_one)

            iGS = 0
            GoOnGS = True

            while GoOnGS:

                # dV_prev = dV
                scipy.linalg.cython_blas.dcopy(&dX_size,&dV[0,0],&int_one,&dV_prev[0,0],&int_one)

                # K_fun = dt * fun(t,v+dV)
                for istep in range(nsteps):
                    scipy.linalg.cython_blas.daxpy(&ndof,&one_double,&v[0],&int_one,&dV[istep,0],&int_one)

                LowLevelFun_apply_vectorized(vector_calls, callback_fun, all_t_v, dV, K_fun)

                scipy.linalg.cython_blas.dscal(&dX_size,&dt,&K_fun[0,0],&int_one)

                # dX = a_table_x .K_fun
                scipy.linalg.cython_blas.dgemm(transn,transn,&ndof,&nsteps,&nsteps,&one_double,&K_fun[0,0],&ndof,&a_table_x[0,0],&nsteps,&zero_double,&dX[0,0],&ndof)

                # dX_prev = dX_prev - dX
                scipy.linalg.cython_blas.daxpy(&dX_size,&minusone_double,&dX[0,0],&int_one,&dX_prev[0,0],&int_one)
                dX_err = scipy.linalg.cython_blas.dasum(&dX_size,&dX_prev[0,0],&int_one)

                # dX_prev = dX
                scipy.linalg.cython_blas.dcopy(&dX_size,&dX[0,0],&int_one,&dX_prev[0,0],&int_one)

                # K_gun = dt * gun(t,x+dX)
                for istep in range(nsteps):
                    scipy.linalg.cython_blas.daxpy(&ndof,&one_double,&x[0],&int_one,&dX[istep,0],&int_one)

                LowLevelFun_apply_vectorized(vector_calls, callback_gun, all_t_x, dX, K_gun)

                scipy.linalg.cython_blas.dscal(&dX_size,&dt,&K_gun[0,0],&int_one)

                # dV = a_table_v . K_gun
                scipy.linalg.cython_blas.dgemm(transn,transn,&ndof,&nsteps,&nsteps,&one_double,&K_gun[0,0],&ndof,&a_table_v[0,0],&nsteps,&zero_double,&dV[0,0],&ndof)

                # dV_prev = dV_prev - dV
                scipy.linalg.cython_blas.daxpy(&dX_size,&minusone_double,&dV[0,0],&int_one,&dV_prev[0,0],&int_one)
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
                    scipy.linalg.cython_blas.daxpy(&ndof,&one_double,&v[0],&int_one,&dV[istep,0],&int_one)

                # grad_dV = beta_table_v . grad_K_gun
                scipy.linalg.cython_blas.dgemm(transn,transn,&grad_nvar,&nsteps,&nsteps,&one_double,&grad_K_gun[0,0,0],&grad_nvar,&beta_table_v[0,0],&nsteps,&zero_double,&grad_dV[0,0,0],&grad_nvar)

                # grad_dX = beta_table_x . grad_K_fun
                scipy.linalg.cython_blas.dgemm(transn,transn,&grad_nvar,&nsteps,&nsteps,&one_double,&grad_K_fun[0,0,0],&grad_nvar,&beta_table_x[0,0],&nsteps,&zero_double,&grad_dX[0,0,0],&grad_nvar)

                # grad_dX_prev = grad_dX
                scipy.linalg.cython_blas.dcopy(&grad_dX_size,&grad_dX[0,0,0],&int_one,&grad_dX_prev[0,0,0],&int_one)

                while GoOnGS:

                    # grad_dV_prev = grad_dV_prev - grad_dV
                    scipy.linalg.cython_blas.daxpy(&grad_dX_size,&minusone_double,&grad_dV[0,0,0],&int_one,&grad_dV_prev[0,0,0],&int_one)

                    # grad_K_fun = dt * grad_fun(t,grad_v+grad_dV)
                    for istep in range(nsteps):
                        scipy.linalg.cython_blas.daxpy(&grad_nvar,&one_double,&grad_v[0,0],&int_one,&grad_dV[istep,0,0],&int_one)

                    LowLevelFun_apply_grad_vectorized(vector_calls, callback_grad_fun, all_t_v, dV, grad_dV, grad_K_fun)

                    scipy.linalg.cython_blas.dscal(&grad_dX_size,&dt,&grad_K_fun[0,0,0],&int_one)

                    # grad_dX = a_table_x . grad_K_fun
                    scipy.linalg.cython_blas.dgemm(transn,transn,&grad_nvar,&nsteps,&nsteps,&one_double,&grad_K_fun[0,0,0],&grad_nvar,&a_table_x[0,0],&nsteps,&zero_double,&grad_dX[0,0,0],&grad_nvar)

                    # grad_dX_prev = grad_dX_prev - grad_dX
                    scipy.linalg.cython_blas.daxpy(&grad_dX_size,&minusone_double,&grad_dX[0,0,0],&int_one,&grad_dX_prev[0,0,0],&int_one)
                    dX_err = scipy.linalg.cython_blas.dasum(&grad_dX_size,&grad_dX_prev[0,0,0],&int_one)

                    # grad_dX_prev = grad_dX
                    scipy.linalg.cython_blas.dcopy(&grad_dX_size,&grad_dX[0,0,0],&int_one,&grad_dX_prev[0,0,0],&int_one)

                    # grad_K_gun = dt * grad_gun(t,grad_x+grad_dX)
                    for istep in range(nsteps):
                        scipy.linalg.cython_blas.daxpy(&grad_nvar,&one_double,&grad_x[0,0],&int_one,&grad_dX[istep,0,0],&int_one)

                    LowLevelFun_apply_grad_vectorized(vector_calls, callback_grad_gun, all_t_x, dX, grad_dX, grad_K_gun)

                    scipy.linalg.cython_blas.dscal(&grad_dX_size,&dt,&grad_K_gun[0,0,0],&int_one)

                    # grad_dV = a_table_v . grad_K_gun
                    scipy.linalg.cython_blas.dgemm(transn,transn,&grad_nvar,&nsteps,&nsteps,&one_double,&grad_K_gun[0,0,0],&grad_nvar,&a_table_v[0,0],&nsteps,&zero_double,&grad_dV[0,0,0],&grad_nvar)

                    # grad_dV_prev = grad_dV_prev - grad_dV
                    scipy.linalg.cython_blas.daxpy(&grad_dX_size,&minusone_double,&grad_dV[0,0,0],&int_one,&grad_dV_prev[0,0,0],&int_one)
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
                scipy.linalg.cython_blas.dgemv(transn,&ndof,&nsteps,&one_double,&K_fun[0,0],&ndof,&b_table_x[0],&int_one,&zero_double,dxv,&int_one)
                # x = x + dxv
                TwoSum_incr(&x[0],dxv,x_eft_comp,ndof)

                # dxv = b_table_v^T . K_gun
                scipy.linalg.cython_blas.dgemv(transn,&ndof,&nsteps,&one_double,&K_gun[0,0],&ndof,&b_table_v[0],&int_one,&zero_double,dxv,&int_one)
                # v = v + dxv
                TwoSum_incr(&v[0],dxv,v_eft_comp,ndof)

                if DoTanIntegration:

                    # grad_dxv = b_table_x^T . grad_K_fun
                    scipy.linalg.cython_blas.dgemv(transn,&grad_nvar,&nsteps,&one_double,&grad_K_fun[0,0,0],&grad_nvar,&b_table_x[0],&int_one,&zero_double,grad_dxv,&int_one)
                    # grad_x = grad_x + grad_dxv
                    TwoSum_incr(&grad_x[0,0],grad_dxv,grad_x_eft_comp,grad_nvar)

                    # grad_dxv = b_table_v^T . grad_K_gun
                    scipy.linalg.cython_blas.dgemv(transn,&grad_nvar,&nsteps,&one_double,&grad_K_gun[0,0,0],&grad_nvar,&b_table_v[0],&int_one,&zero_double,grad_dxv,&int_one)
                    # grad_v = grad_v + grad_dxv
                    TwoSum_incr(&grad_v[0,0],grad_dxv,grad_v_eft_comp,grad_nvar)

            else:

                # x = x + b_table_x^T . K_fun
                scipy.linalg.cython_blas.dgemv(transn,&ndof,&nsteps,&one_double,&K_fun[0,0],&ndof,&b_table_x[0],&int_one,&one_double,&x[0],&int_one)

                # v = v + b_table_v^T . K_gun
                scipy.linalg.cython_blas.dgemv(transn,&ndof,&nsteps,&one_double,&K_gun[0,0],&ndof,&b_table_v[0],&int_one,&one_double,&v[0],&int_one)

                if DoTanIntegration:

                    # grad_x = grad_x + b_table_x^T . grad_K_fun
                    scipy.linalg.cython_blas.dgemv(transn,&grad_nvar,&nsteps,&one_double,&grad_K_fun[0,0,0],&grad_nvar,&b_table_x[0],&int_one,&one_double,&grad_x[0,0],&int_one)

                    # grad_v = grad_v + b_table_v^T . grad_K_gun
                    scipy.linalg.cython_blas.dgemv(transn,&grad_nvar,&nsteps,&one_double,&grad_K_gun[0,0,0],&grad_nvar,&b_table_v[0],&int_one,&one_double,&grad_v[0,0],&int_one)

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

