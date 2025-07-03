'''
ODE.pyx : Defines ODE-related things I designed I feel ought to be in scipy ... but faster !

'''

__all__ = [
    'ExplicitSymplecticRKTable' ,
    'ImplicitRKTable'           ,
    'ExplicitSymplecticIVP'     ,
    'ImplicitSymplecticIVP'     ,
]

from choreo.segm.cython.eft_lib cimport TwoSum_incr, TwoSumScal_incr
from choreo.segm.cython.quad cimport QuadTable

cimport scipy.linalg.cython_blas
from libc.stdlib cimport malloc, free
from libc.string cimport memset

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

from choreo.segm.multiprec_tables import ComputeImplicitRKTable

default_implicit_rk = ComputeImplicitRKTable()
default_explicit_rk = ExplicitSymplecticRKTable(
    c_table = np.array([0.    ,1.      ])   ,
    d_table = np.array([1./2  ,1./2    ])   ,
    th_cvg_rate = 2                         ,
) # Störmer-Verlet 

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
        
            PyFun_apply(callback, t, x, res)

cdef void PyFun_apply(
    ccallback_t callback    ,
    double t                ,
    double[::1] x           ,
    double[::1] res         ,
):

    cdef int n = res.shape[0]
    cdef double[::1] res_1D = (<object> callback.py_function)(t, x)

    scipy.linalg.cython_blas.dcopy(&n,&res_1D[0],&int_one,&res[0],&int_one)

cdef inline void LowLevelFun_grad_apply(
    ccallback_t callback    ,
    double t                ,
    double[::1] x           ,
    double[:,::1] grad_x    ,
    double[:,::1] res       ,
) noexcept nogil:

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

            PyFun_grad_apply(callback, t, x, grad_x, res)

cdef inline void PyFun_grad_apply(
    ccallback_t callback    ,
    double t                ,
    double[::1] x           ,
    double[:,::1] grad_x    ,
    double[:,::1] res       ,
):

    cdef int n = res.shape[0] * res.shape[1]
    cdef double[:,::1] res_2D = (<object> callback.py_function)(t, x, grad_x)

    scipy.linalg.cython_blas.dcopy(&n,&res_2D[0,0],&int_one,&res[0,0],&int_one)

cdef inline void LowLevelFun_apply_vectorized(
    bint vector_calls       ,
    ccallback_t callback    ,
    double[::1] all_t       ,
    double[:,::1] all_x     ,
    double[:,::1] all_res   ,
) noexcept nogil:

    cdef Py_ssize_t i

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

            PyFun_apply_vectorized(vector_calls, callback, all_t, all_x, all_res)

cdef void PyFun_apply_vectorized(
    bint vector_calls       ,
    ccallback_t callback    ,
    double[::1] all_t       ,
    double[:,::1] all_x     ,
    double[:,::1] all_res   ,
):

    cdef Py_ssize_t i
    cdef int n

    cdef double[::1] res_1D
    cdef double[:,::1] res_2D

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

            PyFun_apply_grad_vectorized(vector_calls, callback, all_t, all_x, all_grad_x, all_res)

cdef void PyFun_apply_grad_vectorized(
    bint vector_calls           ,
    ccallback_t callback        ,
    double[::1] all_t           ,
    double[:,::1] all_x         ,
    double[:,:,::1] all_grad_x  ,
    double[:,:,::1] all_res     ,
):

    cdef Py_ssize_t i
    cdef int n

    cdef double[:,::1] res_2D
    cdef double[:,:,::1] res_3D

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
    r""" Butcher Tables for explicit symplectic Runge-Kutta methods applied to partitionned Hamiltonian problems

    cf :footcite:`sanz1992symplectic`

    :cited:
    .. footbibliography::

    See Also
    --------

    * :mod:`choreo.segm.precomputed_tables`

    """
    
    def __init__(
        self                        ,
        c_table     = None          ,
        d_table     = None          ,
        th_cvg_rate = None          ,
        OptimizeFGunCalls = True    ,
        eps = 0.                    ,
    ):

        cdef Py_ssize_t istep
        cdef Py_ssize_t nsteps = c_table.shape[0]

        self._c_table = c_table.copy()
        self._d_table = d_table.copy()

        assert nsteps == d_table.shape[0]

        if th_cvg_rate is None:
            self._th_cvg_rate = -1
        else:
            self._th_cvg_rate = th_cvg_rate

        self._cant_skip_f_eval = np.empty((nsteps), dtype=np.intc)
        self._cant_skip_x_updt = np.empty((nsteps), dtype=np.intc)
        self._cant_skip_g_eval = np.empty((nsteps), dtype=np.intc)
        self._cant_skip_v_updt = np.empty((nsteps), dtype=np.intc)

        # Update loop is (in this order):
        #     - res <- f(t,v)
        #     - x <- x + cdt * res
        #     - res <- g(t,x)
        #     - v <- v + ddt * res

        for istep in range(nsteps):
            self._cant_skip_x_updt[istep] = not (OptimizeFGunCalls and (cfabs(self._c_table[istep]) <= eps))
            self._cant_skip_v_updt[istep] = not (OptimizeFGunCalls and (cfabs(self._d_table[istep]) <= eps))

        for istep in range(nsteps):
            self._cant_skip_f_eval[istep] = self._cant_skip_x_updt[istep] and self._cant_skip_v_updt[(istep-1+nsteps)%nsteps]
            self._cant_skip_g_eval[istep] = self._cant_skip_v_updt[istep] and self._cant_skip_x_updt[istep]

        cdef Py_ssize_t n_eff_steps = 0

        for istep in range(nsteps):

            if self._cant_skip_f_eval[istep]:
                n_eff_steps += 1
            if self._cant_skip_g_eval[istep]:
                n_eff_steps += 1

        n_eff_steps = n_eff_steps // 2

        self.n_eff_steps = n_eff_steps

        # Needed ? Tests so far indicate that no ...
        # self._separate_res_buf = True
        self._separate_res_buf = False

    def __repr__(self):

        res = f'ExplicitSymplecticRKTable object' 

        if self._th_cvg_rate > 0:
            res += f' of order {self._th_cvg_rate}'
        
        res += f' with {self.n_eff_steps} effective step{"s" if self.n_eff_steps > 1 else ""}\n'

        return res

    @cython.final
    @property
    def nsteps(self):
        """ Number of steps of the method. 

        This is the number of functions evaluations needed to solve the ODE for a single timestep.
        """
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
        """Theoretical convergence rate of the method for smooth initial value problems."""
        if self._th_cvg_rate > 0:
            return self._th_cvg_rate
        else:
            return None

    @cython.final
    cpdef ExplicitSymplecticRKTable symmetric_adjoint(self):
        """Computes the symmetric adjoint of a :class:`ExplicitSymplecticRKTable`.

        .. todo:: Define symmetric adjoint

        Returns
        -------
        :class:`choreo.segm.ODE.ExplicitSymplecticRKTable`
            The adjoint Runge-Kutta method.

        """

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

    @cython.final
    cdef double _symmetry_default(
        self                            ,
        ExplicitSymplecticRKTable other ,
    ) noexcept nogil:

        cdef Py_ssize_t nsteps = self._c_table.shape[0]
        cdef Py_ssize_t i,j
        cdef double maxi = -1
        cdef double val

        for i in range(nsteps):

            val = self._c_table[i] - other._d_table[nsteps-1-i] 
            maxi = max(maxi, cfabs(val))

            val = self._d_table[i] - other._c_table[nsteps-1-i] 
            maxi = max(maxi, cfabs(val))

        return maxi  

    @cython.final
    def symmetry_default(
        self                                    ,
        ExplicitSymplecticRKTable other = None   ,
    ):
        r"""Computes the symmetry default of a single / a pair of :class:`ExplicitSymplecticRKTable`.

        A method is said to be symmetric if its symmetry default is zero, namely if it coincides with its :meth:`symmetric_adjoint`.

        Example
        -------

        See Also
        --------

        * :meth:`is_symmetric`
        * :meth:`is_symmetric_pair`

        Parameters
        ----------
        other : :class:`ExplicitSymplecticRKTable`, optional
            By default :data:`python:None`.

        Returns
        -------
        :obj:`numpy:numpy.float64`
            The maximum symmetry violation.
        """    

        if other is None:
            return self._symmetry_default(self)
        else:
            if self._c_table.shape[0] == other._c_table.shape[0]:
                return self._symmetry_default(other)
            else:
                return np.inf

    @cython.final
    cdef bint _is_symmetric_pair(self, ExplicitSymplecticRKTable other, double tol) noexcept nogil:
        return (self._symmetry_default(other) < tol)

    @cython.final
    def is_symmetric_pair(self, ExplicitSymplecticRKTable other, double tol = 1e-12):
        r"""Returns :data:`python:True` if the pair of Runge-Kutta methods is symmetric.

        The pair of methods ``(self, other)`` is inferred symmetric if its symmetry default falls under the specified tolerance ``tol``.

        Example
        -------

        .. todo:: Ex

        See Also
        --------

        * :meth:`symmetry_default`
        * :meth:`is_symmetric`

        Parameters
        ----------
        other : :class:`ExplicitSymplecticRKTable`
        tol : :obj:`numpy:numpy.float64` , optional
            Tolerance on symmetry default, by default ``1e-12``.        

        Returns
        -------
        :class:`python:bool`
            Whether the method is symmetric given the tolerance ``tol``.
        """ 

        if self._c_table.shape[0] == other._c_table.shape[0]:
            return self._is_symmetric_pair(other, tol)
        else:
            return False
        

    @cython.final
    def is_symmetric(self, double tol = 1e-12):
        r"""Returns :data:`python:True` if the Runge-Kutta method is symmetric.

        The method is inferred symmetric if its symmetry default falls under the specified tolerance ``tol``.

        Example
        -------


        .. todo:: Ex



        See Also
        --------

        * :meth:`symmetry_default`
        * :meth:`is_symmetric_pair`

        Parameters
        ----------
        tol : :obj:`numpy:numpy.float64` , optional
            Tolerance on symmetry default, by default ``1e-12``.

        Returns
        -------
        :class:`python:bool`
            Whether the method is symmetric given the tolerance ``tol``.
        """ 

        return self._is_symmetric_pair(self, tol)   

@cython.cdivision(True)
cpdef ExplicitSymplecticIVP(
    object fun                                          ,
    object gun                                          ,
    (double, double) t_span                             ,
    double[::1] xo = None                               ,
    double[::1] vo = None                               ,
    ExplicitSymplecticRKTable rk = default_explicit_rk  ,
    object grad_fun = None                              ,
    object grad_gun = None                              ,
    double[:,::1] grad_xo = None                        ,
    double[:,::1] grad_vo = None                        ,
    object mode = "VX"                                  ,
    # object mode = "XV"                                  ,
    Py_ssize_t nint = 1                                 ,
    Py_ssize_t keep_freq = -1                           ,
    double[:,::1] reg_xo = None                         ,
    double[:,::1] reg_vo = None                         ,
    Py_ssize_t reg_init_freq = -1                       ,
    bint keep_init = False                              ,
    bint DoEFT = True                                   ,
): 
    """Explicit symplectic integration of a partitionned initial value problem.

    Parameters
    ----------
    fun : :obj:`python:callable` or :class:`scipy:scipy.LowLevelCallable`
        Function defining the IVP.
    gun : :obj:`python:callable` or :class:`scipy:scipy.LowLevelCallable`
        Function defining the IVP.
    t_span : :class:`python:tuple` (:obj:`numpy:numpy.float64`, :obj:`numpy:numpy.float64`)
        Initial and final time of integration.
    xo : :class:`numpy:numpy.ndarray`:class:`(shape = (n), dtype = np.float64)`, optional
        Initial value for x. Overriden by reg_xo if provided. By default, :data:`python:None`.
    vo : :class:`numpy:numpy.ndarray`:class:`(shape = (n), dtype = np.float64)`, optional
        Initial value for v. Overriden by reg_xo if provided. By default, :data:`python:None`.
    rk : :class:`ExplicitSymplecticRKTable`, optional
        Runge-Kutta tables for the integration of the IVP. By default, :data:`choreo.segm.precomputed_tables.StormerVerlet`.
    grad_fun : :obj:`python:callable` or :class:`scipy:scipy.LowLevelCallable`, optional
        Gradient of the function defining the IVP, by default :data:`python:None`.
    grad_gun : :obj:`python:callable` or :class:`scipy:scipy.LowLevelCallable`, optional
        Gradient of the function defining the IVP, by default :data:`python:None`.
    mode : :class:`python:str`, optional
        Whether to start the staggered integration with x or v, by default ``"VX"``.
    nint : :class:`python:int`, optional
        Number of integration steps, by default ``1``.
    keep_freq : :class:`python:int`, optional
        Number of integration steps to be taken before saving output, by default ``-1``.
    reg_xo : :class:`numpy:numpy.ndarray`:class:`(shape = (nreg, n), dtype = np.float64)`
        Array of initial values for x for regular reset.
    reg_vo : :class:`numpy:numpy.ndarray`:class:`(shape = (nreg, n), dtype = np.float64)`
        Array of initial values for v for regular reset.
    reg_init_freq : :class:`python:int`, optional
        Number of timesteps before resetting initial values for x and v. Non-positive values disable the reset, by default ``-1``.
    keep_init : :class:`python:bool`, optional
        Whether to save the initial values, by default :data:`python:False`.
    DoEFT : :class:`python:bool`, optional
        Whether to use an error-free transformation for summation, by default :data:`python:True`.

    Returns
    -------
    :class:`python:tuple` of :class:`numpy:numpy.ndarray`.
        Arrays containing the computed approximation of the solution to the IVP at evaluation points.

    """

    if (xo is None) != (vo is None):
        raise ValueError("Only one of reg_xo and reg_vo was provided.")

    if (reg_xo is None) != (reg_vo is None):
        raise ValueError("Only one of reg_xo and reg_vo was provided.")

    if (xo is None) == (reg_xo is None):
        raise ValueError("Exactly one of xo or reg_xo should be provided")

    if (vo is None) == (reg_vo is None):
        raise ValueError("Exactly one of vo or reg_vo should be provided")

    cdef Py_ssize_t ndof

    cdef double[::1] x
    cdef double[::1] v

    if (xo is None):
        x = reg_xo[0,:].copy()
    else:
        x = xo.copy()

    if (vo is None):
        v = reg_vo[0,:].copy()
    else:
        v = vo.copy()

    ndof = x.shape[0]

    if (v.shape[0] != ndof):
        raise ValueError("xo and vo must have the same shape")
    
    cdef Py_ssize_t nreg_init
    cdef Py_ssize_t nreg_needed

    if reg_xo is None:
        reg_xo = np.empty((0, 0), dtype=np.float64)
        reg_vo = np.empty((0, 0), dtype=np.float64)

    else:
        if (reg_xo.shape[1] != ndof) or (reg_xo.shape[1] != ndof):
            raise ValueError("reg_xo or reg_vo have incorrect shapes: reg_xo.shape[1] should be the number of degrees of freedom.")

        nreg_init = reg_xo.shape[0]
        if (reg_vo.shape[0] != nreg_init): 
            raise ValueError("reg_xo and reg_vo should have the same shape.")

        if reg_init_freq < 1:
            reg_init_freq = nint + 1

        nreg_needed = nint // reg_init_freq

        if (nreg_init < nreg_needed):
            raise ValueError("reg_xo and reg_vo do not store enough values")

    cdef ccallback_t callback_fun
    ccallback_prepare(&callback_fun, signatures, fun, CCALLBACK_DEFAULTS)

    cdef ccallback_t callback_gun
    ccallback_prepare(&callback_gun, signatures, gun, CCALLBACK_DEFAULTS)

    cdef bint DoTanIntegration = not(grad_fun is None) or not(grad_gun is None) or not(grad_xo is None) or not(grad_vo is None)

    cdef ccallback_t callback_grad_fun
    cdef ccallback_t callback_grad_gun

    if (keep_freq < 0):
        keep_freq = nint

    cdef Py_ssize_t nint_keep = nint // keep_freq
    cdef Py_ssize_t keep_start
    if keep_init:
        nint_keep += 1
        keep_start = 1
    else:
        keep_start = 0

    cdef double[::1] res_x = np.empty((ndof), dtype=np.float64)
    cdef double[::1] res_v 

    if rk._separate_res_buf:
        res_v = np.empty((ndof), dtype=np.float64)
    else:
        res_v = res_x

    cdef np.ndarray[double, ndim=2, mode="c"] x_keep_np = np.empty((nint_keep, ndof), dtype=np.float64)
    cdef np.ndarray[double, ndim=2, mode="c"] v_keep_np = np.empty((nint_keep, ndof), dtype=np.float64)

    if keep_init:
      x_keep_np[0,:] = x[:]  
      v_keep_np[0,:] = v[:]  

    cdef double[:,::1] x_keep = <double[:(nint_keep-keep_start),:ndof:1]> &x_keep_np[keep_start,0]
    cdef double[:,::1] v_keep = <double[:(nint_keep-keep_start),:ndof:1]> &v_keep_np[keep_start,0]

    cdef double[:,::1] grad_x
    cdef double[:,::1] grad_v
    cdef double[:,::1] grad_res_x
    cdef double[:,::1] grad_res_v

    cdef np.ndarray[double, ndim=3, mode="c"] grad_x_keep_np
    cdef np.ndarray[double, ndim=3, mode="c"] grad_v_keep_np
    cdef double[:,:,::1] grad_x_keep
    cdef double[:,:,::1] grad_v_keep
    cdef int grad_ndof
    cdef Py_ssize_t i,j

    if DoTanIntegration:

        if (grad_xo is None) and (grad_vo is None):

            grad_ndof = 2*ndof

            grad_x = np.zeros((ndof, grad_ndof), dtype=np.float64)
            grad_v = np.zeros((ndof, grad_ndof), dtype=np.float64)

            for i in range(ndof):

                grad_x[i,i] = 1.
                
                j = ndof+i
                grad_v[i,j] = 1.

        elif not(grad_xo is None) and not(grad_vo is None):

            grad_x = grad_xo.copy()
            grad_v = grad_vo.copy()

            assert grad_x.shape[0] == ndof
            assert grad_v.shape[0] == ndof
            
            grad_ndof = grad_x.shape[1]
            assert grad_v.shape[1] == grad_ndof

        else:
            raise ValueError('Wrong values for grad_xo and/or grad_vo')

        grad_x_keep_np = np.empty((nint_keep, ndof, grad_ndof), dtype=np.float64)
        grad_v_keep_np = np.empty((nint_keep, ndof, grad_ndof), dtype=np.float64)

        if keep_init:
            grad_x_keep_np[0,:] = grad_x[:]  
            grad_v_keep_np[0,:] = grad_v[:]  

        grad_x_keep = <double[:(nint_keep-keep_start),:ndof,:grad_ndof:1]> &grad_x_keep_np[keep_start,0,0]
        grad_v_keep = <double[:(nint_keep-keep_start),:ndof,:grad_ndof:1]> &grad_v_keep_np[keep_start,0,0]

        grad_res_x = np.empty((ndof, grad_ndof), dtype=np.float64)
        if rk._separate_res_buf:
            grad_res_v = np.empty((ndof, grad_ndof), dtype=np.float64)
        else:
            grad_res_v = grad_res_x

        ccallback_prepare(&callback_grad_fun, signatures, grad_fun, CCALLBACK_DEFAULTS)
        ccallback_prepare(&callback_grad_gun, signatures, grad_gun, CCALLBACK_DEFAULTS)

    else:

        grad_x = np.zeros((0, 0), dtype=np.float64)
        grad_v = grad_x

        grad_x_keep_np = np.empty((0, 0, 0), dtype=np.float64)
        grad_v_keep_np = grad_x_keep_np

        grad_x_keep = grad_x_keep_np
        grad_v_keep = grad_v_keep_np

        grad_res_x = grad_x
        grad_res_v = grad_x

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
                res_x               ,
                res_v               ,
                grad_res_x          ,
                grad_res_v          ,
                rk                  ,
                nint                ,
                keep_freq           ,
                reg_xo              ,
                reg_vo              ,
                reg_init_freq       ,
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
                res_v               ,
                res_x               ,
                grad_res_v          ,
                grad_res_x          ,
                rk                  ,
                nint                ,
                keep_freq           ,
                reg_vo              ,
                reg_xo              ,
                reg_init_freq       ,
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
    double[::1]     res_x           ,
    double[::1]     res_v           ,
    double[:,::1]   grad_res_x      ,
    double[:,::1]   grad_res_v      ,
    ExplicitSymplecticRKTable rk    ,
    Py_ssize_t nint                 ,
    Py_ssize_t keep_freq            ,
    double[:,::1]   reg_xo          ,
    double[:,::1]   reg_vo          ,
    Py_ssize_t reg_init_freq        ,
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
    cdef bint DoRegInit = reg_init_freq > 0
    cdef Py_ssize_t nsteps = rk._c_table.shape[0]
    cdef Py_ssize_t iint

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

    cdef Py_ssize_t iint_keep = 0
    cdef Py_ssize_t ireg_init = 0
    cdef Py_ssize_t istep

    if DoEFT:

        x_eft_comp = <double *> malloc(sizeof(double) * ndof)
        memset(x_eft_comp, 0, sizeof(double)*ndof)

        v_eft_comp = <double *> malloc(sizeof(double) * ndof)
        memset(v_eft_comp, 0, sizeof(double)*ndof)

        if DoTanIntegration:

            grad_x_eft_comp = <double *> malloc(sizeof(double) * grad_nvar)
            memset(grad_x_eft_comp, 0, sizeof(double)*grad_nvar)

            grad_v_eft_comp = <double *> malloc(sizeof(double) * grad_nvar)
            memset(grad_v_eft_comp, 0, sizeof(double)*grad_nvar)

    for istep in range(nsteps):
        cdt[istep] = rk._c_table[istep] * dt
        ddt[istep] = rk._d_table[istep] * dt

    # Make sure res_x is properly initialized
    if (not rk._cant_skip_f_eval[0]) and rk._cant_skip_x_updt[0]:

        # res_x = f(t,v)
        LowLevelFun_apply(callback_fun, tv, v, res_x)
        
        if DoTanIntegration:
            # grad_res_x = grad_f(t,v,grad_v)
            LowLevelFun_grad_apply(callback_grad_fun, tv, v, grad_v, grad_res_x)

    # Make sure res_v is properly initialized
    if (not rk._cant_skip_g_eval[0]) and rk._cant_skip_v_updt[0]:

        # res_v = g(t,x)
        LowLevelFun_apply(callback_gun, tx, x, res_v)

        if DoTanIntegration:
            # grad_res_v = grad_g(t,x,grad_x)
            LowLevelFun_grad_apply(callback_grad_gun, tx, x, grad_x, grad_res_v)

    for iint in range(nint):

        if DoRegInit:

            if iint % reg_init_freq == 0:

                scipy.linalg.cython_blas.dcopy(&ndof,&reg_xo[ireg_init,0],&int_one,&x[0],&int_one)
                scipy.linalg.cython_blas.dcopy(&ndof,&reg_vo[ireg_init,0],&int_one,&v[0],&int_one)

                if DoEFT:

                    memset(x_eft_comp, 0, sizeof(double)*ndof)        
                    memset(v_eft_comp, 0, sizeof(double)*ndof)

                ireg_init += 1

        for istep in range(nsteps):

            if rk._cant_skip_f_eval[istep]:
                
                # res_x = f(t,v)
                LowLevelFun_apply(callback_fun, tv, v, res_x)
                
                if DoTanIntegration:
                    # grad_res_x = grad_f(t,v,grad_v)
                    LowLevelFun_grad_apply(callback_grad_fun, tv, v, grad_v, grad_res_x)

            if rk._cant_skip_x_updt[istep]:

                # x = x + cdt * res_x
                if DoEFT:
                    TwoSumScal_incr(&x[0],&res_x[0],cdt[istep],x_eft_comp,ndof)
                    TwoSum_incr(&tx,&cdt[istep],&tx_comp,1)

                    if DoTanIntegration:
                        TwoSumScal_incr(&grad_x[0,0],&grad_res_x[0,0],cdt[istep],grad_x_eft_comp,grad_nvar)

                else:
                    scipy.linalg.cython_blas.daxpy(&ndof,&cdt[istep],&res_x[0],&int_one,&x[0],&int_one)
                    tx += cdt[istep]

                    if DoTanIntegration:
                        scipy.linalg.cython_blas.daxpy(&grad_nvar,&cdt[istep],&grad_res_x[0,0],&int_one,&grad_x[0,0],&int_one)

            if rk._cant_skip_g_eval[istep]:

                # res_v = g(t,x)
                LowLevelFun_apply(callback_gun, tx, x, res_v)

                if DoTanIntegration:
                    # grad_res_v = grad_g(t,x,grad_x)
                    LowLevelFun_grad_apply(callback_grad_gun, tx, x, grad_x, grad_res_v)

            if rk._cant_skip_v_updt[istep]:

                # v = v + ddt * res_v
                if DoEFT:
                    TwoSumScal_incr(&v[0],&res_v[0],ddt[istep],v_eft_comp,ndof)
                    TwoSum_incr(&tv,&ddt[istep],&tv_comp,1)

                    if DoTanIntegration:
                        TwoSumScal_incr(&grad_v[0,0],&grad_res_v[0,0],ddt[istep],grad_v_eft_comp,grad_nvar)

                else:
                    scipy.linalg.cython_blas.daxpy(&ndof,&ddt[istep],&res_v[0],&int_one,&v[0],&int_one)
                    tv += ddt[istep]

                    if DoTanIntegration:
                        scipy.linalg.cython_blas.daxpy(&grad_nvar,&ddt[istep],&grad_res_v[0,0],&int_one,&grad_v[0,0],&int_one)

        if (iint+1) % keep_freq == 0:

            scipy.linalg.cython_blas.dcopy(&ndof,&x[0],&int_one,&x_keep[iint_keep,0],&int_one)
            scipy.linalg.cython_blas.dcopy(&ndof,&v[0],&int_one,&v_keep[iint_keep,0],&int_one)

            if DoTanIntegration:

                scipy.linalg.cython_blas.dcopy(&grad_nvar,&grad_x[0,0],&int_one,&grad_x_keep[iint_keep,0,0],&int_one)
                scipy.linalg.cython_blas.dcopy(&grad_nvar,&grad_v[0,0],&int_one,&grad_v_keep[iint_keep,0,0],&int_one)

            iint_keep += 1

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
    """ Butcher Tables for fully implicit Runge-Kutta methods

    Butcher tables defined in :footcite:`butcher2008numerical` and :footcite:`hairer2005numerical`.
    
    :cited:
    .. footbibliography::

    """

    @cython.final
    def __init__(
        self                        ,
        a_table     = None          ,
        quad_table  = None          ,
        beta_table  = None          ,
        gamma_table = None          ,
        th_cvg_rate = None          ,
        OptimizeFGunCalls = True    ,
        eps = 1e-17                 ,
    ):

        self._a_table = a_table.copy()
        self._quad_table = quad_table
        self._beta_table = beta_table.copy()
        self._gamma_table = gamma_table.copy()

        cdef Py_ssize_t nsteps = self._a_table.shape[0]

        assert nsteps == self._a_table.shape[0]
        assert nsteps == self._a_table.shape[1]
        assert nsteps == self._quad_table._w.shape[0]
        assert nsteps == self._quad_table._x.shape[0]
        assert nsteps == self._beta_table.shape[0]
        assert nsteps == self._beta_table.shape[1]
        assert nsteps == self._gamma_table.shape[0]
        assert nsteps == self._gamma_table.shape[1]

        if th_cvg_rate is None:
            self._th_cvg_rate = -1
        else:
            self._th_cvg_rate = th_cvg_rate

#         self._cant_skip_updt = np.empty((nsteps), dtype=np.intc)
#         self._cant_skip_eval = np.empty((nsteps), dtype=np.intc)
#         for istep in range(nsteps):
#             self._cant_skip_updt[istep] = not (OptimizeFGunCalls and (np.linalg.norm(a_table[istep,:]) <= eps))
#             self._cant_skip_eval[istep] = not (OptimizeFGunCalls and (np.linalg.norm(a_table[:,istep]) <= eps))
# 
#         self.n_eff_steps_updt = 0
#         self.n_eff_steps_eval = 0
# 
#         for istep in range(nsteps):
#             if self._cant_skip_updt[istep]:
#                 self.n_eff_steps_updt += 1
#             if self._cant_skip_eval[istep]:
#                 self.n_eff_steps_eval += 1

    def __repr__(self):

        res = f'ImplicitRKTable object with {self._a_table.shape[0]} steps\n'

        return res

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
        return np.asarray(self._quad_table._w)

    @cython.final
    @property
    def c_table(self):
        return np.asarray(self._quad_table._x)

    @cython.final
    @property
    def quad_table(self):
        return self._quad_table
    
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
        r"""Computes the symmetric adjoint of a :class:`ImplicitRKTable`.

        An integration method for an initial value problem maps a function value at an initial time to a final value. The symmetric adjoint of a method applied to the time-reversed system is the method that maps the final value given by the original method to the initial value. A method that is equal to its own symmetric adjoint is called symmetric.
        
        Example
        -------

        >>> import numpy as np
        >>> import choreo
        >>> random_method = choreo.segm.multiprec_tables.ComputeImplicitRKTable(nodes=np.random.random(10))
        >>> random_method.is_symmetric_pair(random_method.symmetric_adjoint())
        True

        See Also
        --------

        * :meth:`is_symmetric`
        * :meth:`is_symmetric_pair`

        Returns
        -------
        :class:`choreo.segm.ODE.ImplicitRKTable`
            The adjoint Runge-Kutta method.

        """

        cdef Py_ssize_t n = self._a_table.shape[0]
        cdef Py_ssize_t i, j

        cdef double[:,::1] a_table_sym = np.empty((n,n), dtype=np.float64)
        cdef double[:,::1] beta_table_sym = np.empty((n,n), dtype=np.float64)
        cdef double[:,::1] gamma_table_sym = np.empty((n,n), dtype=np.float64)

        cdef QuadTable quad_table_sym = self._quad_table.symmetric_adjoint()

        for i in range(n):
            for j in range(n):
                
                a_table_sym[i,j] = self._quad_table._w[n-1-j] - self._a_table[n-1-i,n-1-j]
                beta_table_sym[i,j]  = self._gamma_table[n-1-i,n-1-j]
                gamma_table_sym[i,j] = self._beta_table[n-1-i,n-1-j]

        return ImplicitRKTable(
            a_table     = a_table_sym       ,
            quad_table  = quad_table_sym    ,
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
            for j in range(nsteps):

                val = self._a_table[i,j] - self._quad_table._w[j] + other._a_table[nsteps-1-i,nsteps-1-j]
                maxi = max(maxi, cfabs(val))
                
                val = self._beta_table[i,j] - other._gamma_table[nsteps-1-i,nsteps-1-j]
                maxi = max(maxi, cfabs(val))

                val = self._gamma_table[i,j] - other._beta_table[nsteps-1-i,nsteps-1-j]
                maxi = max(maxi, cfabs(val))
                
        return maxi

    @cython.final
    def symmetry_default(
        self                            ,
        ImplicitRKTable other = None    ,
    ):
        r"""Computes the symmetry default of a single / a pair of :class:`ImplicitRKTable`.

        A method is symmetric if its symmetry default is zero, namely if it coincides with its :meth:`symmetric_adjoint`. Cf Theorem 2.3 of :footcite:`hairer2005numerical`.
        If the two methods do not have the same number of steps, the symmetry default is infinite by convention.

        :cited:
        .. footbibliography::

        Example
        -------

        >>> import choreo
        >>> Radau_IB = choreo.segm.multiprec_tables.ComputeImplicitRKTable(method="Radau_IB")
        >>> Radau_IIB = choreo.segm.multiprec_tables.ComputeImplicitRKTable(method="Radau_IIB")
        >>> Radau_IB.symmetry_default(Radau_IIB)
        2.0816681711721685e-17
        >>> Radau_IIB = choreo.segm.multiprec_tables.ComputeImplicitRKTable(n=10, method="Radau_IIB")
        >>> Radau_IB.symmetry_default(Radau_IIB)
        inf

        See Also
        --------

        * :meth:`is_symmetric`
        * :meth:`is_symmetric_pair`

        Parameters
        ----------
        other : :class:`ImplicitRKTable`, optional
            By default :data:`python:None`.

        Returns
        -------
        :obj:`numpy:numpy.float64`
            The maximum symmetry violation.
        """   
        if other is None:
            return self._symmetry_default(self)
        else:
            if self._a_table.shape[0] == other._a_table.shape[0]:
                return self._symmetry_default(other)
            else:
                return np.inf
    
    @cython.final
    cdef bint _is_symmetric_pair(self, ImplicitRKTable other, double tol) noexcept nogil:
        return (self._symmetry_default(other) < tol)

    @cython.final
    def is_symmetric_pair(self, ImplicitRKTable other, double tol = 1e-12):
        r"""Returns :data:`python:True` if the pair of Runge-Kutta methods is symmetric.

        The pair of methods ``(self, other)`` is inferred symmetric if its symmetry default falls under the specified tolerance ``tol``.

        Example
        -------

        >>> import choreo
        >>> Radau_IA = choreo.segm.multiprec_tables.ComputeImplicitRKTable(method="Radau_IA")
        >>> Radau_IB = choreo.segm.multiprec_tables.ComputeImplicitRKTable(method="Radau_IB")
        >>> Radau_IB.is_symmetric_pair(Radau_IB)
        False
        >>> Radau_IIB = choreo.segm.multiprec_tables.ComputeImplicitRKTable(method="Radau_IIB")
        >>> Radau_IB.is_symmetric_pair(Radau_IIB)
        True

        See Also
        --------

        * :meth:`symmetry_default`
        * :meth:`is_symmetric`

        Parameters
        ----------
        other : :class:`ImplicitRKTable`
        tol : :obj:`numpy:numpy.float64` , optional
            Tolerance on symmetry default, by default ``1e-12``.        

        Returns
        -------
        :class:`python:bool`
            Whether the method is symmetric given the tolerance ``tol``.
        """ 
        return self._is_symmetric_pair(other, tol)

    @cython.final
    def is_symmetric(self, double tol = 1e-12):
        r"""Returns :data:`python:True` if the Runge-Kutta method is symmetric.

        The method is inferred symmetric if its symmetry default falls under the specified tolerance ``tol``.

        Example
        -------

        >>> import choreo
        >>> Gauss = choreo.segm.multiprec_tables.ComputeImplicitRKTable(method="Gauss")
        >>> Gauss.is_symmetric()
        True
        >>> Lobatto_IIIC = choreo.segm.multiprec_tables.ComputeImplicitRKTable(method="Lobatto_IIIC")
        >>> Lobatto_IIIC.is_symmetric()
        False

        See Also
        --------

        * :meth:`symmetry_default`
        * :meth:`is_symmetric_pair`

        Parameters
        ----------
        tol : :obj:`numpy:numpy.float64` , optional
            Tolerance on symmetry default, by default ``1e-12``.

        Returns
        -------
        :class:`python:bool`
            Whether the method is symmetric given the tolerance ``tol``.
        """ 
        return self._is_symmetric_pair(self, tol)

    @cython.final
    @cython.cdivision(True)
    cpdef ImplicitRKTable symplectic_adjoint(self):
        r"""Computes the symplectic adjoint of a :class:`ImplicitRKTable`.

        The flow defined by a Hamiltonian initial value problem preserves the symplectic form. A Runge-Kutta method is said symplectic if this conservation property holds at the discrete level. 
        In the particular case of separable Hamiltonian initial value problems, a Runge-Kutta method paired with its symplectic adjoint in a partitionned integrator is symplectic. Cf :footcite:`butcher2008numerical` and :footcite:`sun2018symmetricadjointsymplecticadjointmethodsapplications` for more details about symplectic adjoints and associated conservation properties.
                
        :cited:
        .. footbibliography::

        Example
        -------

        >>> import choreo
        >>> import numpy as np
        >>> random_method = choreo.segm.multiprec_tables.ComputeImplicitRKTable(nodes=np.random.random(10))
        >>> random_method.is_symplectic_pair(random_method.symplectic_adjoint())
        True

        See Also
        --------

        * :meth:`is_symplectic`
        * :meth:`is_symplectic_pair`
        
        Returns
        -------
        :class:`choreo.segm.ODE.ImplicitRKTable`
            The adjoint Runge-Kutta method.
        """

        cdef Py_ssize_t nsteps = self._a_table.shape[0]
        cdef Py_ssize_t i, j

        cdef double[:,::1] a_table_sym = np.empty((nsteps,nsteps), dtype=np.float64)

        for i in range(nsteps):
            for j in range(nsteps):
                
                a_table_sym[i,j] = self._quad_table._w[j] * (1. - self._a_table[j,i] / self._quad_table._w[i])

        return ImplicitRKTable(
            a_table     = a_table_sym       ,
            quad_table  = self.quad_table   ,
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

            val = self._quad_table._w[i] - other._quad_table._w[i] 
            maxi = max(maxi, cfabs(val))

            val = self._quad_table._x[i] - other._quad_table._x[i] 
            maxi = max(maxi, cfabs(val))

            for j in range(nsteps):
                val = self._quad_table._w[i] * other._a_table[i,j] + other._quad_table._w[j] * self._a_table[j,i] - self._quad_table._w[i] * other._quad_table._w[j] 
                maxi = max(maxi, cfabs(val))

        return maxi

    @cython.final
    def symplectic_default(
        self                        ,
        ImplicitRKTable other = None,
    ):
        r"""Computes the symplecticity default of a single / a pair of :class:`ImplicitRKTable`.

        A method is symplectic if its symplecticity default is zero, namely if it coincides with its :meth:`symplectic_adjoint`. Cf Theorem 4.3 and 4.6 in chapter VI of :footcite:`hairer2005numerical`, as well as Theorem 2.5 of :footcite:`sun2018symmetricadjointsymplecticadjointmethodsapplications`.
        If the two methods do not have the same number of steps, the symplecticity default is infinite by convention.

        :cited:
        .. footbibliography::

        Example
        -------

        >>> import choreo
        >>> Radau_IA = choreo.segm.multiprec_tables.ComputeImplicitRKTable(method="Radau_IA")
        >>> Radau_IA.symplectic_default()
        0.0625
        >>> Radau_IB = choreo.segm.multiprec_tables.ComputeImplicitRKTable(method="Radau_IB")
        >>> Radau_IB.symplectic_default()
        0.0

        See Also
        --------

        * :meth:`is_symplectic`
        * :meth:`is_symplectic_pair`

        Parameters
        ----------
        other : :class:`ImplicitRKTable`, optional
            By default :data:`python:None`.

        Returns
        -------
        :obj:`numpy:numpy.float64`
            The maximum symplecticity violation.
        """   
        if other is None:
            return self._symplectic_default(self)
        else:
            return self._symplectic_default(other)
    
    @cython.final
    cpdef bint _is_symplectic_pair(self, ImplicitRKTable other, double tol) noexcept nogil:
        return (self._symplectic_default(other) < tol)

    @cython.final
    def is_symplectic_pair(self, ImplicitRKTable other, double tol = 1e-12):
        r"""Returns :data:`python:True` if the pair of Runge-Kutta methods is symplectic.

        The pair of methods ``(self, other)`` is inferred symplectic if its symplecticity default falls under the specified tolerance ``tol``.

        Example
        -------

        >>> import choreo
        >>> Radau_IA = choreo.segm.multiprec_tables.ComputeImplicitRKTable(method="Radau_IA")
        >>> Radau_IIA = choreo.segm.multiprec_tables.ComputeImplicitRKTable(method="Radau_IIA")
        >>> Radau_IA.is_symplectic_pair(Radau_IIA)
        False
        >>> Lobatto_IIIA = choreo.segm.multiprec_tables.ComputeImplicitRKTable(method="Lobatto_IIIA")
        >>> Lobatto_IIIB = choreo.segm.multiprec_tables.ComputeImplicitRKTable(method="Lobatto_IIIB")
        >>> Lobatto_IIIA.is_symplectic_pair(Lobatto_IIIB)
        True

        See Also
        --------

        * :meth:`symplectic_default`
        * :meth:`is_symplectic`

        Parameters
        ----------
        other : :class:`ImplicitRKTable`
        tol : :obj:`numpy:numpy.float64` , optional
            Tolerance on symplecticity default, by default ``1e-12``.        

        Returns
        -------
        :class:`python:bool`
            Whether the method is symplectic given the tolerance ``tol``.
        """ 
        return self._is_symplectic_pair(other, tol)

    @cython.final
    def is_symplectic(self, double tol = 1e-12):
        r"""Returns :data:`python:True` if the Runge-Kutta method is symplectic.

        The method is inferred symplectic if its symplecticity default falls under the specified tolerance ``tol``.

        Example
        -------

        >>> import choreo
        >>> Radau_IA = choreo.segm.multiprec_tables.ComputeImplicitRKTable(method="Radau_IA")
        >>> Radau_IA.is_symplectic()
        False
        >>> Radau_IB = choreo.segm.multiprec_tables.ComputeImplicitRKTable(method="Radau_IB")
        >>> Radau_IB.is_symplectic()
        True

        See Also
        --------

        * :meth:`symplectic_default`
        * :meth:`is_symplectic_pair`

        Parameters
        ----------
        tol : :obj:`numpy:numpy.float64` , optional
            Tolerance on symplecticity default, by default ``1e-12``.

        Returns
        -------
        :class:`python:bool`
            Whether the method is symplectic given the tolerance ``tol``.
        """ 
        return self._is_symplectic_pair(self, tol)

@cython.cdivision(True)
cpdef ImplicitSymplecticIVP(
    object fun                                  ,
    object gun                                  ,
    (double, double) t_span                     ,
    double[::1] xo = None                       ,
    double[::1] vo = None                       ,
    ImplicitRKTable rk_x = default_implicit_rk  ,
    ImplicitRKTable rk_v = default_implicit_rk  ,
    bint vector_calls = False                   ,
    object grad_fun = None                      ,
    object grad_gun = None                      ,
    double[:,::1] grad_xo = None                ,
    double[:,::1] grad_vo = None                ,
    Py_ssize_t nint = 1                         ,
    Py_ssize_t keep_freq = -1                   ,
    double[:,::1] reg_xo = None                 ,
    double[:,::1] reg_vo = None                 ,
    Py_ssize_t reg_init_freq = -1               ,
    bint keep_init = False                      ,
    bint DoEFT = True                           ,
    double eps = np.finfo(np.float64).eps       ,
    Py_ssize_t maxiter = 50                     ,
):
    """Implicit symplectic integration of a partitionned initial value problem.

    Follows closely the implementation tips detailled in chapter VIII of :footcite:`hairer2005numerical`

    :cited:
    .. footbibliography::

    Parameters
    ----------
    fun : :obj:`python:callable` or :class:`scipy:scipy.LowLevelCallable`
        Function defining the IVP.
    gun : :obj:`python:callable` or :class:`scipy:scipy.LowLevelCallable`
        Function defining the IVP.
    t_span : :class:`python:tuple` (:obj:`numpy:numpy.float64`, :obj:`numpy:numpy.float64`)
        Initial and final time of integration.
    xo : :class:`numpy:numpy.ndarray`:class:`(shape = (n), dtype = np.float64)`, optional
        Initial value for x. Overriden by reg_xo if provided. By default, :data:`python:None`.
    vo : :class:`numpy:numpy.ndarray`:class:`(shape = (n), dtype = np.float64)`, optional
        Initial value for v. Overriden by reg_xo if provided. By default, :data:`python:None`.
    rk_x : :class:`ImplicitRKTable`, optional
        Runge-Kutta tables for the integration of the IVP. By default, :func:`choreo.segm.multiprec_tables.ComputeImplicitRKTable`.
    rk_v : :class:`ImplicitRKTable`, optional
        Runge-Kutta tables for the integration of the IVP. By default, :func:`choreo.segm.multiprec_tables.ComputeImplicitRKTable`.
    vector_calls : :class:`python:bool`, optional
        Whether to call functions on multiple inputs at once or not, by default :data:`python:False`.
    grad_fun : :obj:`python:callable` or :class:`scipy:scipy.LowLevelCallable`, optional
        Gradient of the function defining the IVP, by default :data:`python:None`.
    grad_gun : :obj:`python:callable` or :class:`scipy:scipy.LowLevelCallable`, optional
        Gradient of the function defining the IVP, by default :data:`python:None`.
    nint : :class:`python:int`, optional
        Number of integration steps, by default ``1``.
    keep_freq : :class:`python:int`, optional
        Number of integration steps to be taken before saving output, by default ``-1``.
    reg_xo : :class:`numpy:numpy.ndarray`:class:`(shape = (nreg, n), dtype = np.float64)`
        Array of initial values for x for regular reset.
    reg_vo : :class:`numpy:numpy.ndarray`:class:`(shape = (nreg, n), dtype = np.float64)`
        Array of initial values for v for regular reset.
    reg_init_freq : :class:`python:int`, optional
        Number of timesteps before resetting initial values for x and v. Non-positive values disable the reset, by default ``-1``.
    keep_init : :class:`python:bool`, optional
        Whether to save the initial values, by default :data:`python:False`.
    DoEFT : :class:`python:bool`, optional
        Whether to use an error-free transformation for summation, by default :data:`python:True`.
    eps : :class:`numpy:numpy.float64`, optional
        Tolerence on the error of each implicit problem, by default ``np.finfo(np.float64).eps``.
    maxiter : :class:`python:int`, optional
        Maximum number of iterations to solve each implicit problem, by default ``50``.

    Returns
    -------
    :class:`python:tuple` of :class:`numpy:numpy.ndarray`.
        Arrays containing the computed approximation of the solution to the IVP at evaluation points.
    """

    cdef Py_ssize_t nsteps = rk_x._a_table.shape[0]
    cdef Py_ssize_t keep_start
    cdef bint correct_shapes
    cdef Py_ssize_t istep
    cdef Py_ssize_t i,j

    if (rk_v._a_table.shape[0] != nsteps):
        raise ValueError("rk_x and rk_v must have the same shape")

    if (xo is None) != (vo is None):
        raise ValueError("Only one of reg_xo and reg_vo was provided.")

    if (reg_xo is None) != (reg_vo is None):
        raise ValueError("Only one of reg_xo and reg_vo was provided.")

    if (xo is None) == (reg_xo is None):
        raise ValueError("Exactly one of xo or reg_xo should be provided")

    if (vo is None) == (reg_vo is None):
        raise ValueError("Exactly one of vo or reg_vo should be provided")

    cdef Py_ssize_t ndof

    cdef double[::1] x
    cdef double[::1] v

    if (xo is None):
        x = reg_xo[0,:].copy()
    else:
        x = xo.copy()

    if (vo is None):
        v = reg_vo[0,:].copy()
    else:
        v = vo.copy()

    ndof = x.shape[0]

    if (v.shape[0] != ndof):
        raise ValueError("xo and vo must have the same shape")
    
    cdef Py_ssize_t nreg_init
    cdef Py_ssize_t nreg_needed

    if reg_xo is None:
        reg_xo = np.empty((0, 0), dtype=np.float64)
        reg_vo = np.empty((0, 0), dtype=np.float64)

    else:
        if (reg_xo.shape[1] != ndof) or (reg_xo.shape[1] != ndof):
            raise ValueError("reg_xo or reg_vo have incorrect shapes: reg_xo.shape[1] should be the number of degrees of freedom.")

        nreg_init = reg_xo.shape[0]
        if (reg_vo.shape[0] != nreg_init): 
            raise ValueError("reg_xo and reg_vo should have the same shape.")

        if reg_init_freq < 1:
            reg_init_freq = nint + 1

        nreg_needed = nint // reg_init_freq

        if (nreg_init < nreg_needed):
            raise ValueError("reg_xo and reg_vo do not store enough values")

    cdef ccallback_t callback_fun
    ccallback_prepare(&callback_fun, signatures, fun, CCALLBACK_DEFAULTS)

    cdef ccallback_t callback_gun
    ccallback_prepare(&callback_gun, signatures, gun, CCALLBACK_DEFAULTS)

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

    cdef bint DoTanIntegration = not(grad_fun is None) or not(grad_gun is None) or not(grad_xo is None) or not(grad_vo is None)

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

        if (grad_xo is None) and (grad_vo is None):

            grad_ndof = 2*ndof

            grad_x = np.zeros((ndof, grad_ndof), dtype=np.float64)
            grad_v = np.zeros((ndof, grad_ndof), dtype=np.float64)

            for i in range(ndof):

                grad_x[i,i] = 1.
                
                j = ndof+i
                grad_v[i,j] = 1.

        elif not(grad_xo is None) and not(grad_vo is None):

            grad_x = grad_xo.copy()
            grad_v = grad_vo.copy()

            assert grad_x.shape[0] == ndof
            assert grad_v.shape[0] == ndof
            
            grad_ndof = grad_x.shape[1]
            assert grad_v.shape[1] == grad_ndof

        else:
            raise ValueError('Wrong values for grad_xo and/or grad_vo')

        grad_x_keep_np = np.empty((nint_keep, ndof, grad_ndof), dtype=np.float64)
        grad_v_keep_np = np.empty((nint_keep, ndof, grad_ndof), dtype=np.float64)

        if keep_init:
            grad_x_keep_np[0,:,:] = grad_x[:,:]  
            grad_v_keep_np[0,:,:] = grad_v[:,:]  

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
            rk_x                ,
            rk_v                ,
            nint                ,
            keep_freq           ,
            reg_xo              ,
            reg_vo              ,
            reg_init_freq       ,
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
    ImplicitRKTable rk_x            ,
    ImplicitRKTable rk_v            ,
    Py_ssize_t nint                 ,
    Py_ssize_t keep_freq            ,
    double[:,::1]   reg_xo          ,
    double[:,::1]   reg_vo          ,
    Py_ssize_t reg_init_freq        ,
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
    cdef bint DoRegInit = reg_init_freq > 0
    cdef int grad_ndof
    cdef Py_ssize_t iGS
    cdef Py_ssize_t istep, jdof
    cdef Py_ssize_t iint_keep = 0
    cdef Py_ssize_t ireg_init = 0
    cdef Py_ssize_t iint
    cdef Py_ssize_t tot_niter = 0
    cdef Py_ssize_t grad_tot_niter = 0

    cdef bint GoOnGS

    cdef double dXV_err, dX_err, dV_err, diff
    cdef double tbeg
    cdef double dt = (t_span[1] - t_span[0]) / nint
    cdef int nsteps = rk_x._a_table.shape[0]
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
        memset(x_eft_comp, 0, sizeof(double)*ndof)        

        v_eft_comp = <double *> malloc(sizeof(double) * ndof)
        memset(v_eft_comp, 0, sizeof(double)*ndof)        

        if DoTanIntegration:

            grad_dxv = <double *> malloc(sizeof(double) * grad_nvar)

            grad_x_eft_comp = <double *> malloc(sizeof(double) * grad_nvar)
            memset(grad_x_eft_comp, 0, sizeof(double)*grad_nvar)        

            grad_v_eft_comp = <double *> malloc(sizeof(double) * grad_nvar)
            memset(grad_v_eft_comp, 0, sizeof(double)*grad_nvar)        

    cdef double *cdt_x = <double *> malloc(sizeof(double) * nsteps)
    for istep in range(nsteps):
        cdt_x[istep] = rk_x._quad_table._x[istep]*dt

    cdef double *cdt_v = <double *> malloc(sizeof(double) * nsteps)
    for istep in range(nsteps):
        cdt_v[istep] = rk_v._quad_table._x[istep]*dt

    for iint in range(nint):

        tbeg = t_span[0] + iint * dt    
        for istep in range(nsteps):
            all_t_v[istep] = tbeg + cdt_v[istep]

        for istep in range(nsteps):
            all_t_x[istep] = tbeg + cdt_x[istep]

        if DoRegInit:

            if iint % reg_init_freq == 0:

                scipy.linalg.cython_blas.dcopy(&ndof,&reg_xo[ireg_init,0],&int_one,&x[0],&int_one)
                scipy.linalg.cython_blas.dcopy(&ndof,&reg_vo[ireg_init,0],&int_one,&v[0],&int_one)

                if DoEFT:

                    memset(x_eft_comp, 0, sizeof(double)*ndof)        
                    memset(v_eft_comp, 0, sizeof(double)*ndof)

                ireg_init += 1

        # dV = rk_v._beta_table . K_gun
        scipy.linalg.cython_blas.dgemm(transn,transn,&ndof,&nsteps,&nsteps,&one_double,&K_gun[0,0],&ndof,&rk_v._beta_table[0,0],&nsteps,&zero_double,&dV[0,0],&ndof)

        # dX = rk_x._beta_table . K_fun
        scipy.linalg.cython_blas.dgemm(transn,transn,&ndof,&nsteps,&nsteps,&one_double,&K_fun[0,0],&ndof,&rk_x._beta_table[0,0],&nsteps,&zero_double,&dX[0,0],&ndof)

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

            # dX = rk_x._a_table . K_fun
            scipy.linalg.cython_blas.dgemm(transn,transn,&ndof,&nsteps,&nsteps,&one_double,&K_fun[0,0],&ndof,&rk_x._a_table[0,0],&nsteps,&zero_double,&dX[0,0],&ndof)

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

            # dV = rk_v._a_table . K_gun
            scipy.linalg.cython_blas.dgemm(transn,transn,&ndof,&nsteps,&nsteps,&one_double,&K_gun[0,0],&ndof,&rk_v._a_table[0,0],&nsteps,&zero_double,&dV[0,0],&ndof)

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

            # grad_dV = rk_v._beta_table . grad_K_gun
            scipy.linalg.cython_blas.dgemm(transn,transn,&grad_nvar,&nsteps,&nsteps,&one_double,&grad_K_gun[0,0,0],&grad_nvar,&rk_v._beta_table[0,0],&nsteps,&zero_double,&grad_dV[0,0,0],&grad_nvar)

            # grad_dX = rk_x._beta_table . grad_K_fun
            scipy.linalg.cython_blas.dgemm(transn,transn,&grad_nvar,&nsteps,&nsteps,&one_double,&grad_K_fun[0,0,0],&grad_nvar,&rk_x._beta_table[0,0],&nsteps,&zero_double,&grad_dX[0,0,0],&grad_nvar)

            # grad_dX_prev = grad_dX
            scipy.linalg.cython_blas.dcopy(&grad_dX_size,&grad_dX[0,0,0],&int_one,&grad_dX_prev[0,0,0],&int_one)

            while GoOnGS:

                # grad_dV_prev = grad_dV
                scipy.linalg.cython_blas.dcopy(&grad_dX_size,&grad_dV[0,0,0],&int_one,&grad_dV_prev[0,0,0],&int_one)

                # grad_K_fun = dt * grad_fun(t,grad_v+grad_dV)
                for istep in range(nsteps):
                    scipy.linalg.cython_blas.daxpy(&grad_nvar,&one_double,&grad_v[0,0],&int_one,&grad_dV[istep,0,0],&int_one)

                LowLevelFun_apply_grad_vectorized(vector_calls, callback_grad_fun, all_t_v, dV, grad_dV, grad_K_fun)

                scipy.linalg.cython_blas.dscal(&grad_dX_size,&dt,&grad_K_fun[0,0,0],&int_one)

                # grad_dX = rk_x._a_table . grad_K_fun
                scipy.linalg.cython_blas.dgemm(transn,transn,&grad_nvar,&nsteps,&nsteps,&one_double,&grad_K_fun[0,0,0],&grad_nvar,&rk_x._a_table[0,0],&nsteps,&zero_double,&grad_dX[0,0,0],&grad_nvar)

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

                # grad_dV = rk_v._a_table . grad_K_gun
                scipy.linalg.cython_blas.dgemm(transn,transn,&grad_nvar,&nsteps,&nsteps,&one_double,&grad_K_gun[0,0,0],&grad_nvar,&rk_v._a_table[0,0],&nsteps,&zero_double,&grad_dV[0,0,0],&grad_nvar)

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

            # dxv = rk_x._quad_table._w^T . K_fun
            scipy.linalg.cython_blas.dgemv(transn,&ndof,&nsteps,&one_double,&K_fun[0,0],&ndof,&rk_x._quad_table._w[0],&int_one,&zero_double,dxv,&int_one)
            # x = x + dxv
            TwoSum_incr(&x[0],dxv,x_eft_comp,ndof)

            # dxv = rk_v._quad_table._w^T . K_gun
            scipy.linalg.cython_blas.dgemv(transn,&ndof,&nsteps,&one_double,&K_gun[0,0],&ndof,&rk_v._quad_table._w[0],&int_one,&zero_double,dxv,&int_one)
            # v = v + dxv
            TwoSum_incr(&v[0],dxv,v_eft_comp,ndof)

            if DoTanIntegration:

                # grad_dxv = rk_x._quad_table._w^T . grad_K_fun
                scipy.linalg.cython_blas.dgemv(transn,&grad_nvar,&nsteps,&one_double,&grad_K_fun[0,0,0],&grad_nvar,&rk_x._quad_table._w[0],&int_one,&zero_double,grad_dxv,&int_one)
                # grad_x = grad_x + grad_dxv
                TwoSum_incr(&grad_x[0,0],grad_dxv,grad_x_eft_comp,grad_nvar)

                # grad_dxv = rk_v._quad_table._w^T . grad_K_gun
                scipy.linalg.cython_blas.dgemv(transn,&grad_nvar,&nsteps,&one_double,&grad_K_gun[0,0,0],&grad_nvar,&rk_v._quad_table._w[0],&int_one,&zero_double,grad_dxv,&int_one)
                # grad_v = grad_v + grad_dxv
                TwoSum_incr(&grad_v[0,0],grad_dxv,grad_v_eft_comp,grad_nvar)

        else:

            # x = x + rk_x._quad_table._w^T . K_fun
            scipy.linalg.cython_blas.dgemv(transn,&ndof,&nsteps,&one_double,&K_fun[0,0],&ndof,&rk_x._quad_table._w[0],&int_one,&one_double,&x[0],&int_one)

            # v = v + rk_v._quad_table._w^T . K_gun
            scipy.linalg.cython_blas.dgemv(transn,&ndof,&nsteps,&one_double,&K_gun[0,0],&ndof,&rk_v._quad_table._w[0],&int_one,&one_double,&v[0],&int_one)

            if DoTanIntegration:

                # grad_x = grad_x + rk_x._quad_table._w^T . grad_K_fun
                scipy.linalg.cython_blas.dgemv(transn,&grad_nvar,&nsteps,&one_double,&grad_K_fun[0,0,0],&grad_nvar,&rk_x._quad_table._w[0],&int_one,&one_double,&grad_x[0,0],&int_one)

                # grad_v = grad_v + rk_v._quad_table._w^T . grad_K_gun
                scipy.linalg.cython_blas.dgemv(transn,&grad_nvar,&nsteps,&one_double,&grad_K_gun[0,0,0],&grad_nvar,&rk_v._quad_table._w[0],&int_one,&one_double,&grad_v[0,0],&int_one)

        if (iint+1) % keep_freq == 0:

            scipy.linalg.cython_blas.dcopy(&ndof,&x[0],&int_one,&x_keep[iint_keep,0],&int_one)
            scipy.linalg.cython_blas.dcopy(&ndof,&v[0],&int_one,&v_keep[iint_keep,0],&int_one)

            if DoTanIntegration:

                scipy.linalg.cython_blas.dcopy(&grad_nvar,&grad_x[0,0],&int_one,&grad_x_keep[iint_keep,0,0],&int_one)
                scipy.linalg.cython_blas.dcopy(&grad_nvar,&grad_v[0,0],&int_one,&grad_v_keep[iint_keep,0,0],&int_one)

            iint_keep += 1

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

