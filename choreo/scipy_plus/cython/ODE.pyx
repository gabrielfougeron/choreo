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

cdef class ExplicitRKTable:
    
    cdef double[::1] c_table
    cdef double[::1] d_table

    def __init__(self, c_table, d_table):

        self.c_table = c_table
        self.d_table = d_table

        assert c_table.shape[0] == d_table.shape[0]

    @property
    def nsteps(self):
        return self.c_table.shape[0]

@cython.cdivision(True)
def ExplicitSymplecticWithTable_VX_cython(
    object fun,
    object gun,
    (double, double) t_span,
    np.ndarray[double, ndim=1, mode="c"] x0,
    np.ndarray[double, ndim=1, mode="c"] v0,
    long nint,
    long keep_freq,
    np.ndarray[double, ndim=1, mode="c"] c_table,
    np.ndarray[double, ndim=1, mode="c"] d_table,
    long nsteps
):
    r"""
    
    This function computes an approximate solution to the :ref:`partitioned Hamiltonian system<ode_PHS>` using an explicit Runge-Kutta method.
    
    :param fun: function 
    :param gun: function 

    :param t_span: initial and final time for simulation
    :type t_span: (double, double)


    :return: two np.ndarray[double, ndim=1, mode="c"] for the final 


    """

    cdef long iint_keep, ifreq
    cdef long nint_keep = nint // keep_freq
    cdef long ndof = x0.size

    cdef np.ndarray[double, ndim=2, mode="c"] x_keep = np.empty((nint_keep,ndof),dtype=np.float64)
    cdef np.ndarray[double, ndim=2, mode="c"] v_keep = np.empty((nint_keep,ndof),dtype=np.float64)

    cdef double tx = t_span[0]
    cdef double tv = t_span[0]
    cdef double dt = (t_span[1] - t_span[0]) / nint

    cdef np.ndarray[double, ndim=1, mode="c"] cdt = np.empty((nsteps),dtype=np.float64)
    cdef np.ndarray[double, ndim=1, mode="c"] ddt = np.empty((nsteps),dtype=np.float64)

    cdef np.ndarray[double, ndim=1, mode="c"] x = x0.copy()
    cdef np.ndarray[double, ndim=1, mode="c"] v = v0.copy()

    cdef np.ndarray[double, ndim=1, mode="c"] res

    cdef long istep,id
    for istep in range(nsteps):
        cdt[istep] = c_table[istep]*dt
        ddt[istep] = d_table[istep]*dt

    for iint_keep in range(nint_keep):

        for ifreq in range(keep_freq):

            for istep in range(nsteps):

                res = fun(tv,v)  
                for idof in range(ndof):
                    x[idof] += cdt[istep] * res[idof]  

                tx += cdt[istep]

                res = gun(tx,x)   
                for idof in range(ndof):
                    v[idof] += ddt[istep] * res[idof]  

                tv += ddt[istep]

        for idof in range(ndof):
            x_keep[iint_keep,idof] = x[idof]
        for idof in range(ndof):
            v_keep[iint_keep,idof] = v[idof]

    return x_keep, v_keep

@cython.cdivision(True)
def ExplicitSymplecticWithTable_XV_cython(
    object fun,
    object gun,
    (double, double) t_span,
    np.ndarray[double, ndim=1, mode="c"] x0,
    np.ndarray[double, ndim=1, mode="c"] v0,
    long nint,
    long keep_freq,
    np.ndarray[double, ndim=1, mode="c"] c_table,
    np.ndarray[double, ndim=1, mode="c"] d_table,
    long nsteps
):

    cdef long iint_keep, ifreq
    cdef long nint_keep = nint // keep_freq
    cdef long ndof = x0.size

    cdef np.ndarray[double, ndim=2, mode="c"] x_keep = np.empty((nint_keep,ndof),dtype=np.float64)
    cdef np.ndarray[double, ndim=2, mode="c"] v_keep = np.empty((nint_keep,ndof),dtype=np.float64)

    cdef double tx = t_span[0]
    cdef double tv = t_span[0]
    cdef double dt = (t_span[1] - t_span[0]) / nint

    cdef np.ndarray[double, ndim=1, mode="c"] cdt = np.empty((nsteps),dtype=np.float64)
    cdef np.ndarray[double, ndim=1, mode="c"] ddt = np.empty((nsteps),dtype=np.float64)

    cdef np.ndarray[double, ndim=1, mode="c"] x = x0.copy()
    cdef np.ndarray[double, ndim=1, mode="c"] v = v0.copy()

    cdef np.ndarray[double, ndim=1, mode="c"] res

    cdef long istep,idof

    for istep in range(nsteps):
        cdt[istep] = c_table[istep]*dt
        ddt[istep] = d_table[istep]*dt

    for iint_keep in range(nint_keep):

        for ifreq in range(keep_freq):

            for istep in range(nsteps):

                res = gun(tx,x)   
                for idof in range(ndof):
                    v[idof] += cdt[istep] * res[idof]  
                tv += cdt[istep]

                res = fun(tv,v)  
                for idof in range(ndof):
                    x[idof] += ddt[istep] * res[idof]  

                tx += ddt[istep]

        for idof in range(ndof):
            x_keep[iint_keep,idof] = x[idof]
        for idof in range(ndof):
            v_keep[iint_keep,idof] = v[idof]

    return x_keep, v_keep

@cython.cdivision(True)
def SymplecticStormerVerlet_XV_cython(
    object fun,
    object gun,
    (double, double) t_span,
    np.ndarray[double, ndim=1, mode="c"] x0,
    np.ndarray[double, ndim=1, mode="c"] v0,
    long nint,
    long keep_freq,
):

    cdef long iint_keep, ifreq
    cdef long nint_keep = nint // keep_freq
    cdef long ndof = x0.size

    cdef np.ndarray[double, ndim=2, mode="c"] x_keep = np.empty((nint_keep,ndof),dtype=np.float64)
    cdef np.ndarray[double, ndim=2, mode="c"] v_keep = np.empty((nint_keep,ndof),dtype=np.float64)

    cdef double t = t_span[0]
    cdef double dt = (t_span[1] - t_span[0]) / nint
    cdef double dt_half = dt*0.5

    cdef np.ndarray[double, ndim=1, mode="c"] x = x0.copy()
    cdef np.ndarray[double, ndim=1, mode="c"] v = v0.copy()

    cdef np.ndarray[double, ndim=1, mode="c"] res

    cdef long idof

    res = gun(t,x)   
    for idof in range(ndof):
        v[idof] += dt_half * res[idof]  

    for iint_keep in range(nint_keep):

        for ifreq in range(keep_freq-1):

            res = fun(t,v)  
            for idof in range(ndof):
                x[idof] += dt* res[idof]  

            t += dt

            res = gun(t,x)   
            for idof in range(ndof):
                v[idof] += dt * res[idof]  

        res = fun(t,v)  
        for idof in range(ndof):
            x[idof] += dt * res[idof]  

        t += dt

        res = gun(t,x)   
        for idof in range(ndof):
            v[idof] += dt_half * res[idof]  

        for idof in range(ndof):
            x_keep[iint_keep,idof] = x[idof]

        for idof in range(ndof):
            v_keep[iint_keep,idof] = v[idof]

        for idof in range(ndof):
            v[idof] += dt_half * res[idof]  

    return x_keep, v_keep

@cython.cdivision(True)
def SymplecticStormerVerlet_VX_cython(
    object fun,
    object gun,
    (double, double) t_span,
    np.ndarray[double, ndim=1, mode="c"] x0,
    np.ndarray[double, ndim=1, mode="c"] v0,
    long nint,
    long keep_freq,
):

    cdef double t = t_span[0]
    cdef double dt = (t_span[1] - t_span[0]) / nint
    cdef double dt_half = dt*0.5
    cdef long nint_keep = nint // keep_freq
    cdef long ndof = x0.size

    cdef np.ndarray[double, ndim=1, mode="c"] x = x0.copy()
    cdef np.ndarray[double, ndim=1, mode="c"] v = v0.copy()

    cdef np.ndarray[double, ndim=2, mode="c"] x_keep = np.empty((nint_keep,ndof),dtype=np.float64)
    cdef np.ndarray[double, ndim=2, mode="c"] v_keep = np.empty((nint_keep,ndof),dtype=np.float64)

    cdef np.ndarray[double, ndim=1, mode="c"] res

    cdef long idof

    res = fun(t,v)  
    
    for iint_keep in range(nint_keep):

        for idof in range(ndof):
            x[idof] += dt_half * res[idof]  

        t += dt_half

        for ifreq in range(keep_freq-1):

            res = gun(t,x)   
            for idof in range(ndof):
                v[idof] += dt * res[idof]  

            res = fun(t,v)  
            for idof in range(ndof):
                x[idof] += dt* res[idof]  

            t += dt

        res = gun(t,x)   
        for idof in range(ndof):
            v[idof] += dt * res[idof]  

        res = fun(t,v)  
        for idof in range(ndof):
            x[idof] += dt_half* res[idof]  

        for idof in range(ndof):
            x_keep[iint_keep,idof] = x[idof]

        for idof in range(ndof):
            v_keep[iint_keep,idof] = v[idof]

        t += dt_half

    return x,v

def ImplicitSymplecticWithTableGaussSeidel_VX_cython(
    object fun,
    object gun,
    (double, double) t_span,
    np.ndarray[double, ndim=1, mode="c"] x0,
    np.ndarray[double, ndim=1, mode="c"] v0,
    int nint,
    int keep_freq,
    np.ndarray[double, ndim=2, mode="c"] a_table_x,
    np.ndarray[double, ndim=1, mode="c"] b_table_x,
    np.ndarray[double, ndim=1, mode="c"] c_table_x,
    np.ndarray[double, ndim=2, mode="c"] beta_table_x,
    np.ndarray[double, ndim=2, mode="c"] a_table_v,
    np.ndarray[double, ndim=1, mode="c"] b_table_v,
    np.ndarray[double, ndim=1, mode="c"] c_table_v,
    np.ndarray[double, ndim=2, mode="c"] beta_table_v,
    int nsteps,
    double eps,
    int maxiter
):
    r"""
    
    This function computes an approximate solution to the :ref:`partitioned Hamiltonian system<ode_PHS>` using an implicit Runge-Kutta method.
    The implicit equations are solved using a Gauss Seidel approach
    
    :param fun: function 
    :param gun: function 

    :param t_span: initial and final time for simulation
    :type t_span: (double, double)


    :return: two np.ndarray[double, ndim=1, mode="c"] for the final 


    """
    cdef int ndof = x0.size
    cdef int istep, id, iGS, jdof
    cdef int kstep
    cdef int tot_niter = 0
    cdef int iint_keep, ifreq
    cdef int nint_keep = nint // keep_freq

    cdef np.ndarray[double, ndim=2, mode="c"] x_keep = np.empty((nint_keep,ndof),dtype=np.float64)
    cdef np.ndarray[double, ndim=2, mode="c"] v_keep = np.empty((nint_keep,ndof),dtype=np.float64)

    cdef bint GoOnGS

    cdef double minus_one = -1.
    cdef double one = 1.
    cdef double zero = 0.
    cdef char *transn = 'n'
    cdef int int_one = 1

    cdef double dXV_err, dX_err, dV_err, diff
    cdef double tbeg, t
    cdef double dt = (t_span[1] - t_span[0]) / nint

    cdef np.ndarray[double, ndim=1, mode="c"] cdt_x = np.empty((nsteps),dtype=np.float64)
    cdef np.ndarray[double, ndim=1, mode="c"] all_t_x = np.empty((nsteps),dtype=np.float64)
    cdef np.ndarray[double, ndim=1, mode="c"] cdt_v = np.empty((nsteps),dtype=np.float64)
    cdef np.ndarray[double, ndim=1, mode="c"] all_t_v = np.empty((nsteps),dtype=np.float64)
    
    cdef np.ndarray[double, ndim=1, mode="c"] arg = np.empty((ndof),dtype=np.float64)

    cdef np.ndarray[double, ndim=1, mode="c"] x = x0.copy()
    cdef np.ndarray[double, ndim=1, mode="c"] v = v0.copy()

    cdef np.ndarray[double, ndim=1, mode="c"] res

    cdef np.ndarray[double, ndim=2, mode="c"] K_fun = np.zeros((nsteps,ndof),dtype=np.float64)
    cdef np.ndarray[double, ndim=2, mode="c"] K_gun = np.zeros((nsteps,ndof),dtype=np.float64)

    cdef np.ndarray[double, ndim=2, mode="c"] dX = np.empty((nsteps,ndof),dtype=np.float64)
    cdef np.ndarray[double, ndim=2, mode="c"] dV = np.empty((nsteps,ndof),dtype=np.float64) 
    cdef np.ndarray[double, ndim=2, mode="c"] dX_prev = np.empty((nsteps,ndof),dtype=np.float64)
    cdef np.ndarray[double, ndim=2, mode="c"] dV_prev = np.empty((nsteps,ndof),dtype=np.float64) 

    cdef int dX_size = nsteps*ndof

    cdef double eps_mul = eps * dX_size * dt

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

            # dX = beta_table_x . K_fun
            scipy.linalg.cython_blas.dgemm(transn,transn,&ndof,&nsteps,&nsteps,&one,&K_fun[0,0],&ndof,&beta_table_x[0,0],&nsteps,&zero,&dX[0,0],&ndof)

            # dV = beta_table_v . K_gun
            scipy.linalg.cython_blas.dgemm(transn,transn,&ndof,&nsteps,&nsteps,&one,&K_gun[0,0],&ndof,&beta_table_v[0,0],&nsteps,&zero,&dV[0,0],&ndof)

            iGS = 0

            GoOnGS = True

            while GoOnGS:

                # dV => dX
                for istep in range(nsteps):

                    for jdof in range(ndof):
                        arg[jdof] = v[jdof] + dV[istep,jdof]

                    res = fun(all_t_v[istep],arg)  

                    for jdof in range(ndof):
                        K_fun[istep,jdof] = dt * res[jdof]

                # dX_prev = dX
                scipy.linalg.cython_blas.dcopy(&dX_size,&dX[0,0],&int_one,&dX_prev[0,0],&int_one)

                # dX = a_table_x .K_fun
                scipy.linalg.cython_blas.dgemm(transn,transn,&ndof,&nsteps,&nsteps,&one,&K_fun[0,0],&ndof,&a_table_x[0,0],&nsteps,&zero,&dX[0,0],&ndof)

                # dX => dV
                for istep in range(nsteps):

                    for jdof in range(ndof):
                        arg[jdof] = x[jdof] + dX[istep,jdof]

                    res = gun(all_t_x[istep],arg)  

                    for jdof in range(ndof):
                        K_gun[istep,jdof] = dt * res[jdof]

                # dV_prev = dV
                scipy.linalg.cython_blas.dcopy(&dX_size,&dV[0,0],&int_one,&dV_prev[0,0],&int_one)

                # dV = a_table_v . K_gun
                scipy.linalg.cython_blas.dgemm(transn,transn,&ndof,&nsteps,&nsteps,&one,&K_gun[0,0],&ndof,&a_table_v[0,0],&nsteps,&zero,&dV[0,0],&ndof)

                # dX_prev = dX_prev - dX
                scipy.linalg.cython_blas.daxpy(&dX_size,&minus_one,&dX[0,0],&int_one,&dX_prev[0,0],&int_one)
                dX_err = scipy.linalg.cython_blas.dasum(&dX_size,&dX_prev[0,0],&int_one)

                # dV_prev = dV_prev - dV
                scipy.linalg.cython_blas.daxpy(&dX_size,&minus_one,&dV[0,0],&int_one,&dV_prev[0,0],&int_one)
                dV_err = scipy.linalg.cython_blas.dasum(&dX_size,&dV_prev[0,0],&int_one)

                dXV_err = dX_err + dV_err  

                iGS += 1

                GoOnGS = (iGS < maxiter) and (dXV_err > eps_mul)

            # if (iGS >= maxiter):
                # print("Max iter exceeded. Rel error : ",dX_err/eps_mul,dV_err/eps_mul)

            tot_niter += iGS

            # Do EFT here ?

            # x = x + b_table_x^T . K_fun
            scipy.linalg.cython_blas.dgemv(transn,&ndof,&nsteps,&one,&K_fun[0,0],&ndof,&b_table_x[0],&int_one,&one,&x[0],&int_one)

            # v = v + b_table_v^T . K_gun
            scipy.linalg.cython_blas.dgemv(transn,&ndof,&nsteps,&one,&K_gun[0,0],&ndof,&b_table_v[0],&int_one,&one,&v[0],&int_one)
        
        for jdof in range(ndof):
            x_keep[iint_keep,jdof] = x[jdof]
        for jdof in range(ndof):
            v_keep[iint_keep,jdof] = v[jdof]
    
    # print(tot_niter / nint)
    # print(1+nsteps*tot_niter)

    return x_keep, v_keep

def ImplicitSymplecticTanWithTableGaussSeidel_VX_cython(
    object fun,
    object gun,
    object grad_fun,
    object grad_gun,
    (double, double) t_span,
    np.ndarray[double, ndim=1, mode="c"] x0,
    np.ndarray[double, ndim=1, mode="c"] v0,
    np.ndarray[double, ndim=2, mode="c"] grad_x0,
    np.ndarray[double, ndim=2, mode="c"] grad_v0,
    int nint,
    int keep_freq,
    np.ndarray[double, ndim=2, mode="c"] a_table_x,
    np.ndarray[double, ndim=1, mode="c"] b_table_x,
    np.ndarray[double, ndim=1, mode="c"] c_table_x,
    np.ndarray[double, ndim=2, mode="c"] beta_table_x,
    np.ndarray[double, ndim=2, mode="c"] a_table_v,
    np.ndarray[double, ndim=1, mode="c"] b_table_v,
    np.ndarray[double, ndim=1, mode="c"] c_table_v,
    np.ndarray[double, ndim=2, mode="c"] beta_table_v,
    int nsteps,
    double eps,
    int maxiter
):
    r"""
    
    This function computes an approximate solution to the :ref:`partitioned Hamiltonian system<ode_PHS>` using an implicit Runge-Kutta method.
    The implicit equations are solved using a Gauss Seidel approach
    
    :param fun: function 
    :param gun: function 

    :param t_span: initial and final time for simulation
    :type t_span: (double, double)


    :return: two np.ndarray[double, ndim=1, mode="c"] for the final 


    """
    cdef int ndof = x0.size
    # ~ cdef int grad_ndof = grad_x0.shape[1] # Does this not work on numpy arrays ?
    cdef int grad_ndof = grad_x0.size // ndof
    cdef int istep, id, iGS, jdof, kdof
    cdef int kstep
    cdef int tot_niter = 0
    cdef int grad_tot_niter = 0
    cdef int iint_keep, ifreq
    cdef int nint_keep = nint // keep_freq

    cdef np.ndarray[double, ndim=2, mode="c"] x_keep = np.empty((nint_keep,ndof),dtype=np.float64)
    cdef np.ndarray[double, ndim=2, mode="c"] v_keep = np.empty((nint_keep,ndof),dtype=np.float64)

    cdef np.ndarray[double, ndim=3, mode="c"] grad_x_keep = np.empty((nint_keep,ndof,grad_ndof),dtype=np.float64)
    cdef np.ndarray[double, ndim=3, mode="c"] grad_v_keep = np.empty((nint_keep,ndof,grad_ndof),dtype=np.float64)

    cdef bint GoOnGS

    cdef double minus_one = -1.
    cdef double one = 1.
    cdef double zero = 0.
    cdef char *transn = 'n'
    cdef int int_one = 1

    cdef double dX_err, dV_err
    cdef double dXV_err, diff
    cdef double tbeg, t
    cdef double dt = (t_span[1] - t_span[0]) / nint

    cdef np.ndarray[double, ndim=1, mode="c"] cdt_x = np.empty((nsteps),dtype=np.float64)
    cdef np.ndarray[double, ndim=1, mode="c"] all_t_x = np.empty((nsteps),dtype=np.float64)
    cdef np.ndarray[double, ndim=1, mode="c"] cdt_v = np.empty((nsteps),dtype=np.float64)
    cdef np.ndarray[double, ndim=1, mode="c"] all_t_v = np.empty((nsteps),dtype=np.float64)

    cdef np.ndarray[double, ndim=1, mode="c"] arg = np.empty((ndof),dtype=np.float64)
    cdef np.ndarray[double, ndim=2, mode="c"] grad_arg = np.empty((ndof,grad_ndof),dtype=np.float64)

    cdef np.ndarray[double, ndim=1, mode="c"] x = x0.copy()
    cdef np.ndarray[double, ndim=1, mode="c"] v = v0.copy()

    cdef np.ndarray[double, ndim=2, mode="c"] grad_x = grad_x0.copy()
    cdef np.ndarray[double, ndim=2, mode="c"] grad_v = grad_v0.copy()

    cdef np.ndarray[double, ndim=1, mode="c"] res
    cdef np.ndarray[double, ndim=2, mode="c"] K_fun = np.zeros((nsteps,ndof),dtype=np.float64)
    cdef np.ndarray[double, ndim=2, mode="c"] K_gun = np.zeros((nsteps,ndof),dtype=np.float64)

    cdef np.ndarray[double, ndim=2, mode="c"] dX = np.empty((nsteps,ndof),dtype=np.float64)
    cdef np.ndarray[double, ndim=2, mode="c"] dV = np.empty((nsteps,ndof),dtype=np.float64) 
    cdef np.ndarray[double, ndim=2, mode="c"] dX_prev = np.empty((nsteps,ndof),dtype=np.float64)
    cdef np.ndarray[double, ndim=2, mode="c"] dV_prev = np.empty((nsteps,ndof),dtype=np.float64) 

    cdef np.ndarray[double, ndim=2, mode="c"] grad_res
    cdef np.ndarray[double, ndim=3, mode="c"] grad_K_fun = np.zeros((nsteps,ndof,grad_ndof),dtype=np.float64)
    cdef np.ndarray[double, ndim=3, mode="c"] grad_K_gun = np.zeros((nsteps,ndof,grad_ndof),dtype=np.float64)

    cdef np.ndarray[double, ndim=3, mode="c"] grad_dX = np.empty((nsteps,ndof,grad_ndof),dtype=np.float64)
    cdef np.ndarray[double, ndim=3, mode="c"] grad_dV = np.empty((nsteps,ndof,grad_ndof),dtype=np.float64) 
    cdef np.ndarray[double, ndim=3, mode="c"] grad_dX_prev = np.empty((nsteps,ndof,grad_ndof),dtype=np.float64)
    cdef np.ndarray[double, ndim=3, mode="c"] grad_dV_prev = np.empty((nsteps,ndof,grad_ndof),dtype=np.float64) 

    cdef int dX_size = nsteps*ndof
    cdef int grad_nvar = ndof * grad_ndof
    cdef int grad_dX_size = nsteps * grad_nvar

    cdef double eps_mul = eps * dX_size * dt
    cdef double grad_eps_mul = eps * grad_dX_size * dt

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

            # dX = beta_table_x . K_fun
            scipy.linalg.cython_blas.dgemm(transn,transn,&ndof,&nsteps,&nsteps,&one,&K_fun[0,0],&ndof,&beta_table_x[0,0],&nsteps,&zero,&dX[0,0],&ndof)

            # dV = beta_table_v . K_gun
            scipy.linalg.cython_blas.dgemm(transn,transn,&ndof,&nsteps,&nsteps,&one,&K_gun[0,0],&ndof,&beta_table_v[0,0],&nsteps,&zero,&dV[0,0],&ndof)

            iGS = 0
            GoOnGS = True

            while GoOnGS:

                # dV => dX
                for istep in range(nsteps):

                    for jdof in range(ndof):
                        arg[jdof] = v[jdof] + dV[istep,jdof]

                    res = fun(all_t_v[istep],arg)  

                    for jdof in range(ndof):
                        K_fun[istep,jdof] = dt * res[jdof]

                # dX_prev = dX
                scipy.linalg.cython_blas.dcopy(&dX_size,&dX[0,0],&int_one,&dX_prev[0,0],&int_one)

                # dX = a_table_x .K_fun
                scipy.linalg.cython_blas.dgemm(transn,transn,&ndof,&nsteps,&nsteps,&one,&K_fun[0,0],&ndof,&a_table_x[0,0],&nsteps,&zero,&dX[0,0],&ndof)

                # dX => dV
                for istep in range(nsteps):

                    for jdof in range(ndof):
                        arg[jdof] = x[jdof] + dX[istep,jdof]

                    res = gun(all_t_x[istep],arg)  

                    for jdof in range(ndof):
                        K_gun[istep,jdof] = dt * res[jdof]

                # dV_prev = dV
                scipy.linalg.cython_blas.dcopy(&dX_size,&dV[0,0],&int_one,&dV_prev[0,0],&int_one)
                
                # dV = a_table_v . K_gun
                scipy.linalg.cython_blas.dgemm(transn,transn,&ndof,&nsteps,&nsteps,&one,&K_gun[0,0],&ndof,&a_table_v[0,0],&nsteps,&zero,&dV[0,0],&ndof)

                # dX_prev = dX_prev - dX
                scipy.linalg.cython_blas.daxpy(&dX_size,&minus_one,&dX[0,0],&int_one,&dX_prev[0,0],&int_one)
                dX_err = scipy.linalg.cython_blas.dasum(&dX_size,&dX_prev[0,0],&int_one)

                # dV_prev = dV_prev - dV
                scipy.linalg.cython_blas.daxpy(&dX_size,&minus_one,&dV[0,0],&int_one,&dV_prev[0,0],&int_one)
                dV_err = scipy.linalg.cython_blas.dasum(&dX_size,&dV_prev[0,0],&int_one)
                
                dXV_err = dX_err + dV_err  

                iGS += 1

                GoOnGS = (iGS < maxiter) and (dXV_err > eps_mul)

            # if (iGS >= maxiter):
                # print("NonLin Max iter exceeded. Rel error : ",dXV_err,eps_mul,iint)
                # print("NonLin Max iter exceeded. Error : ",dX_err,dV_err,eps_mul,iint)

            tot_niter += iGS

            iGS = 0
            GoOnGS = True

            # grad_dV = beta_table_v . grad_K_gun
            scipy.linalg.cython_blas.dgemm(transn,transn,&grad_nvar,&nsteps,&nsteps,&one,&grad_K_gun[0,0,0],&grad_nvar,&beta_table_v[0,0],&nsteps,&zero,&grad_dV[0,0,0],&grad_nvar)

            # grad_dX = beta_table_x . grad_K_fun
            scipy.linalg.cython_blas.dgemm(transn,transn,&grad_nvar,&nsteps,&nsteps,&one,&grad_K_fun[0,0,0],&grad_nvar,&beta_table_x[0,0],&nsteps,&zero,&grad_dX[0,0,0],&grad_nvar)

            while GoOnGS:

                # grad_dV => grad_dX
                for istep in range(nsteps):

                    for jdof in range(ndof):
                        arg[jdof] = v[jdof] + dV[istep,jdof]

                    for jdof in range(ndof):
                        for kdof in range(grad_ndof):
                            grad_arg[jdof,kdof] = grad_v[jdof,kdof] + grad_dV[istep,jdof,kdof]

                    grad_res = grad_fun(all_t_v[istep],arg,grad_arg)  

                    for jdof in range(ndof):
                        for kdof in range(grad_ndof):
                            grad_K_fun[istep,jdof,kdof] = dt * grad_res[jdof,kdof]

                # grad_dX_prev = grad_dX
                scipy.linalg.cython_blas.dcopy(&grad_dX_size,&grad_dX[0,0,0],&int_one,&grad_dX_prev[0,0,0],&int_one)

                # grad_dX = a_table_x . grad_K_fun
                scipy.linalg.cython_blas.dgemm(transn,transn,&grad_nvar,&nsteps,&nsteps,&one,&grad_K_fun[0,0,0],&grad_nvar,&a_table_x[0,0],&nsteps,&zero,&grad_dX[0,0,0],&grad_nvar)

                # grad_dX => grad_dV
                for istep in range(nsteps):

                    for jdof in range(ndof):
                        arg[jdof] = x[jdof] + dX[istep,jdof]

                    for jdof in range(ndof):
                        for kdof in range(grad_ndof):
                            grad_arg[jdof,kdof] = grad_x[jdof,kdof] + grad_dX[istep,jdof,kdof]

                    grad_res = grad_gun(all_t_x[istep],arg,grad_arg)  

                    for jdof in range(ndof):
                        for kdof in range(grad_ndof):
                            grad_K_gun[istep,jdof,kdof] = dt * grad_res[jdof,kdof]

                # grad_dV_prev = grad_dV
                scipy.linalg.cython_blas.dcopy(&grad_dX_size,&grad_dV[0,0,0],&int_one,&grad_dV_prev[0,0,0],&int_one)

                # grad_dV = a_table_v . grad_K_gun
                scipy.linalg.cython_blas.dgemm(transn,transn,&grad_nvar,&nsteps,&nsteps,&one,&grad_K_gun[0,0,0],&grad_nvar,&a_table_v[0,0],&nsteps,&zero,&grad_dV[0,0,0],&grad_nvar)

                # grad_dX_prev = grad_dX_prev - grad_dX
                scipy.linalg.cython_blas.daxpy(&grad_dX_size,&minus_one,&grad_dX[0,0,0],&int_one,&grad_dX_prev[0,0,0],&int_one)
                dX_err = scipy.linalg.cython_blas.dasum(&grad_dX_size,&grad_dX_prev[0,0,0],&int_one)

                # grad_dV_prev = grad_dV_prev - grad_dV
                scipy.linalg.cython_blas.daxpy(&grad_dX_size,&minus_one,&grad_dV[0,0,0],&int_one,&grad_dV_prev[0,0,0],&int_one)
                dV_err = scipy.linalg.cython_blas.dasum(&grad_dX_size,&grad_dV_prev[0,0,0],&int_one)

                dXV_err = dX_err + dV_err  

                iGS += 1

                GoOnGS = (iGS < maxiter) and (dXV_err > grad_eps_mul)

            # if (iGS >= maxiter):
                # print("Tangent Max iter exceeded. Rel error : ",dXV_err/grad_eps_mul,iint)
                # print("Tangent Max iter exceeded. Error : ",dX_err,dV_err,grad_eps_mul,iint)

            grad_tot_niter += iGS

            # Do EFT here ?

            # x = x + b_table_x^T . K_fun
            scipy.linalg.cython_blas.dgemv(transn,&ndof,&nsteps,&one,&K_fun[0,0],&ndof,&b_table_x[0],&int_one,&one,&x[0],&int_one)

            # v = v + b_table_v^T . K_gun
            scipy.linalg.cython_blas.dgemv(transn,&ndof,&nsteps,&one,&K_gun[0,0],&ndof,&b_table_v[0],&int_one,&one,&v[0],&int_one)

            # grad_x = grad_x + b_table_x^T . grad_K_fun
            scipy.linalg.cython_blas.dgemv(transn,&grad_nvar,&nsteps,&one,&grad_K_fun[0,0,0],&grad_nvar,&b_table_x[0],&int_one,&one,&grad_x[0,0],&int_one)

            # grad_v = grad_v + b_table_v^T . grad_K_gun
            scipy.linalg.cython_blas.dgemv(transn,&grad_nvar,&nsteps,&one,&grad_K_gun[0,0,0],&grad_nvar,&b_table_v[0],&int_one,&one,&grad_v[0,0],&int_one)

        for jdof in range(ndof):
            x_keep[iint_keep,jdof] = x[jdof]
        for jdof in range(ndof):
            v_keep[iint_keep,jdof] = v[jdof]

        for jdof in range(ndof):
            for kdof in range(grad_ndof):
                grad_x_keep[iint_keep,jdof,kdof] = grad_x[jdof,kdof]
        for jdof in range(ndof):
            for kdof in range(grad_ndof):
                grad_v_keep[iint_keep,jdof,kdof] = grad_v[jdof,kdof]
    
    # print('Avg nit fun & gun : ',tot_niter/nint)
    # print('Avg nit grad fun & gun : ',grad_tot_niter/nint)
    # print(1+nsteps*tot_niter)

    return x_keep, v_keep, grad_x_keep, grad_v_keep



def ImplicitSymplecticWithTableGaussSeidel_VX_cython_mulfun(
    object fun,
    object gun,
    (double, double) t_span,
    np.ndarray[double, ndim=1, mode="c"] x0,
    np.ndarray[double, ndim=1, mode="c"] v0,
    int nint,
    int keep_freq,
    np.ndarray[double, ndim=2, mode="c"] a_table_x,
    np.ndarray[double, ndim=1, mode="c"] b_table_x,
    np.ndarray[double, ndim=1, mode="c"] c_table_x,
    np.ndarray[double, ndim=2, mode="c"] beta_table_x,
    np.ndarray[double, ndim=2, mode="c"] a_table_v,
    np.ndarray[double, ndim=1, mode="c"] b_table_v,
    np.ndarray[double, ndim=1, mode="c"] c_table_v,
    np.ndarray[double, ndim=2, mode="c"] beta_table_v,
    int nsteps,
    double eps,
    int maxiter
):
    r"""
    
    This function computes an approximate solution to the :ref:`partitioned Hamiltonian system<ode_PHS>` using an implicit Runge-Kutta method.
    The implicit equations are solved using a Gauss Seidel approach
    
    :param fun: function 
    :param gun: function 

    :param t_span: initial and final time for simulation
    :type t_span: (double, double)


    :return: two np.ndarray[double, ndim=1, mode="c"] for the final 


    """
    cdef int ndof = x0.size
    cdef int istep, id, iGS, jdof
    cdef int kstep
    cdef int tot_niter = 0
    cdef int iint_keep, ifreq
    cdef int nint_keep = nint // keep_freq

    cdef np.ndarray[double, ndim=2, mode="c"] x_keep = np.empty((nint_keep,ndof),dtype=np.float64)
    cdef np.ndarray[double, ndim=2, mode="c"] v_keep = np.empty((nint_keep,ndof),dtype=np.float64)

    cdef bint GoOnGS

    cdef double minus_one = -1.
    cdef double one = 1.
    cdef double zero = 0.
    cdef char *transn = 'n'
    cdef int int_one = 1

    cdef double dXV_err, dX_err, dV_err, diff
    cdef double tbeg, t
    cdef double dt = (t_span[1] - t_span[0]) / nint

    cdef np.ndarray[double, ndim=1, mode="c"] cdt_x = np.empty((nsteps),dtype=np.float64)
    cdef np.ndarray[double, ndim=1, mode="c"] all_t_x = np.empty((nsteps),dtype=np.float64)
    cdef np.ndarray[double, ndim=1, mode="c"] cdt_v = np.empty((nsteps),dtype=np.float64)
    cdef np.ndarray[double, ndim=1, mode="c"] all_t_v = np.empty((nsteps),dtype=np.float64)
    
    cdef np.ndarray[double, ndim=2, mode="c"] all_args = np.empty((nsteps,ndof),dtype=np.float64)

    cdef np.ndarray[double, ndim=1, mode="c"] x = x0.copy()
    cdef np.ndarray[double, ndim=1, mode="c"] v = v0.copy()

    cdef np.ndarray[double, ndim=2, mode="c"] K_fun = np.zeros((nsteps,ndof),dtype=np.float64)
    cdef np.ndarray[double, ndim=2, mode="c"] K_gun = np.zeros((nsteps,ndof),dtype=np.float64)

    cdef np.ndarray[double, ndim=2, mode="c"] dX = np.empty((nsteps,ndof),dtype=np.float64)
    cdef np.ndarray[double, ndim=2, mode="c"] dV = np.empty((nsteps,ndof),dtype=np.float64) 
    cdef np.ndarray[double, ndim=2, mode="c"] dX_prev = np.empty((nsteps,ndof),dtype=np.float64)
    cdef np.ndarray[double, ndim=2, mode="c"] dV_prev = np.empty((nsteps,ndof),dtype=np.float64) 

    cdef int dX_size = nsteps*ndof

    cdef double eps_mul = eps * dX_size * dt

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

            # dX = beta_table_x . K_fun
            scipy.linalg.cython_blas.dgemm(transn,transn,&ndof,&nsteps,&nsteps,&one,&K_fun[0,0],&ndof,&beta_table_x[0,0],&nsteps,&zero,&dX[0,0],&ndof)

            # dV = beta_table_v . K_gun
            scipy.linalg.cython_blas.dgemm(transn,transn,&ndof,&nsteps,&nsteps,&one,&K_gun[0,0],&ndof,&beta_table_v[0,0],&nsteps,&zero,&dV[0,0],&ndof)

            iGS = 0

            GoOnGS = True

            while GoOnGS:

                # all_args = v + dV
                for istep in range(nsteps):
                    for jdof in range(ndof):
                        all_args[istep,jdof] = v[jdof] + dV[istep,jdof]

                # K_fun = dt * fun(t,v)
                K_fun = fun(all_t_v,all_args)  

                scipy.linalg.cython_blas.dscal(&dX_size,&dt,&K_fun[0,0],&int_one)

                # dX_prev = dX
                scipy.linalg.cython_blas.dcopy(&dX_size,&dX[0,0],&int_one,&dX_prev[0,0],&int_one)

                # dX = a_table_x .K_fun
                scipy.linalg.cython_blas.dgemm(transn,transn,&ndof,&nsteps,&nsteps,&one,&K_fun[0,0],&ndof,&a_table_x[0,0],&nsteps,&zero,&dX[0,0],&ndof)

                # all_args = x + dX
                for istep in range(nsteps):
                    for jdof in range(ndof):
                        all_args[istep,jdof] = x[jdof] + dX[istep,jdof]

                # K_gun = dt * gun(t,x)
                K_gun = gun(all_t_x,all_args)  

                scipy.linalg.cython_blas.dscal(&dX_size,&dt,&K_gun[0,0],&int_one)

                # dV_prev = dV
                scipy.linalg.cython_blas.dcopy(&dX_size,&dV[0,0],&int_one,&dV_prev[0,0],&int_one)

                # dV = a_table_v . K_gun
                scipy.linalg.cython_blas.dgemm(transn,transn,&ndof,&nsteps,&nsteps,&one,&K_gun[0,0],&ndof,&a_table_v[0,0],&nsteps,&zero,&dV[0,0],&ndof)

                # dX_prev = dX_prev - dX
                scipy.linalg.cython_blas.daxpy(&dX_size,&minus_one,&dX[0,0],&int_one,&dX_prev[0,0],&int_one)
                dX_err = scipy.linalg.cython_blas.dasum(&dX_size,&dX_prev[0,0],&int_one)

                # dV_prev = dV_prev - dV
                scipy.linalg.cython_blas.daxpy(&dX_size,&minus_one,&dV[0,0],&int_one,&dV_prev[0,0],&int_one)
                dV_err = scipy.linalg.cython_blas.dasum(&dX_size,&dV_prev[0,0],&int_one)

                dXV_err = dX_err + dV_err  

                iGS += 1

                GoOnGS = (iGS < maxiter) and (dXV_err > eps_mul)

            # exit()
            # if (iGS >= maxiter):
                # print("Max iter exceeded. Rel error : ",dX_err/eps_mul,dV_err/eps_mul)

            tot_niter += iGS

            # Do EFT here ?

            # x = x + b_table_x^T . K_fun
            scipy.linalg.cython_blas.dgemv(transn,&ndof,&nsteps,&one,&K_fun[0,0],&ndof,&b_table_x[0],&int_one,&one,&x[0],&int_one)

            # v = v + b_table_v^T . K_gun
            scipy.linalg.cython_blas.dgemv(transn,&ndof,&nsteps,&one,&K_gun[0,0],&ndof,&b_table_v[0],&int_one,&one,&v[0],&int_one)
        
        for jdof in range(ndof):
            x_keep[iint_keep,jdof] = x[jdof]
        for jdof in range(ndof):
            v_keep[iint_keep,jdof] = v[jdof]
    
    # print(tot_niter / nint)
    # print(1+nsteps*tot_niter)

    return x_keep, v_keep


def ImplicitSymplecticTanWithTableGaussSeidel_VX_cython_mulfun(
    object fun,
    object gun,
    object grad_fun,
    object grad_gun,
    (double, double) t_span,
    np.ndarray[double, ndim=1, mode="c"] x0,
    np.ndarray[double, ndim=1, mode="c"] v0,
    np.ndarray[double, ndim=2, mode="c"] grad_x0,
    np.ndarray[double, ndim=2, mode="c"] grad_v0,
    int nint,
    int keep_freq,
    np.ndarray[double, ndim=2, mode="c"] a_table_x,
    np.ndarray[double, ndim=1, mode="c"] b_table_x,
    np.ndarray[double, ndim=1, mode="c"] c_table_x,
    np.ndarray[double, ndim=2, mode="c"] beta_table_x,
    np.ndarray[double, ndim=2, mode="c"] a_table_v,
    np.ndarray[double, ndim=1, mode="c"] b_table_v,
    np.ndarray[double, ndim=1, mode="c"] c_table_v,
    np.ndarray[double, ndim=2, mode="c"] beta_table_v,
    int nsteps,
    double eps,
    int maxiter
):
    r"""
    
    This function computes an approximate solution to the :ref:`partitioned Hamiltonian system<ode_PHS>` using an implicit Runge-Kutta method.
    The implicit equations are solved using a Gauss Seidel approach
    
    :param fun: function 
    :param gun: function 

    :param t_span: initial and final time for simulation
    :type t_span: (double, double)


    :return: two np.ndarray[double, ndim=1, mode="c"] for the final 


    """
    cdef int ndof = x0.size
    # ~ cdef int grad_ndof = grad_x0.shape[1] # Does this not work on numpy arrays ?
    cdef int grad_ndof = grad_x0.size // ndof
    cdef int istep, id, iGS, jdof, kdof
    cdef int kstep
    cdef int tot_niter = 0
    cdef int grad_tot_niter = 0
    cdef int iint_keep, ifreq
    cdef int nint_keep = nint // keep_freq

    cdef np.ndarray[double, ndim=2, mode="c"] x_keep = np.empty((nint_keep,ndof),dtype=np.float64)
    cdef np.ndarray[double, ndim=2, mode="c"] v_keep = np.empty((nint_keep,ndof),dtype=np.float64)

    cdef np.ndarray[double, ndim=3, mode="c"] grad_x_keep = np.empty((nint_keep,ndof,grad_ndof),dtype=np.float64)
    cdef np.ndarray[double, ndim=3, mode="c"] grad_v_keep = np.empty((nint_keep,ndof,grad_ndof),dtype=np.float64)

    cdef bint GoOnGS

    cdef double minus_one = -1.
    cdef double one = 1.
    cdef double zero = 0.
    cdef char *transn = 'n'
    cdef int int_one = 1

    cdef double dX_err, dV_err
    cdef double dXV_err, diff
    cdef double tbeg, t
    cdef double dt = (t_span[1] - t_span[0]) / nint

    cdef np.ndarray[double, ndim=1, mode="c"] cdt_x = np.empty((nsteps),dtype=np.float64)
    cdef np.ndarray[double, ndim=1, mode="c"] all_t_x = np.empty((nsteps),dtype=np.float64)
    cdef np.ndarray[double, ndim=1, mode="c"] cdt_v = np.empty((nsteps),dtype=np.float64)
    cdef np.ndarray[double, ndim=1, mode="c"] all_t_v = np.empty((nsteps),dtype=np.float64)

    cdef np.ndarray[double, ndim=2, mode="c"] all_args = np.empty((nsteps,ndof),dtype=np.float64)
    cdef np.ndarray[double, ndim=3, mode="c"] all_grad_args = np.empty((nsteps,ndof,grad_ndof),dtype=np.float64)

    cdef np.ndarray[double, ndim=1, mode="c"] x = x0.copy()
    cdef np.ndarray[double, ndim=1, mode="c"] v = v0.copy()

    cdef np.ndarray[double, ndim=2, mode="c"] grad_x = grad_x0.copy()
    cdef np.ndarray[double, ndim=2, mode="c"] grad_v = grad_v0.copy()

    cdef np.ndarray[double, ndim=1, mode="c"] res
    cdef np.ndarray[double, ndim=2, mode="c"] K_fun = np.zeros((nsteps,ndof),dtype=np.float64)
    cdef np.ndarray[double, ndim=2, mode="c"] K_gun = np.zeros((nsteps,ndof),dtype=np.float64)

    cdef np.ndarray[double, ndim=2, mode="c"] dX = np.empty((nsteps,ndof),dtype=np.float64)
    cdef np.ndarray[double, ndim=2, mode="c"] dV = np.empty((nsteps,ndof),dtype=np.float64) 
    cdef np.ndarray[double, ndim=2, mode="c"] dX_prev = np.empty((nsteps,ndof),dtype=np.float64)
    cdef np.ndarray[double, ndim=2, mode="c"] dV_prev = np.empty((nsteps,ndof),dtype=np.float64) 

    cdef np.ndarray[double, ndim=2, mode="c"] grad_res
    cdef np.ndarray[double, ndim=3, mode="c"] grad_K_fun = np.zeros((nsteps,ndof,grad_ndof),dtype=np.float64)
    cdef np.ndarray[double, ndim=3, mode="c"] grad_K_gun = np.zeros((nsteps,ndof,grad_ndof),dtype=np.float64)

    cdef np.ndarray[double, ndim=3, mode="c"] grad_dX = np.empty((nsteps,ndof,grad_ndof),dtype=np.float64)
    cdef np.ndarray[double, ndim=3, mode="c"] grad_dV = np.empty((nsteps,ndof,grad_ndof),dtype=np.float64) 
    cdef np.ndarray[double, ndim=3, mode="c"] grad_dX_prev = np.empty((nsteps,ndof,grad_ndof),dtype=np.float64)
    cdef np.ndarray[double, ndim=3, mode="c"] grad_dV_prev = np.empty((nsteps,ndof,grad_ndof),dtype=np.float64) 

    cdef int dX_size = nsteps*ndof
    cdef int grad_nvar = ndof * grad_ndof
    cdef int grad_dX_size = nsteps * grad_nvar

    cdef double eps_mul = eps * dX_size * dt
    cdef double grad_eps_mul = eps * grad_dX_size * dt

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

            # dX = beta_table_x . K_fun
            scipy.linalg.cython_blas.dgemm(transn,transn,&ndof,&nsteps,&nsteps,&one,&K_fun[0,0],&ndof,&beta_table_x[0,0],&nsteps,&zero,&dX[0,0],&ndof)

            # dV = beta_table_v . K_gun
            scipy.linalg.cython_blas.dgemm(transn,transn,&ndof,&nsteps,&nsteps,&one,&K_gun[0,0],&ndof,&beta_table_v[0,0],&nsteps,&zero,&dV[0,0],&ndof)

            iGS = 0
            GoOnGS = True

            while GoOnGS:

                # all_args = v + dV
                for istep in range(nsteps):
                    for jdof in range(ndof):
                        all_args[istep,jdof] = v[jdof] + dV[istep,jdof]

                # K_fun = dt * fun(t,v)
                K_fun = fun(all_t_v,all_args)  

                scipy.linalg.cython_blas.dscal(&dX_size,&dt,&K_fun[0,0],&int_one)

                # dX_prev = dX
                scipy.linalg.cython_blas.dcopy(&dX_size,&dX[0,0],&int_one,&dX_prev[0,0],&int_one)

                # dX = a_table_x .K_fun
                scipy.linalg.cython_blas.dgemm(transn,transn,&ndof,&nsteps,&nsteps,&one,&K_fun[0,0],&ndof,&a_table_x[0,0],&nsteps,&zero,&dX[0,0],&ndof)

                # all_args = x + dX
                for istep in range(nsteps):
                    for jdof in range(ndof):
                        all_args[istep,jdof] = x[jdof] + dX[istep,jdof]

                # K_gun = dt * gun(t,x)
                K_gun = gun(all_t_x,all_args)  

                scipy.linalg.cython_blas.dscal(&dX_size,&dt,&K_gun[0,0],&int_one)

                # dV_prev = dV
                scipy.linalg.cython_blas.dcopy(&dX_size,&dV[0,0],&int_one,&dV_prev[0,0],&int_one)
                
                # dV = a_table_v . K_gun
                scipy.linalg.cython_blas.dgemm(transn,transn,&ndof,&nsteps,&nsteps,&one,&K_gun[0,0],&ndof,&a_table_v[0,0],&nsteps,&zero,&dV[0,0],&ndof)

                # dX_prev = dX_prev - dX
                scipy.linalg.cython_blas.daxpy(&dX_size,&minus_one,&dX[0,0],&int_one,&dX_prev[0,0],&int_one)
                dX_err = scipy.linalg.cython_blas.dasum(&dX_size,&dX_prev[0,0],&int_one)

                # dV_prev = dV_prev - dV
                scipy.linalg.cython_blas.daxpy(&dX_size,&minus_one,&dV[0,0],&int_one,&dV_prev[0,0],&int_one)
                dV_err = scipy.linalg.cython_blas.dasum(&dX_size,&dV_prev[0,0],&int_one)
                
                dXV_err = dX_err + dV_err  

                iGS += 1

                GoOnGS = (iGS < maxiter) and (dXV_err > eps_mul)

            # if (iGS >= maxiter):
                # print("NonLin Max iter exceeded. Rel error : ",dXV_err,eps_mul,iint)
                # print("NonLin Max iter exceeded. Error : ",dX_err,dV_err,eps_mul,iint)

            tot_niter += iGS

            iGS = 0
            GoOnGS = True

            # grad_dV = beta_table_v . grad_K_gun
            scipy.linalg.cython_blas.dgemm(transn,transn,&grad_nvar,&nsteps,&nsteps,&one,&grad_K_gun[0,0,0],&grad_nvar,&beta_table_v[0,0],&nsteps,&zero,&grad_dV[0,0,0],&grad_nvar)

            # grad_dX = beta_table_x . grad_K_fun
            scipy.linalg.cython_blas.dgemm(transn,transn,&grad_nvar,&nsteps,&nsteps,&one,&grad_K_fun[0,0,0],&grad_nvar,&beta_table_x[0,0],&nsteps,&zero,&grad_dX[0,0,0],&grad_nvar)

            while GoOnGS:

                # all_args = v + dV
                for istep in range(nsteps):
                    for jdof in range(ndof):
                        all_args[istep,jdof] = v[jdof] + dV[istep,jdof]

                # all_grad_args = grad_v + grad_dV
                for istep in range(nsteps):
                    for jdof in range(ndof):
                        for kdof in range(grad_ndof):
                            all_grad_args[istep,jdof,kdof] = grad_v[jdof,kdof] + grad_dV[istep,jdof,kdof]

                # grad_K_fun = dt * fun(t,v,grad_v)
                grad_K_fun = grad_fun(all_t_v,all_args,all_grad_args)  

                scipy.linalg.cython_blas.dscal(&grad_dX_size,&dt,&grad_K_fun[0,0,0],&int_one)

                # grad_dX_prev = grad_dX
                scipy.linalg.cython_blas.dcopy(&grad_dX_size,&grad_dX[0,0,0],&int_one,&grad_dX_prev[0,0,0],&int_one)

                # grad_dX = a_table_x . grad_K_fun
                scipy.linalg.cython_blas.dgemm(transn,transn,&grad_nvar,&nsteps,&nsteps,&one,&grad_K_fun[0,0,0],&grad_nvar,&a_table_x[0,0],&nsteps,&zero,&grad_dX[0,0,0],&grad_nvar)

                # all_args = x + dX
                for istep in range(nsteps):
                    for jdof in range(ndof):
                        all_args[istep,jdof] = x[jdof] + dX[istep,jdof]

                # all_grad_args = grad_x + grad_dX
                for istep in range(nsteps):
                    for jdof in range(ndof):
                        for kdof in range(grad_ndof):
                            all_grad_args[istep,jdof,kdof] = grad_x[jdof,kdof] + grad_dX[istep,jdof,kdof]

                # grad_K_gun = dt * gun(t,x,grad_x)
                grad_K_gun = grad_gun(all_t_x,all_args,all_grad_args)  

                scipy.linalg.cython_blas.dscal(&grad_dX_size,&dt,&grad_K_gun[0,0,0],&int_one)

                # grad_dV_prev = grad_dV
                scipy.linalg.cython_blas.dcopy(&grad_dX_size,&grad_dV[0,0,0],&int_one,&grad_dV_prev[0,0,0],&int_one)

                # grad_dV = a_table_v . grad_K_gun
                scipy.linalg.cython_blas.dgemm(transn,transn,&grad_nvar,&nsteps,&nsteps,&one,&grad_K_gun[0,0,0],&grad_nvar,&a_table_v[0,0],&nsteps,&zero,&grad_dV[0,0,0],&grad_nvar)

                # grad_dX_prev = grad_dX_prev - grad_dX
                scipy.linalg.cython_blas.daxpy(&grad_dX_size,&minus_one,&grad_dX[0,0,0],&int_one,&grad_dX_prev[0,0,0],&int_one)
                dX_err = scipy.linalg.cython_blas.dasum(&grad_dX_size,&grad_dX_prev[0,0,0],&int_one)

                # grad_dV_prev = grad_dV_prev - grad_dV
                scipy.linalg.cython_blas.daxpy(&grad_dX_size,&minus_one,&grad_dV[0,0,0],&int_one,&grad_dV_prev[0,0,0],&int_one)
                dV_err = scipy.linalg.cython_blas.dasum(&grad_dX_size,&grad_dV_prev[0,0,0],&int_one)

                dXV_err = dX_err + dV_err  

                iGS += 1

                GoOnGS = (iGS < maxiter) and (dXV_err > grad_eps_mul)

            # if (iGS >= maxiter):
                # print("Tangent Max iter exceeded. Rel error : ",dXV_err/grad_eps_mul,iint)
                # print("Tangent Max iter exceeded. Error : ",dX_err,dV_err,grad_eps_mul,iint)

            grad_tot_niter += iGS

            # Do EFT here ?

            # x = x + b_table_x^T . K_fun
            scipy.linalg.cython_blas.dgemv(transn,&ndof,&nsteps,&one,&K_fun[0,0],&ndof,&b_table_x[0],&int_one,&one,&x[0],&int_one)

            # v = v + b_table_v^T . K_gun
            scipy.linalg.cython_blas.dgemv(transn,&ndof,&nsteps,&one,&K_gun[0,0],&ndof,&b_table_v[0],&int_one,&one,&v[0],&int_one)

            # grad_x = grad_x + b_table_x^T . grad_K_fun
            scipy.linalg.cython_blas.dgemv(transn,&grad_nvar,&nsteps,&one,&grad_K_fun[0,0,0],&grad_nvar,&b_table_x[0],&int_one,&one,&grad_x[0,0],&int_one)

            # grad_v = grad_v + b_table_v^T . grad_K_gun
            scipy.linalg.cython_blas.dgemv(transn,&grad_nvar,&nsteps,&one,&grad_K_gun[0,0,0],&grad_nvar,&b_table_v[0],&int_one,&one,&grad_v[0,0],&int_one)

        for jdof in range(ndof):
            x_keep[iint_keep,jdof] = x[jdof]
        for jdof in range(ndof):
            v_keep[iint_keep,jdof] = v[jdof]

        for jdof in range(ndof):
            for kdof in range(grad_ndof):
                grad_x_keep[iint_keep,jdof,kdof] = grad_x[jdof,kdof]
        for jdof in range(ndof):
            for kdof in range(grad_ndof):
                grad_v_keep[iint_keep,jdof,kdof] = grad_v[jdof,kdof]
    
    # print('Avg nit fun & gun : ',tot_niter/nint)
    # print('Avg nit grad fun & gun : ',grad_tot_niter/nint)
    # print(1+nsteps*tot_niter)

    return x_keep, v_keep, grad_x_keep, grad_v_keep



