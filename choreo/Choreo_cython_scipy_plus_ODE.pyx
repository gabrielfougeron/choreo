'''
Choreo_cython_scipy_plus_ODE.pyx : Defines ODE-related things I designed I feel ought to be in scipy ... but faster !


'''


import os
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

# @cython.cdivision(True)
# def SymplecticStormerVerlet_XV_cython(
#     object fun,
#     object gun,
#     (double, double) t_span,
#     np.ndarray[double, ndim=1, mode="c"] x0,
#     np.ndarray[double, ndim=1, mode="c"] v0,
#     long nint,
#     long keep_freq,
# ):
# 
#     cdef long iint_keep, ifreq
#     cdef long nint_keep = nint // keep_freq
#     cdef long ndof = x0.size
# 
#     cdef np.ndarray[double, ndim=2, mode="c"] x_keep = np.empty((nint_keep,ndof),dtype=np.float64)
#     cdef np.ndarray[double, ndim=2, mode="c"] v_keep = np.empty((nint_keep,ndof),dtype=np.float64)
# 
#     cdef double t = t_span[0]
#     cdef double dt = (t_span[1] - t_span[0]) / nint
#     cdef double dt_half = dt*0.5
# 
#     cdef np.ndarray[double, ndim=1, mode="c"] x = x0.copy()
#     cdef np.ndarray[double, ndim=1, mode="c"] v = v0.copy()
# 
#     cdef np.ndarray[double, ndim=1, mode="c"] res
# 
#     cdef long idof
# 
#     res = gun(t,x)   
#     for idof in range(ndof):
#         v[idof] += dt_half * res[idof]  
# 
#     for iint in range(nint-1):
# 
#         res = fun(t,v)  
#         for idof in range(ndof):
#             x[idof] += dt* res[idof]  
# 
#         t += dt
# 
#         res = gun(t,x)   
#         for idof in range(ndof):
#             v[idof] += dt * res[idof]  
# 
#     res = fun(t,v)  
#     for idof in range(ndof):
#         x[idof] += dt * res[idof]  
# 
#     t += dt
# 
#     res = gun(t,x)   
#     for idof in range(ndof):
#         v[idof] += dt_half * res[idof]  
# 
#     return x_keep, v_keep
# 
# @cython.cdivision(True)
# def SymplecticStormerVerlet_VX_cython(
#     object fun,
#     object gun,
#     (double, double) t_span,
#     np.ndarray[double, ndim=1, mode="c"] x0,
#     np.ndarray[double, ndim=1, mode="c"] v0,
#     long nint,
# ):
# 
#     cdef double t = t_span[0]
#     cdef double dt = (t_span[1] - t_span[0]) / nint
#     cdef double dt_half = dt*0.5
# 
#     cdef np.ndarray[double, ndim=1, mode="c"] x = x0.copy()
#     cdef np.ndarray[double, ndim=1, mode="c"] v = v0.copy()
# 
#     cdef long ndof = x0.size
#     cdef np.ndarray[double, ndim=1, mode="c"] res
# 
#     cdef long idof
# 
#     res = fun(t,v)  
#     for idof in range(ndof):
#         x[idof] += dt_half * res[idof]  
# 
#     t += dt_half
#     
#     for iint in range(nint-1):
# 
#         res = gun(t,x)   
#         for idof in range(ndof):
#             v[idof] += dt * res[idof]  
# 
#         res = fun(t,v)  
#         for idof in range(ndof):
#             x[idof] += dt* res[idof]  
# 
#         t += dt
# 
#     res = gun(t,x)   
#     for idof in range(ndof):
#         v[idof] += dt * res[idof]  
# 
#     res = fun(t,v)  
#     for idof in range(ndof):
#         x[idof] += dt_half* res[idof]  
# 
#     t += dt_half
# 
#     return x,v


def ImplicitSymplecticWithTableGaussSeidel_VX_cython(
    object fun,
    object gun,
    (double, double) t_span,
    np.ndarray[double, ndim=1, mode="c"] x0,
    np.ndarray[double, ndim=1, mode="c"] v0,
    long nint,
    long keep_freq,
    np.ndarray[double, ndim=2, mode="c"] a_table,
    np.ndarray[double, ndim=1, mode="c"] b_table,
    np.ndarray[double, ndim=1, mode="c"] c_table,
    np.ndarray[double, ndim=2, mode="c"] beta_table,
    long nsteps,
    double eps,
    long maxiter
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
    cdef long ndof = x0.size
    cdef long istep, id, iGS, jdof
    cdef long kstep
    cdef long tot_niter = 0
    cdef long iint_keep, ifreq
    cdef long nint_keep = nint // keep_freq

    cdef np.ndarray[double, ndim=2, mode="c"] x_keep = np.empty((nint_keep,ndof),dtype=np.float64)
    cdef np.ndarray[double, ndim=2, mode="c"] v_keep = np.empty((nint_keep,ndof),dtype=np.float64)

    cdef bint GoOnGS

    cdef double dXV_err, dX_err,dV_err, diff
    cdef double tbeg, t
    cdef double dt = (t_span[1] - t_span[0]) / nint

    cdef np.ndarray[double, ndim=1, mode="c"] cdt = np.empty((nsteps),dtype=np.float64)
    cdef np.ndarray[double, ndim=1, mode="c"] all_t = np.empty((nsteps),dtype=np.float64)
    cdef np.ndarray[double, ndim=1, mode="c"] arg = np.empty((ndof),dtype=np.float64)

    for istep in range(nsteps):
        cdt[istep] = c_table[istep]*dt

    cdef np.ndarray[double, ndim=1, mode="c"] x = x0.copy()
    cdef np.ndarray[double, ndim=1, mode="c"] v = v0.copy()

    cdef np.ndarray[double, ndim=1, mode="c"] res
    cdef np.ndarray[double, ndim=2, mode="c"] K_fun = np.empty((nsteps,ndof),dtype=np.float64)
    cdef np.ndarray[double, ndim=2, mode="c"] K_gun = np.zeros((nsteps,ndof),dtype=np.float64) # zeros here to ensure correct first init.

    cdef np.ndarray[double, ndim=2, mode="c"] dX = np.zeros((nsteps,ndof),dtype=np.float64)
    cdef np.ndarray[double, ndim=2, mode="c"] dV = np.zeros((nsteps,ndof),dtype=np.float64) 
    cdef np.ndarray[double, ndim=2, mode="c"] dX_prev = np.empty((nsteps,ndof),dtype=np.float64)
    cdef np.ndarray[double, ndim=2, mode="c"] dV_prev = np.empty((nsteps,ndof),dtype=np.float64) 

    cdef double eps_mul = eps * ndof * nsteps * dt
    cdef double eps_mul2 = eps_mul * eps_mul

    # First starting approximation using only one function evaluation
    tbeg = t_span[0]
    res = gun(tbeg,x)  
    for istep in range(nsteps):
        for jdof in range(ndof):
            dV[istep,jdof] = cdt[istep] * res[jdof]

    for iint_keep in range(nint_keep):

        for ifreq in range(keep_freq):

            iint = iint_keep * keep_freq + ifreq

            tbeg = t_span[0] + iint * dt    
            for istep in range(nsteps):
                all_t[istep] = tbeg + cdt[istep]

            for istep in range(nsteps):
                for jdof in range(ndof):
                    dV[istep,jdof] = beta_table[istep,0] * K_gun[0,jdof]
                    for kstep in range(1,nsteps):
                        dV[istep,jdof] += beta_table[istep,kstep] * K_gun[kstep,jdof]

            iGS = 0

            GoOnGS = True

            while GoOnGS:

                # dV => dX
                for istep in range(nsteps):

                    for jdof in range(ndof):
                        arg[jdof] = v[jdof] + dV[istep,jdof]

                    res = fun(all_t[istep],arg)  

                    for jdof in range(ndof):
                        K_fun[istep,jdof] = dt * res[jdof]

                for istep in range(nsteps):
                    for jdof in range(ndof):
                        dX_prev[istep,jdof] = dX[istep,jdof] 


                for istep in range(nsteps):
                    for jdof in range(ndof):
                        dX[istep,jdof] = a_table[istep,0] * K_fun[0,jdof]
                        for kstep in range(1,nsteps):
                            dX[istep,jdof] += a_table[istep,kstep] * K_fun[kstep,jdof]


                # dX => dV
                for istep in range(nsteps):

                    for jdof in range(ndof):
                        arg[jdof] = x[jdof] + dX[istep,jdof]

                    res = gun(all_t[istep],arg)  

                    for jdof in range(ndof):
                        K_gun[istep,jdof] = dt * res[jdof]

                for istep in range(nsteps):
                    for jdof in range(ndof):
                        dV_prev[istep,jdof] = dV[istep,jdof] 
                        
                for istep in range(nsteps):
                    for jdof in range(ndof):
                        dV[istep,jdof] = a_table[istep,0] * K_gun[0,jdof]
                        for kstep in range(1,nsteps):
                            dV[istep,jdof] += a_table[istep,kstep] * K_gun[kstep,jdof]

                dX_err = 0
                for istep in range(nsteps):
                    for jdof in range(ndof):
                        diff = dX[istep,jdof] - dX_prev[istep,jdof]
                        dX_err += diff * diff

                dV_err = 0
                for istep in range(nsteps):
                    for jdof in range(ndof):
                        diff = dV[istep,jdof] - dV_prev[istep,jdof]
                        dV_err += diff * diff
                
                dXV_err = dX_err + dV_err  
                # dXV_err = dX_err + dV_err * dt * dt 

                iGS += 1

                GoOnGS = (iGS < maxiter) and (dXV_err > eps_mul2)

            # if (iGS >= maxiter):
            #     print("Max iter exceeded")

            tot_niter += iGS

            # Do EFT here ?

            for kstep in range(nsteps):
                for jdof in range(ndof):
                    x[jdof] += b_table[kstep] * K_fun[kstep,jdof]

            for kstep in range(nsteps):
                for jdof in range(ndof):
                    v[jdof] += b_table[kstep] * K_gun[kstep,jdof]
        
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
    long nint,
    long keep_freq,
    np.ndarray[double, ndim=2, mode="c"] a_table,
    np.ndarray[double, ndim=1, mode="c"] b_table,
    np.ndarray[double, ndim=1, mode="c"] c_table,
    np.ndarray[double, ndim=2, mode="c"] beta_table,
    long nsteps,
    double eps,
    long maxiter
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
    cdef long ndof = x0.size
    # ~ cdef long grad_ndof = grad_x0.shape[1] # Does this not work on numpy arrays ?
    cdef long grad_ndof = grad_x0.size // ndof
    cdef long istep, id, iGS, jdof, kdof
    cdef long kstep
    cdef long tot_niter = 0
    cdef long grad_tot_niter = 0
    cdef long iint_keep, ifreq
    cdef long nint_keep = nint // keep_freq

    cdef np.ndarray[double, ndim=2, mode="c"] x_keep = np.empty((nint_keep,ndof),dtype=np.float64)
    cdef np.ndarray[double, ndim=2, mode="c"] v_keep = np.empty((nint_keep,ndof),dtype=np.float64)

    cdef np.ndarray[double, ndim=3, mode="c"] grad_x_keep = np.empty((nint_keep,ndof,grad_ndof),dtype=np.float64)
    cdef np.ndarray[double, ndim=3, mode="c"] grad_v_keep = np.empty((nint_keep,ndof,grad_ndof),dtype=np.float64)

    cdef bint GoOnGS

    cdef double dXV_err, diff
    cdef double tbeg, t
    cdef double dt = (t_span[1] - t_span[0]) / nint

    cdef np.ndarray[double, ndim=1, mode="c"] cdt = np.empty((nsteps),dtype=np.float64)
    cdef np.ndarray[double, ndim=1, mode="c"] all_t = np.empty((nsteps),dtype=np.float64)
    cdef np.ndarray[double, ndim=1, mode="c"] arg = np.empty((ndof),dtype=np.float64)
    cdef np.ndarray[double, ndim=2, mode="c"] grad_arg = np.empty((ndof,grad_ndof),dtype=np.float64)

    for istep in range(nsteps):
        cdt[istep] = c_table[istep]*dt

    cdef np.ndarray[double, ndim=1, mode="c"] x = x0.copy()
    cdef np.ndarray[double, ndim=1, mode="c"] v = v0.copy()

    cdef np.ndarray[double, ndim=2, mode="c"] grad_x = grad_x0.copy()
    cdef np.ndarray[double, ndim=2, mode="c"] grad_v = grad_v0.copy()

    cdef np.ndarray[double, ndim=1, mode="c"] res
    cdef np.ndarray[double, ndim=2, mode="c"] K_fun = np.empty((nsteps,ndof),dtype=np.float64)
    cdef np.ndarray[double, ndim=2, mode="c"] K_gun = np.zeros((nsteps,ndof),dtype=np.float64) # zeros here to ensure correct first init.

    cdef np.ndarray[double, ndim=2, mode="c"] dX = np.zeros((nsteps,ndof),dtype=np.float64)
    cdef np.ndarray[double, ndim=2, mode="c"] dV = np.zeros((nsteps,ndof),dtype=np.float64) 
    cdef np.ndarray[double, ndim=2, mode="c"] dX_prev = np.empty((nsteps,ndof),dtype=np.float64)
    cdef np.ndarray[double, ndim=2, mode="c"] dV_prev = np.empty((nsteps,ndof),dtype=np.float64) 


    cdef np.ndarray[double, ndim=2, mode="c"] grad_res
    cdef np.ndarray[double, ndim=3, mode="c"] grad_K_fun = np.empty((nsteps,ndof,grad_ndof),dtype=np.float64)
    cdef np.ndarray[double, ndim=3, mode="c"] grad_K_gun = np.zeros((nsteps,ndof,grad_ndof),dtype=np.float64) # zeros here to ensure correct first init.

    cdef np.ndarray[double, ndim=3, mode="c"] grad_dX = np.zeros((nsteps,ndof,grad_ndof),dtype=np.float64)
    cdef np.ndarray[double, ndim=3, mode="c"] grad_dV = np.zeros((nsteps,ndof,grad_ndof),dtype=np.float64) 
    cdef np.ndarray[double, ndim=3, mode="c"] grad_dX_prev = np.empty((nsteps,ndof,grad_ndof),dtype=np.float64)
    cdef np.ndarray[double, ndim=3, mode="c"] grad_dV_prev = np.empty((nsteps,ndof,grad_ndof),dtype=np.float64) 

    cdef double eps_mul = eps * ndof * nsteps * dt
    # cdef double eps_mul = eps * dt
    cdef double eps_mul2 = eps_mul * eps_mul

    cdef double grad_eps_mul = eps * ndof * grad_ndof * nsteps * dt
    cdef double grad_eps_mul2 = grad_eps_mul * grad_eps_mul

    # First starting approximation using only one function evaluation
    tbeg = t_span[0]
    res = gun(tbeg,x)  
    for istep in range(nsteps):
        for jdof in range(ndof):
            dV[istep,jdof] = cdt[istep] * res[jdof]

    grad_res = grad_gun(tbeg,x,grad_x)  
    for istep in range(nsteps):
        for jdof in range(ndof):
            for kdof in range(grad_ndof):
                grad_dV[istep,jdof,kdof] = cdt[istep] * grad_res[jdof,kdof]

    for iint_keep in range(nint_keep):

        for ifreq in range(keep_freq):

            iint = iint_keep * keep_freq + ifreq

            tbeg = t_span[0] + iint * dt    
            for istep in range(nsteps):
                all_t[istep] = tbeg + cdt[istep]

            for istep in range(nsteps):
                for jdof in range(ndof):
                    dV[istep,jdof] = beta_table[istep,0] * K_gun[0,jdof]
                    for kstep in range(1,nsteps):
                        dV[istep,jdof] += beta_table[istep,kstep] * K_gun[kstep,jdof]

            iGS = 0
            GoOnGS = True

            while GoOnGS:

                # dV => dX
                for istep in range(nsteps):

                    for jdof in range(ndof):
                        arg[jdof] = v[jdof] + dV[istep,jdof]

                    res = fun(all_t[istep],arg)  

                    for jdof in range(ndof):
                        K_fun[istep,jdof] = dt * res[jdof]

                for istep in range(nsteps):
                    for jdof in range(ndof):
                        dX_prev[istep,jdof] = dX[istep,jdof] 


                for istep in range(nsteps):
                    for jdof in range(ndof):
                        dX[istep,jdof] = a_table[istep,0] * K_fun[0,jdof]
                        for kstep in range(1,nsteps):
                            dX[istep,jdof] += a_table[istep,kstep] * K_fun[kstep,jdof]

                # dX => dV
                for istep in range(nsteps):

                    for jdof in range(ndof):
                        arg[jdof] = x[jdof] + dX[istep,jdof]

                    res = gun(all_t[istep],arg)  

                    for jdof in range(ndof):
                        K_gun[istep,jdof] = dt * res[jdof]

                for istep in range(nsteps):
                    for jdof in range(ndof):
                        dV_prev[istep,jdof] = dV[istep,jdof] 
                        
                for istep in range(nsteps):
                    for jdof in range(ndof):
                        dV[istep,jdof] = a_table[istep,0] * K_gun[0,jdof]
                        for kstep in range(1,nsteps):
                            dV[istep,jdof] += a_table[istep,kstep] * K_gun[kstep,jdof]

                dXV_err = 0.
                for istep in range(nsteps):
                    for jdof in range(ndof):
                        diff = dX[istep,jdof] - dX_prev[istep,jdof]
                        dXV_err += diff * diff
                        diff = dV[istep,jdof] - dV_prev[istep,jdof]
                        dXV_err += diff * diff
                
                iGS += 1

                GoOnGS = (iGS < maxiter) and (dXV_err > eps_mul2)

            if (iGS >= maxiter):
                print("Max iter exceeded")

            tot_niter += iGS

            iGS = 0
            GoOnGS = True

            for istep in range(nsteps):
                for jdof in range(ndof):
                    for kdof in range(grad_ndof):
                        grad_dV[istep,jdof,kdof] = beta_table[istep,0] * grad_K_gun[0,jdof,kdof]                        
                        for kstep in range(1,nsteps):
                            grad_dV[istep,jdof,kdof] += beta_table[istep,kstep] * grad_K_gun[kstep,jdof,kdof]

            while GoOnGS:

                # grad_dV => grad_dX
                for istep in range(nsteps):

                    for jdof in range(ndof):
                        arg[jdof] = v[jdof] + dV[istep,jdof]

                    for jdof in range(ndof):
                        for kdof in range(grad_ndof):
                            grad_arg[jdof,kdof] = grad_v[jdof,kdof] + grad_dV[istep,jdof,kdof]

                    grad_res = grad_fun(all_t[istep],arg,grad_arg)  

                    for jdof in range(ndof):
                        for kdof in range(grad_ndof):
                            grad_K_fun[istep,jdof,kdof] = dt * grad_res[jdof,kdof]

                for istep in range(nsteps):
                    for jdof in range(ndof):
                        for kdof in range(grad_ndof):
                            grad_dX_prev[istep,jdof,kdof] = grad_dX[istep,jdof,kdof] 

                for istep in range(nsteps):
                    for jdof in range(ndof):
                        for kdof in range(grad_ndof):
                            grad_dX[istep,jdof,kdof] = a_table[istep,0] * grad_K_fun[0,jdof,kdof]
                            for kstep in range(1,nsteps):
                                grad_dX[istep,jdof,kdof] += a_table[istep,kstep] * grad_K_fun[kstep,jdof,kdof]

                # grad_dX => grad_dV
                for istep in range(nsteps):

                    for jdof in range(ndof):
                        arg[jdof] = x[jdof] + dX[istep,jdof]

                    for jdof in range(ndof):
                        for kdof in range(grad_ndof):
                            grad_arg[jdof,kdof] = grad_x[jdof,kdof] + grad_dX[istep,jdof,kdof]

                    grad_res = grad_gun(all_t[istep],arg,grad_arg)  

                    for jdof in range(ndof):
                        for kdof in range(grad_ndof):
                            grad_K_gun[istep,jdof,kdof] = dt * grad_res[jdof,kdof]

                for istep in range(nsteps):
                    for jdof in range(ndof):
                        for kdof in range(grad_ndof):
                            grad_dV_prev[istep,jdof,kdof] = grad_dV[istep,jdof,kdof] 
                        
                for istep in range(nsteps):
                    for jdof in range(ndof):
                        for kdof in range(grad_ndof):
                            grad_dV[istep,jdof,kdof] = a_table[istep,0] * grad_K_gun[0,jdof,kdof]
                            for kstep in range(1,nsteps):
                                grad_dV[istep,jdof,kdof] += a_table[istep,kstep] * grad_K_gun[kstep,jdof,kdof]



                            

                dXV_err = 0.
                for istep in range(nsteps):
                    for jdof in range(ndof):
                        for kdof in range(grad_ndof):
                            diff = grad_dX[istep,jdof,kdof] - grad_dX_prev[istep,jdof,kdof]
                            dXV_err += diff * diff
                            diff = grad_dV[istep,jdof,kdof] - grad_dV_prev[istep,jdof,kdof]
                            dXV_err += diff * diff
                
                iGS += 1

                GoOnGS = (iGS < maxiter) and (dXV_err > grad_eps_mul)

            if (iGS >= maxiter):
                print("Max iter exceeded")

            grad_tot_niter += iGS

            # Do EFT here ?

            for kstep in range(nsteps):
                for jdof in range(ndof):
                    x[jdof] += b_table[kstep] * K_fun[kstep,jdof]

            for kstep in range(nsteps):
                for jdof in range(ndof):
                    v[jdof] += b_table[kstep] * K_gun[kstep,jdof]
        
            for kstep in range(nsteps):
                for jdof in range(ndof):
                    for kdof in range(grad_ndof):
                        grad_x[jdof,kdof] += b_table[kstep] * grad_K_fun[kstep,jdof,kdof]

            for kstep in range(nsteps):
                for jdof in range(ndof):
                    for kdof in range(grad_ndof):
                        grad_v[jdof,kdof] += b_table[kstep] * grad_K_gun[kstep,jdof,kdof]
        
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
    
    # print(tot_niter/nint)
    # print(grad_tot_niter/nint)
    # print(1+nsteps*tot_niter)

    return x_keep, v_keep, grad_x_keep, grad_v_keep



