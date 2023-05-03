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

def ExplicitSymplecticWithTable_VX_cython(
    object fun,
    object gun,
    (double, double) t_span,
    np.ndarray[double, ndim=1, mode="c"] x0,
    np.ndarray[double, ndim=1, mode="c"] v0,
    long nint,
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

    cdef double tx = t_span[0]
    cdef double tv = t_span[0]
    cdef double dt = (t_span[1] - t_span[0]) / nint

    cdef np.ndarray[double, ndim=1, mode="c"] cdt = np.empty((nsteps),dtype=np.float64)
    cdef np.ndarray[double, ndim=1, mode="c"] ddt = np.empty((nsteps),dtype=np.float64)

    cdef np.ndarray[double, ndim=1, mode="c"] x = x0.copy()
    cdef np.ndarray[double, ndim=1, mode="c"] v = v0.copy()

    cdef long ndof = x0.size
    cdef np.ndarray[double, ndim=1, mode="c"] res

    cdef long istep,id
    for istep in range(nsteps):
        cdt[istep] = c_table[istep]*dt
        ddt[istep] = d_table[istep]*dt

    for iint in range(nint):

        for istep in range(nsteps):

            res = fun(tv,v)  
            for idof in range(ndof):
                x[idof] += cdt[istep] * res[idof]  

            tx += cdt[istep]

            res = gun(tx,x)   
            for idof in range(ndof):
                v[idof] += ddt[istep] * res[idof]  
            tv += ddt[istep]

    return x,v

def ExplicitSymplecticWithTable_XV_cython(
    object fun,
    object gun,
    (double, double) t_span,
    np.ndarray[double, ndim=1, mode="c"] x0,
    np.ndarray[double, ndim=1, mode="c"] v0,
    long nint,
    np.ndarray[double, ndim=1, mode="c"] c_table,
    np.ndarray[double, ndim=1, mode="c"] d_table,
    long nsteps
):

    cdef double tx = t_span[0]
    cdef double tv = t_span[0]
    cdef double dt = (t_span[1] - t_span[0]) / nint

    cdef np.ndarray[double, ndim=1, mode="c"] cdt = np.empty((nsteps),dtype=np.float64)
    cdef np.ndarray[double, ndim=1, mode="c"] ddt = np.empty((nsteps),dtype=np.float64)

    cdef np.ndarray[double, ndim=1, mode="c"] x = x0.copy()
    cdef np.ndarray[double, ndim=1, mode="c"] v = v0.copy()

    cdef long ndof = x0.size
    cdef np.ndarray[double, ndim=1, mode="c"] res

    cdef long istep,idof

    for istep in range(nsteps):
        cdt[istep] = c_table[istep]*dt
        ddt[istep] = d_table[istep]*dt

    for iint in range(nint):

        for istep in range(nsteps):

            res = gun(tx,x)   
            for idof in range(ndof):
                v[idof] += cdt[istep] * res[idof]  
            tv += cdt[istep]

            res = fun(tv,v)  
            for idof in range(ndof):
                x[idof] += ddt[istep] * res[idof]  

            tx += ddt[istep]

    return x,v

def SymplecticStormerVerlet_XV_cython(
    object fun,
    object gun,
    (double, double) t_span,
    np.ndarray[double, ndim=1, mode="c"] x0,
    np.ndarray[double, ndim=1, mode="c"] v0,
    long nint,
):

    cdef double t = t_span[0]
    cdef double dt = (t_span[1] - t_span[0]) / nint
    cdef double dt_half = dt*0.5

    cdef np.ndarray[double, ndim=1, mode="c"] x = x0.copy()
    cdef np.ndarray[double, ndim=1, mode="c"] v = v0.copy()

    cdef long ndof = x0.size
    cdef np.ndarray[double, ndim=1, mode="c"] res

    cdef long idof

    res = gun(t,x)   
    for idof in range(ndof):
        v[idof] += dt_half * res[idof]  

    for iint in range(nint-1):

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

    return x,v

def SymplecticStormerVerlet_VX_cython(
    object fun,
    object gun,
    (double, double) t_span,
    np.ndarray[double, ndim=1, mode="c"] x0,
    np.ndarray[double, ndim=1, mode="c"] v0,
    long nint,
):

    cdef double t = t_span[0]
    cdef double dt = (t_span[1] - t_span[0]) / nint
    cdef double dt_half = dt*0.5

    cdef np.ndarray[double, ndim=1, mode="c"] x = x0.copy()
    cdef np.ndarray[double, ndim=1, mode="c"] v = v0.copy()

    cdef long ndof = x0.size
    cdef np.ndarray[double, ndim=1, mode="c"] res

    cdef long idof

    res = fun(t,v)  
    for idof in range(ndof):
        x[idof] += dt_half * res[idof]  

    t += dt_half
    
    for iint in range(nint-1):

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

    t += dt_half

    return x,v



def ImplicitSymplecticWithTableGaussSeidel_VX_cython(
    object fun,
    object gun,
    (double, double) t_span,
    np.ndarray[double, ndim=1, mode="c"] x0,
    np.ndarray[double, ndim=1, mode="c"] v0,
    long nint,
    np.ndarray[double, ndim=2, mode="c"] a_table,
    np.ndarray[double, ndim=1, mode="c"] b_table,
    np.ndarray[double, ndim=1, mode="c"] c_table,
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
    cdef long istep, id, iGS
    cdef long i,j,k

    cdef bint GoOnGS

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
    cdef np.ndarray[double, ndim=1, mode="c"] dxdt = np.empty((ndof),dtype=np.float64)
    cdef np.ndarray[double, ndim=2, mode="c"] all_res = np.empty((nsteps,ndof),dtype=np.float64)

    cdef np.ndarray[double, ndim=2, mode="c"] dX = np.empty((nsteps,ndof),dtype=np.float64)
    cdef np.ndarray[double, ndim=2, mode="c"] dV

    cdef double eps_mul = eps * ndof * nsteps


    for iint in range(nint):

        tbeg = t_span[0] + iint * dt
        for istep in range(nsteps):
            all_t[istep] = tbeg + cdt[istep]



        # TODO : Apply starting approximation to V
        dV  = np.zeros((nsteps,ndof),dtype=np.float64)



        iGS = 0

        GoOnGS = True

        while GoOnGS:

            # dV => dX
            for istep in range(nsteps):

                for idof in range(ndof):
                    arg[idof] = v[idof] + dV[istep,idof]

                res = fun(all_t[istep],arg)  

                for idof in range(ndof):
                    all_res[istep,idof] = dt * res[idof]


            for i in range(nsteps):
                for j in range(ndof):
                    dX[i,j] = a_table[i,0] * all_res[0,j]
                    for k in range(1,nsteps):
                        dX[i,j] += a_table[i,k] * all_res[k,j]


            # dX => dV
            for istep in range(nsteps):

                for idof in range(ndof):
                    arg[idof] = x[idof] + dX[istep,idof]

                res = gun(all_t[istep],arg)  

                for idof in range(ndof):
                    all_res[istep,idof] = dt * res[idof]
                    
            for i in range(nsteps):
                for j in range(ndof):
                    dV[i,j] = a_table[i,0] * all_res[0,j]
                    for k in range(1,nsteps):
                        dV[i,j] += a_table[i,k] * all_res[k,j]


            
            iGS += 1

            GoOnGS = (iGS < maxiter)

        # x
        for istep in range(nsteps):

            for idof in range(ndof):
                arg[idof] = v[idof] + dV[istep,idof]

            res = fun(all_t[istep],arg)  

            for idof in range(ndof):
                all_res[istep,idof] = res[idof]

        for i in range(ndof):
            dxdt[i] = b_table[0] * all_res[0,i]
            for k in range(1,nsteps):
                dxdt[i] += b_table[k] * all_res[k,i]

        # v
        for istep in range(nsteps):

            for idof in range(ndof):
                arg[idof] = x[idof] + dX[istep,idof]

            res = gun(all_t[istep],arg)  

            for idof in range(ndof):
                all_res[istep,idof] = res[idof]

        for i in range(ndof):
            res[i] = b_table[0] * all_res[0,i]
            for k in range(1,nsteps):
                res[i] += b_table[k] * all_res[k,i]


        for idof in range(ndof):
            x[idof] += dt * dxdt[idof]

        for idof in range(ndof):
            v[idof] += dt * res[idof]


        

    return x,v



