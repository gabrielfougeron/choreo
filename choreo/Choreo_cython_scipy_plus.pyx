'''
Choreo_cython_scipy_plus.pyx : Defines things I designed I feel ought to be in scipy ... but faster !


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

    cdef np.ndarray[double, ndim=1, mode="c"] cdt = np.zeros((nsteps),dtype=np.float64)
    cdef np.ndarray[double, ndim=1, mode="c"] ddt = np.zeros((nsteps),dtype=np.float64)

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

    cdef np.ndarray[double, ndim=1, mode="c"] cdt = np.zeros((nsteps),dtype=np.float64)
    cdef np.ndarray[double, ndim=1, mode="c"] ddt = np.zeros((nsteps),dtype=np.float64)

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




