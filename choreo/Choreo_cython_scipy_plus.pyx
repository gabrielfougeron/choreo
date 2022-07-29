#cython: language_level=3, boundscheck=False, wraparound = False

'''
Choreo_cython_scipy_plus.pyx : Defines things I designed I feel ought to be in scipy ... but faster !

'''


import os
import numpy as np
cimport numpy as np
cimport cython

from libc.math cimport pow as cpow
from libc.math cimport fabs as cfabs
from libc.math cimport cos as ccos
from libc.math cimport sin as csin
from libc.math cimport sqrt as csqrt
from libc.math cimport isnan as cisnan
from libc.math cimport isinf as cisinf

def SymplecticWithTable_VX_cython(
    object fun,
    object gun,
    (double, double) t_span,
    double [:] x0,
    double [:] v0,
    long nint,
    double [:] c_table,
    double [:] d_table,
    long nsteps
    ):

    # Warning : x0 and v0 might get erased

    cdef double tx = t_span[0]
    cdef double tv = t_span[0]
    cdef double dt = (t_span[1] - t_span[0]) / nint

    cdef double [:] cdt = np.zeros((nsteps),dtype=np.float64)
    cdef double [:] ddt = np.zeros((nsteps),dtype=np.float64)

    cdef double [:] x = x0
    cdef double [:] v = v0

    cdef long ndof = x0.size
    cdef double [:] res

    cdef long istep,id
    for istep in range(nsteps):
        cdt[istep] = c_table[istep]*dt
        ddt[istep] = d_table[istep]*dt

    for iint in range(nint):

        for istep in range(nsteps):

            res = fun(tv,np.asarray(v))  
            for idof in range(ndof):
                x[idof] += cdt[istep] * res[idof]  

            tx += cdt[istep]

            res = gun(tx,np.asarray(x))   
            for idof in range(ndof):
                v[idof] += ddt[istep] * res[idof]  
            tv += ddt[istep]


    return np.asarray(x),np.asarray(v)

def SymplecticWithTable_XV_cython(
    object fun,
    object gun,
    (double, double) t_span,
    double [:] x0,
    double [:] v0,
    long nint,
    double [:] c_table,
    double [:] d_table,
    long nsteps
    ):

    # Warning : x0 and v0 might get erased

    cdef double tx = t_span[0]
    cdef double tv = t_span[0]
    cdef double dt = (t_span[1] - t_span[0]) / nint

    cdef double [:] cdt = np.zeros((nsteps),dtype=np.float64)
    cdef double [:] ddt = np.zeros((nsteps),dtype=np.float64)

    cdef double [:] x = x0
    cdef double [:] v = v0

    cdef long ndof = x0.size
    cdef double [:] res

    cdef long istep,idof

    for istep in range(nsteps):
        cdt[istep] = c_table[istep]*dt
        ddt[istep] = d_table[istep]*dt

    for iint in range(nint):

        for istep in range(nsteps):

            res = gun(tx,np.asarray(x))   
            for idof in range(ndof):
                v[idof] += cdt[istep] * res[idof]  
            tv += cdt[istep]

            res = fun(tv,np.asarray(v))  
            for idof in range(ndof):
                x[idof] += ddt[istep] * res[idof]  

            tx += ddt[istep]

    return np.asarray(x),np.asarray(v)

def SymplecticStormerVerlet_XV_cython(
    object fun,
    object gun,
    (double, double) t_span,
    double [:]  x0,
    double [:]  v0,
    long nint,
    ):

    # Warning : x0 and v0 might get erased

    cdef double t = t_span[0]
    cdef double dt = (t_span[1] - t_span[0]) / nint
    cdef double dt_half = dt*0.5

    cdef double [:]  x = x0
    cdef double [:]  v = v0

    cdef long ndof = x0.size
    cdef double [:]  res

    cdef long idof

    res = gun(t,x)   
    for idof in range(ndof):
        v[idof] += dt_half * res[idof]  

    for iint in range(nint-1):

        res = fun(t,np.asarray(v))  
        for idof in range(ndof):
            x[idof] += dt* res[idof]  

        t += dt

        res = gun(t,np.asarray(x))   
        for idof in range(ndof):
            v[idof] += dt * res[idof]  

    res = fun(t,np.asarray(v))  
    for idof in range(ndof):
        x[idof] += dt * res[idof]  

    t += dt

    res = gun(t,np.asarray(x))   
    for idof in range(ndof):
        v[idof] += dt_half * res[idof]  

    return np.asarray(x),np.asarray(v)

def SymplecticStormerVerlet_VX_cython(
    object fun,
    object gun,
    (double, double) t_span,
    double [:] x0,
    double [:] v0,
    long nint,
    ):

    # Warning : x0 and v0 might get erased

    cdef double t = t_span[0]
    cdef double dt = (t_span[1] - t_span[0]) / nint
    cdef double dt_half = dt*0.5

    cdef double [:] x = x0
    cdef double [:]  v = v0

    cdef long ndof = x0.size
    cdef double [:] res

    cdef long idof

    res = fun(t,np.asarray(v))  
    for idof in range(ndof):
        x[idof] += dt_half * res[idof]  

    t += dt_half
    
    for iint in range(nint-1):

        res = gun(t,np.asarray(x))   
        for idof in range(ndof):
            v[idof] += dt * res[idof]  

        res = fun(t,np.asarray(v))  
        for idof in range(ndof):
            x[idof] += dt* res[idof]  

        t += dt

    res = gun(t,np.asarray(x))   
    for idof in range(ndof):
        v[idof] += dt * res[idof]  

    res = fun(t,np.asarray(v))  
    for idof in range(ndof):
        x[idof] += dt_half* res[idof]  

    t += dt_half

    return np.asarray(x),np.asarray(v)


def ComputeSpectralODERes(
    fun,
    x_coeffs,
    ncoeffs,
    nint,
    ):

    pass

    # Computes the residuals of the spectral solve of the ODE dx/dt = f(t,x)

