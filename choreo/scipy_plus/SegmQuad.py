'''
ODE.py : Defines ODE-related things I designed I feel ought to be in scipy.

'''

import numpy as np
import math as m
import mpmath
import functools
import scipy

from choreo.scipy_plus.cython.SegmQuad import IntegrateOnSegment
from choreo.scipy_plus.cython.SegmQuad import QuadFormula

from choreo.scipy_plus.multiprec_tables import ShiftedGaussLegendre3Term
from choreo.scipy_plus.multiprec_tables import QuadFrom3Term

try:
    import numba 
    UseNumba = True
except ImportError:
    UseNumba = False

@functools.cache
def ComputeQuadrature(method,n,dps=30):

    if method == "Gauss" :
        a, b = ShiftedGaussLegendre3Term(n)
        th_cvg_rate = 2*n
        
    else:
        raise ValueError(f"Method not found: {method}")
    
    w, z = QuadFrom3Term(a,b,n)

    w_np = np.array(w.tolist(),dtype=np.float64).reshape(n)
    z_np = np.array(z.tolist(),dtype=np.float64).reshape(n)
    
    return QuadFormula(
        w = w_np                    ,
        x = z_np                    ,
        th_cvg_rate = th_cvg_rate   ,
    )

if UseNumba:
    # Define decorators to make scipy.LowLevelCallable from python functions using numba
    
    default_numba_kwargs = {
        'nopython':True     ,
        'cache':True        ,
        'fastmath':True     ,
        'nogil':True        ,
    }

    def nb_jit_double_double(integrand_function, numba_kwargs = default_numba_kwargs):
        jitted_function = numba.jit(integrand_function, **numba_kwargs)

        #double func(double x)
        @numba.cfunc(numba.types.float64(numba.types.float64))
        def wrapped(x):        
            return jitted_function(x)
        
        return scipy.LowLevelCallable(wrapped.ctypes)

    def nb_jit_array_double(integrand_function, numba_kwargs = default_numba_kwargs):
        jitted_function = numba.jit(integrand_function, **numba_kwargs)

        #func(double x, double * res)
        @numba.cfunc(numba.types.void(numba.types.float64, numba.types.CPointer(numba.types.float64)))
        def wrapped(x, res):   
            res = jitted_function(x)
        
        return scipy.LowLevelCallable(wrapped.ctypes)

    def nb_jit_inplace_double_array(integrand_function, numba_kwargs = default_numba_kwargs):
        jitted_function = numba.jit(integrand_function, **numba_kwargs)

        #func(double x, double * res)
        @numba.cfunc(numba.types.void(numba.types.float64, numba.types.CPointer(numba.types.float64)))
        def wrapped(x, res):   
            jitted_function(x, res)
        
        return scipy.LowLevelCallable(wrapped.ctypes)
