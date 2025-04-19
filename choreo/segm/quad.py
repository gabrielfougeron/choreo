'''
quad.py : Defines segment quadrature related things.

'''

import numpy as np
import math as m
import mpmath
import functools
import scipy

from choreo.segm.cython.quad import *

try:
    import numba 
    UseNumba = True
except ImportError:
    UseNumba = False

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
