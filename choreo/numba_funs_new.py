import os
import numpy as np
import numba
import multiprocessing
import scipy

max_num_threads = multiprocessing.cpu_count()

numba_kwargs = {
    'nopython':True     ,
    'cache':True        ,
    'fastmath':True     ,
    'nogil':True        ,
}

def pow_inter_law(n):

    nm2 = n-2
    mnnm1 = -n*(n-1)
    def pot_fun(xsq, res):
        
        a = xsq ** nm2
        b = xsq*a

        res[0] = -xsq*b
        res[1] = -n*b
        res[2] = mnnm1*a

    return jit_inter_law(pot_fun)

def jit_inter_law(py_inter_law):
    
    sig = numba.types.void(numba.types.float64, numba.types.CPointer(numba.types.float64))
    jit_fun = numba.jit(sig, **numba_kwargs)(py_inter_law)
    cfunc_fun = numba.cfunc(sig)(jit_fun)
    
    return scipy.LowLevelCallable(cfunc_fun.ctypes)
    