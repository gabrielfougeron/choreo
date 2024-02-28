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

    @numba.cfunc(numba.types.void(numba.types.float64, numba.types.CPointer(numba.types.float64)))
    @numba.jit(**numba_kwargs)
    def pot_fun(xsq, res):
        
        a = xsq ** nm2
        b = xsq*a

        res[0] = -xsq*b
        res[1] = -n*b
        res[2] = mnnm1*a
        
    return scipy.LowLevelCallable(pot_fun.ctypes)

