import os
import numpy as np
import numba
from numba.core.typing import cffi_utils
import cffi
import multiprocessing
import scipy

max_num_threads = multiprocessing.cpu_count()

numba_kwargs = {
    'nopython':True     ,
    'cache':True        ,
    'fastmath':True     ,
    'nogil':True        ,
}
# 
# src = """
# typedef struct {
#   double pot;
#   double potp;
#   double potpp;
# } pot_t;
# 
# 
# typedef void (*inter_law_fun_type)(double, pot_t*) ;
# """
# 
# ffi = cffi.FFI()
# ffi.cdef(src)
# 
# law_fun_sig = cffi_utils.map_type(ffi.typeof('inter_law_fun_type'), use_record_dtype=True)
# 
# def power_law_pot(n):
# 
#     nm2 = n-2
#     mnnm1 = -n*(n-1)
# 
#     @numba.jit(**numba_kwargs)
#     @numba.cfunc(law_fun_sig)
#     def power_law_pot(xsq, pot):
#         a = xsq ** nm2
#         b = xsq*a
# 
#         pot.pot = -xsq*b
#         pot.potp = -n*b
#         pot.potpp = mnnm1*a
#         
#     return scipy.LowLevelCallable(power_law_pot.ctypes)

src = """
typedef struct {
  double a;
} a_t;


typedef  double (*fun_t)(double) ;
"""

ffi = cffi.FFI()
ffi.cdef(src)

sig = cffi_utils.map_type(ffi.typeof('fun_t'), use_record_dtype=True)

def param_fun(n):

    @numba.cfunc(sig)
    @numba.jit(signature=sig, **numba_kwargs)
    def param_fun(x):
        return n*x

        
    return scipy.LowLevelCallable(param_fun.ctypes)

