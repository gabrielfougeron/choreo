import cffi
import numba
from numba.core.typing import cffi_utils
import scipy

numba_kwargs = {
    'nopython':True     ,
    'cache':True        ,
    'fastmath':True     ,
    'nogil':True        ,
}


src = """
typedef  double (*fun_t)(double) ;
"""

ffi = cffi.FFI()
ffi.cdef(src)

sig = cffi_utils.map_type(ffi.typeof('fun_t'), use_record_dtype=True)

def param_fun(n):

    @numba.cfunc(sig)
    @numba.jit(signature=sig, **numba_kwargs)
    def fun(x):
        return n*x

    return scipy.LowLevelCallable(fun.ctypes)

toto = param_fun(0.5)



src = """
typedef struct {
  double a;
} a_t;

typedef a_t (*fun_t)(double) ;
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

toto = param_fun(0.5)