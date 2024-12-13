import numba
import functools
import ctypes

numba_kwargs = {
    'nopython':True     ,
    # 'cache':True        ,
    'fastmath':True     ,
    'nogil':True        ,
}

sig = numba.types.float64(numba.types.float64)

def py_fun_2(a,b):
    return a+b

py_fun_1 = functools.partial(py_fun_2, b=1.)
py_fun_1.__name__ = "py_fun_1"


print(py_fun_1(2.))

# cfunc_fun = numba.cfunc(sig, **numba_kwargs)(py_fun_1)


print(dir(ctypes.c_void_p))