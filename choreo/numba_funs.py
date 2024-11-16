import numba
import multiprocessing
import scipy
import types
import inspect
import tempfile

max_num_threads = multiprocessing.cpu_count()

numba_kwargs = {
    'nopython':True     ,
    'cache':True        ,
    'fastmath':True     ,
    'nogil':True        ,
}

def jit_inter_law(py_inter_law):
    
    sig = numba.types.void(numba.types.CPointer(numba.types.float64), numba.types.float64, numba.types.CPointer(numba.types.float64))
    jit_fun = numba.jit(sig, **numba_kwargs)(py_inter_law)
    cfunc_fun = numba.cfunc(sig)(jit_fun)
    
    return scipy.LowLevelCallable(cfunc_fun.ctypes)
    
def jit_inter_law_str(py_inter_law_str):
    
    tmpdir = tempfile.gettempdir()
    code_obj = compile(py_inter_law_str, tmpdir, 'exec')
    module_obj = types.ModuleType("modname")
    exec(code_obj, module_obj.__dict__)
    inter_law_py = module_obj.inter_law
    if not inspect.isfunction(inter_law_py):
        raise ValueError('Could not compile provided string to a Python function named "inter_law"')
    
    return jit_inter_law(inter_law_py)
    