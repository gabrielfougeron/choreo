import os
import sys
__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(__PROJECT_ROOT__)

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import functools
import matplotlib.pyplot as plt
import numpy as np

import choreo 
import scipy

timings_folder = os.path.join(__PROJECT_ROOT__,'build')
basename = 'LowLevelCallable_bench'
timings_filename = os.path.join(timings_folder,basename+'.npy')

def py_fun(x):
    return x

all_funs = {
    'py_fun' : py_fun,
    'def_cy_fun' : choreo.scipy_plus.cython.CallableInterface.def_cy_fun,
    'cpdef_cy_fun' : choreo.scipy_plus.cython.CallableInterface.cpdef_cy_fun,
    'cdef_cy_fun_LowLevel' : scipy.LowLevelCallable.from_cython(choreo.scipy_plus.cython.CallableInterface, "cdef_cy_fun"),
    'cpdef_cy_fun_LowLevel' : scipy.LowLevelCallable.from_cython(choreo.scipy_plus.cython.CallableInterface, "cpdef_cy_fun"),
    # 'def_cy_fun_LowLevel' : scipy.LowLevelCallable.from_cython(choreo.scipy_plus.cython.CallableInterface, "def_cy_fun"), # not in __pyx_capi__
}



all_funs_list = []
all_names_list = []
for name, fun in all_funs.items():
    
    all_funs_list.append(functools.partial(choreo.scipy_plus.cython.CallableInterface.add_values, fun))
    all_names_list.append(name)

# 
# def python_add(maxval):
#     return choreo.scipy_plus.cython.CallableInterface.add_values(py_fun, maxval)
# 
# def cython_add(maxval):
#     return choreo.scipy_plus.cython.CallableInterface.add_values(cy_fun, maxval)


# all_funs_oneaarg = [
#     python_add,
#     cython_add,
# ]

all_maxval = np.array([2**i for i in range(10)])


def prepare_maxval(maxval):
    return [(maxval, 'maxval')]

dpi = 150
figsize = (1600/dpi, 1 * 800 / dpi)

fig, ax = plt.subplots(
    nrows = 1,
    ncols = 1,
    sharex = True,
    sharey = True,
    figsize = figsize,
    dpi = dpi   ,
    # squeeze = False,
)

n_repeat = 1

all_times = choreo.benchmark.run_benchmark(
    all_maxval,
    all_funs_list,
    setup = prepare_maxval,
    n_repeat = n_repeat,
    time_per_test = 0.2,
    timings_filename = timings_filename,
)

choreo.plot_benchmark(
    all_times                               ,
    all_maxval                              ,
    all_funs_list                           ,
    all_names_list                          ,
    n_repeat = n_repeat                     ,
    fig = fig                               ,
    ax = ax                                 ,
    title = 'LowLevelCallable_bench'        ,
)

plt.tight_layout()
plt.savefig('LowLevelCallable_bench.png')

# 
# 
# for name, fun in all_funs.items():
#     
#         print()
#         print(name)
#         
#         for maxval in all_maxval:
#             
#             tbeg = time.perf_counter()
#             res = choreo.scipy_plus.cython.CallableInterface.add_values(fun, maxval)
#             tend = time.perf_counter()
#             
#             print(maxval, tend-tbeg)
#             
#             # print(res)
#             # res = choreo.scipy_plus.cython.CallableInterface.add_values(cy_fun)
#             # print(res)
# 
# 
