import os
import sys
__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(__PROJECT_ROOT__)

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import functools
import time
import matplotlib.pyplot as plt
import numpy as np
import math as m

import choreo 
import scipy

import pyquickbench

timings_folder = os.path.join(__PROJECT_ROOT__,'build')
basename = 'LowLevelCallable_bench'
timings_filename = os.path.join(timings_folder,basename+'.npy')

all_funs = {
    'cpdef_cy_fun' : functools.partial(choreo.scipy_plus.cython.CallableInterface.add_values,choreo.scipy_plus.cython.CallableInterface.cpdef_cy_fun),
    'cdef_cy_fun_LowLevel' : functools.partial(choreo.scipy_plus.cython.CallableInterface.add_values,scipy.LowLevelCallable.from_cython(choreo.scipy_plus.cython.CallableInterface, "cdef_cy_fun")),
    'conly_fixed' : choreo.scipy_plus.cython.CallableInterface.add_values_conly_fixed,
}

all_funs_list = []
all_names_list = []
for name, fun in all_funs.items():
    
    all_funs_list.append(fun)
    all_names_list.append(name)

all_maxval = np.array([2**i for i in range(18)])

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
)

all_times = pyquickbench.run_benchmark(
    all_maxval,
    all_funs_list,
    setup = prepare_maxval,
    filename = timings_filename,
)

pyquickbench.plot_benchmark(
    all_times                               ,
    all_maxval                              ,
    all_funs_list                           ,
    all_names_list                          ,
    fig = fig                               ,
    ax = ax                                 ,
    title = 'LowLevelCallable_bench'        ,
)

plt.tight_layout()
plt.savefig('LowLevelCallable_bench.png')

# # 
# # 
# for name, fun in all_funs.items():
#     
#         print()
#         print(name)
#         
#         for maxval in all_maxval:
#             
#             tbeg = time.perf_counter()
#             res = fun(maxval)
#             tend = time.perf_counter()
#             
#             # print(maxval)
#             print(res)
#             # print(tend-tbeg)
#             
#             # print(res)
#             # res = choreo.scipy_plus.cython.CallableInterface.add_values(cy_fun)
#             # print(res)

# 
# 

for maxval in all_maxval:

    print()
    print(maxval)
    
            
    for name, fun in all_funs.items():
        baseline_res = fun(maxval)
        break
        
    for name, fun in all_funs.items():

            
            tbeg = time.perf_counter()
            res = fun(maxval)
            tend = time.perf_counter()
            
            # print(maxval)
            # print(res)
            print(abs(res-baseline_res))
            # print(tend-tbeg)
            
            # print(res)
            # res = choreo.scipy_plus.cython.CallableInterface.add_values(cy_fun)
            # print(res)


