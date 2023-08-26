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

timings_folder = os.path.join(__PROJECT_ROOT__,'build')
basename = 'SegmQuad_bench'
timings_filename = os.path.join(timings_folder,basename+'.npy')



ndim = 10
x_span = (0., 1.)
method = 'Gauss'
nsteps = 10
quad = choreo.scipy_plus.SegmQuad.ComputeQuadrature(method, nsteps)


py_fun = functools.partial(
    choreo.scipy_plus.SegmQuad.IntegrateOnSegment,
    fun = choreo.scipy_plus.cython.SegmQuad.py_fun,
    ndim = ndim,
    x_span = x_span,
    quad = quad
)

cy_fun_LowLevel = functools.partial(
    choreo.scipy_plus.SegmQuad.IntegrateOnSegment,
    fun = scipy.LowLevelCallable.from_cython( choreo.scipy_plus.cython.SegmQuad, "cy_fun") ,
    ndim = ndim,
    x_span = x_span,
    quad = quad
)


for fun in [
    choreo.scipy_plus.cython.SegmQuad.py_fun,
    scipy.LowLevelCallable.from_cython( choreo.scipy_plus.cython.SegmQuad, "cy_fun"),
]:

    print(
        choreo.scipy_plus.SegmQuad.IntegrateOnSegment(
            fun = fun,
            ndim = ndim,
            x_span = x_span,
            quad = quad  ,
            nint = 10
        )
    )


exit()


all_funs = {
    'py_fun' : py_fun ,
    'cy_fun_LowLevel' : cy_fun_LowLevel,
}

all_funs_list = []
all_names_list = []
for name, fun in all_funs.items():
    
    all_funs_list.append(fun)
    all_names_list.append(name)

# all_maxval = np.array([2**i for i in range(18)])
all_maxval = np.array([2**i for i in range(3)])

def prepare_maxval(nint):
    return [(nint, 'nint')]

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
# 
# all_times = choreo.benchmark.run_benchmark(
#     all_maxval,
#     all_funs_list,
#     setup = prepare_maxval,
#     n_repeat = n_repeat,
#     time_per_test = 0.2,
#     # timings_filename = timings_filename,
# )
# 
# choreo.plot_benchmark(
#     all_times                               ,
#     all_maxval                              ,
#     all_funs_list                           ,
#     all_names_list                          ,
#     n_repeat = n_repeat                     ,
#     fig = fig                               ,
#     ax = ax                                 ,
#     title = 'LowLevelCallable_bench'        ,
# )
# 
# plt.tight_layout()
# plt.savefig('LowLevelCallable_bench.png')

# # 
# # 
for name, fun in all_funs.items():
    
        print()
        print(name)
        
        for maxval in all_maxval:
            
            tbeg = time.perf_counter()
            res = fun(maxval)
            tend = time.perf_counter()
            
            # print(maxval)
            print(res)
            # print(tend-tbeg)
            
            # print(res)
            # res = choreo.scipy_plus.cython.CallableInterface.add_values(cy_fun)
            # print(res)




# for maxval in all_maxval:
# 
#     print()
#     print(maxval)
#     
#             
#     for name, fun in all_funs.items():
#         baseline_res = fun(maxval)
#         break
#         
#     for name, fun in all_funs.items():
# 
#             
#             tbeg = time.perf_counter()
#             res = fun(maxval)
#             tend = time.perf_counter()
#             
#             # print(maxval)
#             # print(res)
#             print(abs(res-baseline_res))
#             # print(tend-tbeg)
#             
#             # print(res)
#             # res = choreo.scipy_plus.cython.CallableInterface.add_values(cy_fun)
#             # print(res)


