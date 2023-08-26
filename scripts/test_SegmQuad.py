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

ndim = 1
x_span = (0., 1.)
method = 'Gauss'
nsteps = 10
quad = choreo.scipy_plus.SegmQuad.ComputeQuadrature(method, nsteps)


py_fun = functools.partial(
    choreo.scipy_plus.SegmQuad.IntegrateOnSegment,
    choreo.scipy_plus.cython.test.py_fun,
    ndim,
    x_span,
    quad
)

cy_fun_pointer_LowLevel = functools.partial(
    choreo.scipy_plus.SegmQuad.IntegrateOnSegment,
    scipy.LowLevelCallable.from_cython( choreo.scipy_plus.cython.test, "cy_fun_pointer") ,
    ndim,
    x_span,
    quad
)

cy_fun_memoryview_LowLevel = functools.partial(
    choreo.scipy_plus.SegmQuad.IntegrateOnSegment,
    scipy.LowLevelCallable.from_cython( choreo.scipy_plus.cython.test, "cy_fun_memoryview") ,
    ndim,
    x_span,
    quad
)

cy_fun_oneval_LowLevel = functools.partial(
    choreo.scipy_plus.SegmQuad.IntegrateOnSegment,
    scipy.LowLevelCallable.from_cython( choreo.scipy_plus.cython.test, "cy_fun_oneval") ,
    ndim,
    x_span,
    quad
)



all_funs = {
    'cy_fun_pointer_LowLevel' : cy_fun_pointer_LowLevel,
    'cy_fun_memoryview_LowLevel' : cy_fun_memoryview_LowLevel,
    'cy_fun_oneval_LowLevel' : cy_fun_oneval_LowLevel,
    'py_fun' : py_fun ,
}

all_funs_list = []
all_names_list = []
for name, fun in all_funs.items():
    
    all_funs_list.append(fun)
    all_names_list.append(name)

all_nint = np.array([2**i for i in range(10)])

def setup(nint):
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

all_times = choreo.benchmark.run_benchmark(
    all_nint,
    all_funs_list,
    setup = setup,
    n_repeat = n_repeat,
    time_per_test = 0.2,
    # timings_filename = timings_filename,
)

choreo.plot_benchmark(
    all_times                               ,
    all_nint                              ,
    all_funs_list                           ,
    all_names_list                          ,
    n_repeat = n_repeat                     ,
    fig = fig                               ,
    ax = ax                                 ,
    title = 'SegmQuad_bench'        ,
)

plt.tight_layout()
plt.savefig('SegmQuad_bench.png')

# # 
# # # 
# for name, fun in all_funs.items():
#     
#         print()
#         print(name)
#         
#         for nint in all_nint:
#             
#             tbeg = time.perf_counter()
#             res = fun(nint)
#             tend = time.perf_counter()
#             
#             print(res)



# for nint in all_nint:
# 
#     print()
#     print(nint)
#     
#             
#     for name, fun in all_funs.items():
#         baseline_res = fun(nint)
#         break
#         
#     for name, fun in all_funs.items():
# 
#             
#             tbeg = time.perf_counter()
#             res = fun(nint)
#             tend = time.perf_counter()
#             
#             # print(nint)
#             # print(res)
#             print(abs(res-baseline_res))
#             # print(tend-tbeg)
#             
#             # print(res)
#             # res = choreo.scipy_plus.cython.CallableInterface.add_values(cy_fun)
#             # print(res)


