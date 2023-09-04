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
basename_timings_filename = 'SegmQuad_bench'

ndim = 1
x_span = (0., 1.)
method = 'Gauss'
nsteps = 10
quad = choreo.scipy_plus.multiprec_tables.ComputeQuadrature(nsteps, method=method)

single_py_fun = functools.partial(
    choreo.scipy_plus.SegmQuad.IntegrateOnSegment,
    choreo.scipy_plus.cython.test.single_py_fun,
    ndim,
    x_span,
    quad
)

single_cy_fun_pointer_LowLevel = functools.partial(
    choreo.scipy_plus.SegmQuad.IntegrateOnSegment,
    scipy.LowLevelCallable.from_cython( choreo.scipy_plus.cython.test, "single_cy_fun_pointer") ,
    ndim,
    x_span,
    quad
)

single_cy_fun_memoryview_LowLevel = functools.partial(
    choreo.scipy_plus.SegmQuad.IntegrateOnSegment,
    scipy.LowLevelCallable.from_cython( choreo.scipy_plus.cython.test, "single_cy_fun_memoryview") ,
    ndim,
    x_span,
    quad
)

single_cy_fun_oneval_LowLevel = functools.partial(
    choreo.scipy_plus.SegmQuad.IntegrateOnSegment,
    scipy.LowLevelCallable.from_cython( choreo.scipy_plus.cython.test, "single_cy_fun_oneval") ,
    ndim,
    x_span,
    quad
)

all_funs_1 = {
    'py_fun' : single_py_fun ,
    'cy_fun_pointer_LowLevel' : single_cy_fun_pointer_LowLevel,
    'cy_fun_memoryview_LowLevel' : single_cy_fun_memoryview_LowLevel,
    'cy_fun_oneval_LowLevel' : single_cy_fun_oneval_LowLevel,
}

ndim = choreo.scipy_plus.cython.test.mul_size_py

mul_py_fun = functools.partial(
    choreo.scipy_plus.SegmQuad.IntegrateOnSegment,
    choreo.scipy_plus.cython.test.mul_py_fun,
    ndim,
    x_span,
    quad
)

mul_cy_fun_pointer_LowLevel = functools.partial(
    choreo.scipy_plus.SegmQuad.IntegrateOnSegment,
    scipy.LowLevelCallable.from_cython( choreo.scipy_plus.cython.test, "mul_cy_fun_pointer") ,
    ndim,
    x_span,
    quad
)

mul_cy_fun_memoryview_LowLevel = functools.partial(
    choreo.scipy_plus.SegmQuad.IntegrateOnSegment,
    scipy.LowLevelCallable.from_cython( choreo.scipy_plus.cython.test, "mul_cy_fun_memoryview") ,
    ndim,
    x_span,
    quad
)

all_funs_2 = {
    'py_fun' : mul_py_fun ,
    'cy_fun_pointer_LowLevel' : mul_cy_fun_pointer_LowLevel,
    'cy_fun_memoryview_LowLevel' : mul_cy_fun_memoryview_LowLevel,
}

all_benchs = {
    'Scalar function' : all_funs_1  ,
    f'Vector function of size {ndim}' : all_funs_2  ,
}

n_bench = len(all_benchs)

dpi = 150
figsize = (1600/dpi, n_bench * 800 / dpi)

fig, axs = plt.subplots(
    nrows = n_bench,
    ncols = 1,
    sharex = True,
    sharey = True,
    figsize = figsize,
    dpi = dpi   ,
    squeeze = False,
)

i_bench = -1

all_nint = np.array([2**i for i in range(18)])

def setup(nint):
    return [(nint, 'nint')]

n_repeat = 1

for bench_name, all_funs in all_benchs.items():

    i_bench += 1

    timings_filename = os.path.join(timings_folder,basename_timings_filename+bench_name.replace(' ','_')+'.npy')

    all_times = choreo.benchmark.run_benchmark(
        all_nint,
        all_funs,
        setup = setup,
        n_repeat = n_repeat,
        time_per_test = 0.2,
        timings_filename = timings_filename,
    )

    choreo.plot_benchmark(
        all_times                               ,
        all_nint                                ,
        all_funs                                ,
        n_repeat = n_repeat                     ,
        fig = fig                               ,
        ax = axs[i_bench,0]                     ,
        title = bench_name                      ,
    )


plt.tight_layout()
plt.savefig('SegmQuad_bench.png')
