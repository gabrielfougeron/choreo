"""
Benchmark of LowLevelCallable for Segment Quadrature
====================================================
"""

# %%
# Definition of benchmarked integrands

# sphinx_gallery_start_ignore
import os
import sys

try:
    __PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir,os.pardir))

    if ':' in __PROJECT_ROOT__:
        __PROJECT_ROOT__ = os.getcwd()

except (NameError, ValueError): 

    __PROJECT_ROOT__ = os.path.abspath(os.path.join(os.getcwd(),os.pardir,os.pardir))

sys.path.append(__PROJECT_ROOT__)

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import functools
import time
import matplotlib.pyplot as plt
import numpy as np
import math as m

import choreo 
import scipy

timings_folder = os.path.join(__PROJECT_ROOT__,'examples','generated_files')

if not(os.path.isdir(timings_folder)):
    os.makedirs(timings_folder)

basename_timings_filename = 'SegmQuad_bench_'

# ForceBenchmark = True
ForceBenchmark = False

ndim = 1
x_span = (0., 1.)
method = 'Gauss'
nsteps = 10
quad = choreo.scipy_plus.multiprec_tables.ComputeQuadrature(nsteps, method=method)

def test_from_scalar_fun(fun):
    
    return functools.partial(
        choreo.scipy_plus.SegmQuad.IntegrateOnSegment,
        fun,
        ndim,
        x_span,
        quad
    )


def py_fun(x):
    return np.sin(x)

nb_fun = choreo.scipy_plus.SegmQuad.nb_jit_double_double(py_fun)

def py_fun_array(x):
    res = np.empty((ndim))
    res[0] = np.sin(x)
    return res

nb_fun_pointer = choreo.scipy_plus.SegmQuad.nb_jit_array_double(py_fun_array)

def py_fun_inplace_pointer(x, res):
    res[0] = np.sin(x)

nb_fun_inplace_pointer = choreo.scipy_plus.SegmQuad.nb_jit_inplace_double_array(py_fun_inplace_pointer)


# sphinx_gallery_end_ignore

all_funs_scalar = {
    'py_fun' : test_from_scalar_fun(py_fun) ,
    'py_fun_array' : test_from_scalar_fun(py_fun_array) ,
    'nb_fun' : test_from_scalar_fun(nb_fun) ,
    'nb_fun_pointer' : test_from_scalar_fun(nb_fun_pointer) ,
    'nb_fun_inplace_pointer' : test_from_scalar_fun(nb_fun_inplace_pointer) ,
    'py_fun_in_pyx' : test_from_scalar_fun(choreo.scipy_plus.cython.test.single_py_fun) ,
    'cy_fun_pointer_LowLevel' : test_from_scalar_fun(scipy.LowLevelCallable.from_cython(choreo.scipy_plus.cython.test, "single_cy_fun_pointer")),
    'cy_fun_memoryview_LowLevel' : test_from_scalar_fun(scipy.LowLevelCallable.from_cython(choreo.scipy_plus.cython.test, "single_cy_fun_memoryview")),
    'cy_fun_oneval_LowLevel' : test_from_scalar_fun(scipy.LowLevelCallable.from_cython(choreo.scipy_plus.cython.test, "single_cy_fun_oneval")),
}

# sphinx_gallery_start_ignore

ndim_mul = choreo.scipy_plus.cython.test.mul_size_py

def test_from_vect_fun(fun):
    
    return functools.partial(
        choreo.scipy_plus.SegmQuad.IntegrateOnSegment,
        fun,
        ndim_mul,
        x_span,
        quad
    )

def mul_py_fun_array(x):
    res = np.empty((ndim_mul))
    for i in range(ndim_mul):
        val = (i+1) * x
        res[i] = np.sin(val)
    return res

mul_nb_fun_pointer = choreo.scipy_plus.SegmQuad.nb_jit_array_double(mul_py_fun_array)

def mul_py_fun_inplace_pointer(x, res):
    for i in range(ndim_mul):
        val = (i+1) * x
        res[i] = np.sin(val)

mul_nb_fun_inplace_pointer = choreo.scipy_plus.SegmQuad.nb_jit_inplace_double_array(mul_py_fun_inplace_pointer)

# sphinx_gallery_end_ignore

all_funs_vect = {
    'py_fun' : None, 
    'py_fun_array' : test_from_vect_fun(mul_py_fun_array) ,
    'nb_fun' : None ,
    'nb_fun_pointer' : test_from_vect_fun(mul_nb_fun_pointer) ,
    'nb_fun_inplace_pointer' : test_from_vect_fun(mul_nb_fun_inplace_pointer) ,
    'py_fun_in_pyx' : test_from_vect_fun(choreo.scipy_plus.cython.test.mul_py_fun),
    'cy_fun_pointer_LowLevel' : test_from_vect_fun(scipy.LowLevelCallable.from_cython(choreo.scipy_plus.cython.test, "mul_cy_fun_pointer")),
    'cy_fun_memoryview_LowLevel' : test_from_vect_fun(scipy.LowLevelCallable.from_cython(choreo.scipy_plus.cython.test, "mul_cy_fun_memoryview")),
}

all_benchs = {
    'Scalar function' : all_funs_scalar  ,
    f'Vector function of size {choreo.scipy_plus.cython.test.mul_size_py}' : all_funs_vect  ,
}

# sphinx_gallery_start_ignore

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

all_nint = np.array([2**i for i in range(12)])

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
        filename = timings_filename,
        ForceBenchmark = ForceBenchmark ,
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
plt.show()

    
# sphinx_gallery_end_ignore


