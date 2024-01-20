"""
Benchmark of LowLevelCallable for ODE IVP
=========================================
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

import functools
import time
import matplotlib.pyplot as plt
import numpy as np
import math as m

import choreo 
import scipy
import numba
import pyquickbench

if ("--no-show" in sys.argv):
    plt.show = (lambda : None)

timings_folder = os.path.join(__PROJECT_ROOT__,'examples','generated_files')

if not(os.path.isdir(timings_folder)):
    os.makedirs(timings_folder)

basename_timings_filename = 'ODE_ivp_lowlevel_bench_'

# ForceBenchmark = True
ForceBenchmark = False


ndim_mul = choreo.scipy_plus.cython.test.mul_size_py

t_span = (0., 1.)

x0 = np.random.random(ndim_mul)
v0 = np.random.random(ndim_mul)

nsteps = 8
rk_method = choreo.scipy_plus.multiprec_tables.ComputeImplicitRKTable_Gauss(nsteps)

def test_from_fun(fun):
    
    return lambda nint : choreo.scipy_plus.ODE.SymplecticIVP(
        fun             ,
        fun             ,
        t_span          ,
        x0              ,
        v0              ,
        rk_method       ,
        nint = nint     ,
    )



def mul_py_fun_array(t,x):
    res = np.empty((ndim_mul))
    for i in range(ndim_mul):
        val = (i+1) * x[i]
        res[i] = np.sin(t*val)
    return res

# mul_nb_fun_pointer = choreo.scipy_plus.SegmQuad.nb_jit_array_double(mul_py_fun_array)

def mul_py_fun_inplace_pointer(t,x, res):
    for i in range(ndim_mul):
        val = (i+1) * x[i]
        res[i] = np.sin(t*val)

mul_nb_fun_inplace_pointer = choreo.scipy_plus.ODE.nb_jit_inplace_double_array(mul_py_fun_inplace_pointer)


# sphinx_gallery_end_ignore

all_funs_vect = {
    'mul_py_fun_array' : test_from_fun(mul_py_fun_array),
    'mul_nb_fun_inplace_pointer' : test_from_fun(mul_nb_fun_inplace_pointer),
    'py_fun_in_pyx' : test_from_fun(choreo.scipy_plus.cython.test.mul_py_fun_tx) ,
    'cy_fun_pointer_LowLevel' : test_from_fun(scipy.LowLevelCallable.from_cython(choreo.scipy_plus.cython.test, "mul_cy_fun_pointer_tx")),
    'cy_fun_memoryview_LowLevel' : test_from_fun(scipy.LowLevelCallable.from_cython(choreo.scipy_plus.cython.test, "mul_cy_fun_memoryview_tx")),
}

all_benchs = {
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

# all_nint = np.array([2**i for i in range(12)])
all_nint = np.array([2**i for i in range(8)])

def setup(nint):
    return {'nint': nint}

for bench_name, all_funs in all_benchs.items():

    i_bench += 1

    timings_filename = os.path.join(timings_folder,basename_timings_filename+bench_name.replace(' ','_')+'.npy')

    all_times = pyquickbench.run_benchmark(
        all_nint,
        all_funs,
        setup = setup,
        filename = timings_filename,
        ForceBenchmark = ForceBenchmark ,
        StopOnExcept = True,
    )

    pyquickbench.plot_benchmark(
        all_times                               ,
        all_nint                                ,
        all_funs                                ,
        fig = fig                               ,
        ax = axs[i_bench,0]                     ,
        title = bench_name                      ,
    )
    
plt.tight_layout()
plt.show()

    
# sphinx_gallery_end_ignore
