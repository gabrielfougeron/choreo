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

timings_folder = os.path.join(__PROJECT_ROOT__,'examples','generated_files')

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

if not(os.path.isdir(timings_folder)):
    os.makedirs(timings_folder)

basename_timings_filename = 'ODE_ivp_lowlevel_bench'

# ForceBenchmark = True
ForceBenchmark = False

ndim_mul = choreo.segm.cython.test.mul_size_py

t_span = (0., 1.)

x0 = np.random.random(ndim_mul)
v0 = np.random.random(ndim_mul)

nsteps = 8
rk_method = choreo.segm.multiprec_tables.ComputeImplicitRKTable(nsteps)

def test_from_fun(fun):
    
    return lambda nint : choreo.segm.ODE.SymplecticIVP(
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

# mul_nb_fun_pointer = choreo.segm.quad.nb_jit_array_double(mul_py_fun_array)

def mul_py_fun_inplace_pointer(t,x, res):
    for i in range(ndim_mul):
        val = (i+1) * x[i]
        res[i] = np.sin(t*val)

mul_nb_fun_inplace_pointer = choreo.segm.ODE.nb_jit_c_fun_pointer(mul_py_fun_inplace_pointer)

# sphinx_gallery_end_ignore

all_funs = {
    'mul_py_fun_array' : test_from_fun(mul_py_fun_array),
    'mul_nb_fun_inplace_pointer' : test_from_fun(mul_nb_fun_inplace_pointer),
    'py_fun_in_pyx' : test_from_fun(choreo.segm.cython.test.mul_py_fun_tx) ,
    'cy_fun_pointer_LowLevel' : test_from_fun(scipy.LowLevelCallable.from_cython(choreo.segm.cython.test, "mul_cy_fun_pointer_tx")),
    'cy_fun_memoryview_LowLevel' : test_from_fun(scipy.LowLevelCallable.from_cython(choreo.segm.cython.test, "mul_cy_fun_memoryview_tx")),
}

# sphinx_gallery_start_ignore

all_nint = np.array([2**i for i in range(8)])

def setup(nint):
    return {'nint': nint}

timings_filename = os.path.join(timings_folder,basename_timings_filename+'.npz')

n_repeat = 10

all_times = pyquickbench.run_benchmark(
    all_nint                        ,
    all_funs                        ,
    n_repeat = n_repeat             ,
    setup = setup                   ,
    filename = timings_filename     ,
    ForceBenchmark = ForceBenchmark ,
    StopOnExcept = True             ,
)

pyquickbench.plot_benchmark(
    all_times           ,
    all_nint            ,
    all_funs            ,
    show = True         ,
)

# sphinx_gallery_end_ignore

# %%
# 

relative_to_val = {pyquickbench.fun_ax_name:'cy_fun_memoryview_LowLevel'}

pyquickbench.plot_benchmark(
    all_times                           ,
    all_nint                            ,
    all_funs                            ,
    show = True                         ,
    relative_to_val = relative_to_val   ,
    title = 'Timings relative to cy_fun_memoryview_LowLevel'      ,
)

