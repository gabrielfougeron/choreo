"""
Benchmark of Error-Free Transforms for summation
================================================
"""

# %% 
# This benchmark compares accuracy and efficiency of several summation algorithms in floating point arithmetics

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
os.environ['OMP_NUM_THREADS'] = '1'

import functools
import matplotlib.pyplot as plt
import numpy as np
import math as m
import scipy
import choreo
import pyquickbench

if ("--no-show" in sys.argv):
    plt.show = (lambda : None)

# ForceBenchmark = True
ForceBenchmark = False

if not(os.path.isdir(timings_folder)):
    os.makedirs(timings_folder)

# sphinx_gallery_end_ignore

def naive_sum(x):
    return choreo.scipy_plus.cython.eft_lib.FastSumK(x,0)

def builtin_sum(x):
    return sum(x)

def np_sum(x):
    return np.sum(x)

def m_fsum(x):
    return m.fsum(x)

def SumK_1(x):
    return choreo.scipy_plus.cython.eft_lib.SumK(x,1)

def SumK_2(x):
    return choreo.scipy_plus.cython.eft_lib.SumK(x,2)

def SumK_3(x):
    return choreo.scipy_plus.cython.eft_lib.SumK(x,3)

def FastSumK_1(x):
    return choreo.scipy_plus.cython.eft_lib.FastSumK(x,1)

def FastSumK_2(x):
    return choreo.scipy_plus.cython.eft_lib.FastSumK(x,2)

def FastSumK_3(x):
    return choreo.scipy_plus.cython.eft_lib.FastSumK(x,3)

# sphinx_gallery_start_ignore

def setup(alpha):
        
    n = int(1e5)
    x = np.zeros((n),dtype=np.float64)
    choreo.segm.cython.test.inplace_taylor_poly(x, -alpha)
    
    return {'x' : x}

@functools.cache
def exact_sum(alpha):
    y = setup(alpha)['x']
    return m.fsum(y)
    
def compute_error(f, x):
    
    ex_res =  exact_sum(-x[1])
    res = f(x)
    
    rel_err = abs(ex_res - res) / abs(ex_res)
    
    return rel_err + 1e-40

basename = 'sum_bench_accuracy'
error_filename = os.path.join(timings_folder,basename+'.npz')

all_alphas = {"alpha" : np.array([float(alpha) for alpha in range(200)])}

all_funs = [
    naive_sum   ,
    builtin_sum ,
    np_sum      ,
    m_fsum      ,
    SumK_1      ,
    SumK_2      ,
    SumK_3      ,
    FastSumK_1  ,
    FastSumK_2  ,
    FastSumK_3  ,
]

all_error_funs = { f.__name__ :  functools.partial(compute_error, f) for f in all_funs if f is not m_fsum}

all_times = pyquickbench.run_benchmark(
    all_alphas                      ,
    all_error_funs                  ,
    setup = setup                   ,
    mode = "scalar_output"          ,
    filename = error_filename       ,
    ForceBenchmark = ForceBenchmark ,
)

pyquickbench.plot_benchmark(
    all_times       ,
    all_alphas      ,
    all_error_funs  ,
    show = True     ,
    title = "Relative error for increasing conditionning"   ,
)
    
# sphinx_gallery_end_ignore


# %%

def prepare_x(n):
    x = np.random.random(n)
    return {'x': x}

# sphinx_gallery_start_ignore

basename = 'sum_bench_time'
timings_filename = os.path.join(timings_folder,basename+'.npz')

all_sizes = {"n" : np.array([2**n for n in range(21)])}

all_times = pyquickbench.run_benchmark(
    all_sizes                       ,
    all_funs                        ,
    setup = prepare_x               ,
    filename = timings_filename     ,
    ForceBenchmark = ForceBenchmark ,
)

pyquickbench.plot_benchmark(
    all_times   ,
    all_sizes   ,
    all_funs    ,
    show = True ,
    title = "Time (s) as a function of array size"   ,
)

# sphinx_gallery_end_ignore
