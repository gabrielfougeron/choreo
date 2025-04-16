"""
Benchmark of Error-Free Transforms for dot product
==================================================
"""

# %% 
# This benchmark compares the accuracy and efficiency of several dot product algorithms in floating point arithmetics.

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
from fractions import Fraction
import choreo
import pyquickbench

if ("--no-show" in sys.argv):
    plt.show = (lambda : None)

# ForceBenchmark = True
ForceBenchmark = False

if not(os.path.isdir(timings_folder)):
    os.makedirs(timings_folder)

# sphinx_gallery_end_ignore

def naive_dot(x, y):
    return choreo.segm.cython.eft_lib.naive_dot(x,y)

def naive_dot_ptr(x, y):
    return choreo.segm.cython.eft_lib.naive_dot_ptr(x,y)

def np_dot(x, y):
    return np.dot(x,y)

def m_sumprod(x, y):
    return m.sumprod(x, y)

def DotK_1(x, y):
    return choreo.segm.cython.eft_lib.DotK(x,y,1)

def DotK_2(x, y):
    return choreo.segm.cython.eft_lib.DotK(x,y,2)

def DotK_3(x, y):
    return choreo.segm.cython.eft_lib.DotK(x,y,3)

def DotK_4(x, y):
    return choreo.segm.cython.eft_lib.DotK(x,y,4)

def DotK_5(x, y):
    return choreo.segm.cython.eft_lib.DotK(x,y,5)

# sphinx_gallery_start_ignore
        
n = int(1e2)
def setup(alpha):
    x, y, ex = choreo.segm.test.GenDot(n, alpha)
    return {'x' : x, 'y' : y, 'ex' : ex}

def compute_error(f, x, y, ex):
    
    res = f(x, y)
    rel_err = 2 * abs(ex - Fraction(res)) / (abs(ex) + abs(res))
    
    return float(rel_err) + 1e-20

basename = 'dot_bench_accuracy'
error_filename = os.path.join(timings_folder, basename+'.npz')

all_alphas = {"alpha" : np.array([2**alpha for alpha in range(2,300)])}

all_funs = [
    naive_dot       ,
    np_dot          ,
    m_sumprod       ,
    DotK_1          ,
    DotK_2          ,
    DotK_3          ,
    DotK_4          ,
    DotK_5          ,
]

all_error_funs = { f.__name__ :  functools.partial(compute_error, f) for f in all_funs}

all_times = pyquickbench.run_benchmark(
    all_alphas                      ,
    all_error_funs                  ,
    setup = setup                   ,
    mode = "scalar_output"          ,
    filename = error_filename       ,
    allow_pickle = True             ,
    ForceBenchmark = ForceBenchmark ,
    StopOnExcept = True ,
)

pyquickbench.plot_benchmark(
    all_times       ,
    all_alphas      ,
    all_error_funs  ,
    show = True     ,
    title = "Relative error for increasing conditionning"   ,
    xlabel = "Approximate condition number" ,
    ylabel = "Relative error on dot product" ,
)
    
# sphinx_gallery_end_ignore

# %%

def prepare_x(n):
    x = np.random.random(n)
    y = np.random.random(n)
    return {'x': x, 'y': y}

# sphinx_gallery_start_ignore

basename = 'dot_bench_time'
timings_filename = os.path.join(timings_folder, basename+'.npz')

all_sizes = {"n" : np.array([2**n for n in range(30)])}

MonotonicAxes = ["n"]

all_times = pyquickbench.run_benchmark(
    all_sizes                       ,
    all_funs                        ,
    setup = prepare_x               ,
    MonotonicAxes = MonotonicAxes   ,
    filename = timings_filename     ,
    ForceBenchmark = ForceBenchmark ,
    StopOnExcept = True             ,
)

# sphinx_gallery_end_ignore

pyquickbench.plot_benchmark(
    all_times   ,
    all_sizes   ,
    all_funs    ,
    show = True ,
    title = "Measured CPU time"   ,
)

# %%

pyquickbench.plot_benchmark(
    all_times   ,
    all_sizes   ,
    all_funs    ,
    show = True ,
    relative_to_val = {pyquickbench.fun_ax_name: "naive_dot"},
    title = "Relative measured CPU time compared to naive_dot"   ,
)
