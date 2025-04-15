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
from fractions import Fraction
import choreo
import pyquickbench

if ("--no-show" in sys.argv):
    plt.show = (lambda : None)

ForceBenchmark = True
# ForceBenchmark = False

if not(os.path.isdir(timings_folder)):
    os.makedirs(timings_folder)

# sphinx_gallery_end_ignore

def naive_sum(x):
    return choreo.segm.cython.eft_lib.SumK(x,0)

def np_sum(x):
    return np.sum(x)

def m_fsum(x):
    return m.fsum(x)

def SumK_1(x):
    return choreo.segm.cython.eft_lib.SumK(x,1)

def SumK_2(x):
    return choreo.segm.cython.eft_lib.SumK(x,2)

def SumK_3(x):
    return choreo.segm.cython.eft_lib.SumK(x,3)

def SumK_4(x):
    return choreo.segm.cython.eft_lib.SumK(x,4)

def SumK_5(x):
    return choreo.segm.cython.eft_lib.SumK(x,5)

# sphinx_gallery_start_ignore
        
n = int(1e2)
def setup(alpha):

    x, ex = choreo.segm.test.GenSum(n, alpha)

    return {'x' : x, 'ex' : ex}

def compute_error(f, x, ex):
    
    res = f(x)
    rel_err = 2 * abs(ex - Fraction(res)) / (abs(ex) + abs(res))
    
    return float(rel_err) + 1e-20

basename = 'sum_bench_accuracy'
error_filename = os.path.join(timings_folder, basename+'.npz')

all_alphas = {"alpha" : np.array([2**alpha for alpha in range(2,500)])}

all_funs = [
    naive_sum   ,
    np_sum      ,
    m_fsum      ,
    SumK_1      ,
    SumK_2      ,
    SumK_3      ,
    SumK_4      ,
    SumK_5      ,
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
    ylabel = "Relative error on sum" ,
)
    
# sphinx_gallery_end_ignore


# %%

def prepare_x(n):
    x = np.random.random(n)
    return {'x': x}

# sphinx_gallery_start_ignore

basename = 'sum_bench_time'
timings_filename = os.path.join(timings_folder, basename+'.npz')

all_sizes = {"n" : np.array([2**n for n in range(30)])}

MonotonicAxes = ["n"]

# all_times = pyquickbench.run_benchmark(
#     all_sizes                       ,
#     all_funs                        ,
#     setup = prepare_x               ,
#     MonotonicAxes = MonotonicAxes   ,
#     filename = timings_filename     ,
#     ForceBenchmark = ForceBenchmark ,
#     StopOnExcept = True             ,
# )
# 
# for relative_to_val in [
#     None    ,
#     {
#         pyquickbench.fun_ax_name: "naive_sum"    ,
#     }
# ]:
# 
#     pyquickbench.plot_benchmark(
#         all_times   ,
#         all_sizes   ,
#         all_funs    ,
#         show = True ,
#         relative_to_val = relative_to_val      ,
#     )

# sphinx_gallery_end_ignore
