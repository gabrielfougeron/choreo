"""
Matmul stuff
============
"""

# %% 
# This benchmark compares accuracy and efficiency of several summation algorithms in floating point arithmetics

# sphinx_gallery_start_ignore

import os
import sys


os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TBB_NUM_THREADS'] = '1'


try:
    __PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir,os.pardir))

    if ':' in __PROJECT_ROOT__:
        __PROJECT_ROOT__ = os.getcwd()

except (NameError, ValueError): 

    __PROJECT_ROOT__ = os.path.abspath(os.path.join(os.getcwd(),os.pardir,os.pardir))

sys.path.append(__PROJECT_ROOT__)

import functools
import matplotlib.pyplot as plt
import numpy as np
import math as m
import scipy
import numba as nb
import choreo
import copy
import itertools
import time

numba_opt_dict = {
    'nopython':True     ,
    'cache':True        ,
    'fastmath':True     ,
    'nogil':True        ,
}

import pyquickbench

if ("--no-show" in sys.argv):
    plt.show = (lambda : None)

timings_folder = os.path.join(__PROJECT_ROOT__,'examples','generated_files_time_consuming')

if not(os.path.isdir(timings_folder)):
    os.makedirs(timings_folder)

# sphinx_gallery_end_ignore


def numpy_matmul_complex(a, b, c):
    c[:,:] = np.matmul(a, b).real
    
def numpy_matmul_real(a, b, c):
    a = a.view(dtype=np.float64).reshape(  a.shape[0],2*a.shape[1])
    b = b.view(dtype=np.float64).reshape(2*b.shape[0],  b.shape[1])
    np.matmul(a, b, out=c)
    
    
    
dtypes_dict = {
    "float64" : np.float64,
}
dtypes_complex_dict = {
    "float64" : np.complex128,
}

def setup_abc(PR, Q, real_dtype):
    
    P = PR[0]
    R = PR[1]
    
    a = np.random.random((P,R)).astype(dtype=dtypes_dict[real_dtype]) + 1j*np.random.random((P,R)).astype(dtype=dtypes_dict[real_dtype])
    b = np.random.random((R,Q)).astype(dtype=dtypes_dict[real_dtype]) + 1j*np.random.random((R,Q)).astype(dtype=dtypes_dict[real_dtype])
    c = np.zeros((P,Q),dtype=dtypes_dict[real_dtype])
    return {'a':a, 'b':b, 'c':c}

max_exp = 20
# max_exp = 10

all_args = {
    "PR" :  [[2,2],[2,4],[3,3],[3,6]]                               ,
    "Q" :  [2 ** k for k in range(max_exp)]                                 ,
    "real_dtype": ["float64"]    ,
}

all_funs = [
    numpy_matmul_complex    ,
    numpy_matmul_real       ,
    choreo.cython.test_blis.blas_matmul_real,
    choreo.cython.test_blis.blis_matmul_real,
    choreo.cython.test_blis.blas_matmul_real_copy,
]

n_repeat = 100

MonotonicAxes = ["Q"]


# # %%
# 
basename = 'matmul_timingsRE'
filename = os.path.join(timings_folder,basename+'.npz')

all_timings = pyquickbench.run_benchmark(
    all_args                ,
    all_funs                ,
    setup = setup_abc       ,
    filename = filename     ,
    StopOnExcept = True     ,
    ShowProgress = True     ,
    n_repeat = n_repeat     ,
    MonotonicAxes = MonotonicAxes,
    # ForceBenchmark = True
)

plot_intent = {
    "PR" : 'subplot_grid_y'   ,
    "Q" : 'points'              ,
    "real_dtype": 'curve_linestyle'  ,
    pyquickbench.fun_ax_name :  'curve_color'             ,
    pyquickbench.repeat_ax_name :  'reduction_avg'             ,
}

pyquickbench.plot_benchmark(
    all_timings                             ,
    all_args                                ,
    all_funs                                ,
    plot_intent = plot_intent               ,
    show = True                             ,
)

# # %%

relative_to_val = {
    "real_dtype": 'float64'  ,
    pyquickbench.fun_ax_name :  'blas_matmul_real'   ,
}


pyquickbench.plot_benchmark(
    all_timings                             ,
    all_args                                ,
    all_funs                                ,
    plot_intent = plot_intent               ,
    show = True                             ,
    relative_to_val = relative_to_val       ,
)


