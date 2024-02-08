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

def rfft_l(l, m, r):
    scipy.fft.rfft(l, axis=0)
def rfft_m(l, m, r):
    scipy.fft.rfft(m, axis=1)
def rfft_r(l, m, r):
    scipy.fft.rfft(r, axis=2)
    
def setup_lmr(fft_axis_len, other_axes_len):

    l = np.random.random((fft_axis_len,other_axes_len[0],other_axes_len[1]))
    m = np.random.random((other_axes_len[0],fft_axis_len,other_axes_len[1]))
    r = np.random.random((other_axes_len[0],other_axes_len[1],fft_axis_len)) 
    
    return {'l':l, 'm':m, 'r':r}

max_exp = 20
# max_exp = 10

all_args = {
    "fft_axis_len" : [3*5*7*2 ** k for k in range(max_exp)]     ,
    "other_axes_len" : [[2,4]]                            ,
}

all_funs = [
    rfft_l,
    rfft_m,
    rfft_r,
]

n_repeat = 10

MonotonicAxes = ["fft_axis_len"]


# # %%
# 
basename = 'fft_axis_timings'
filename = os.path.join(timings_folder,basename+'.npz')

all_timings = pyquickbench.run_benchmark(
    all_args                ,
    all_funs                ,
    setup = setup_lmr       ,
    filename = filename     ,
    StopOnExcept = True     ,
    ShowProgress = True     ,
    n_repeat = n_repeat     ,
    MonotonicAxes = MonotonicAxes,
    # ForceBenchmark = True
)

plot_intent = {
    "other_axes_len" : 'single_value'    ,
    "fft_axis_len" : 'points'           ,
    pyquickbench.fun_ax_name : 'curve_color',
    pyquickbench.repeat_ax_name : 'reduction_avg',
}


pyquickbench.plot_benchmark(
    all_timings                             ,
    all_args                                ,
    all_funs                                ,
    plot_intent = plot_intent               ,
    show = True                             ,
)

# %%

relative_to_val = {
    pyquickbench.fun_ax_name :  'rfft_l'   ,
}


pyquickbench.plot_benchmark(
    all_timings                             ,
    all_args                                ,
    all_funs                                ,
    plot_intent = plot_intent               ,
    show = True                             ,
    relative_to_val = relative_to_val       ,
)

