"""
Benchmark of LowLevelCallable for ODE IVP
=========================================
"""

# %%
# Definition of benchmarked integrands
# 
# Talk about the types of the arguments
# 
# * ``double (double)``
# * ``double (double, void *)``
# * ``void (double, double *)``
# * ``void (double, double *, void *)``
# * ``void (double, __Pyx_memviewslice)``
# * ``void (double, __Pyx_memviewslice, void *)``
# 

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

import matplotlib.pyplot as plt
import numpy as np

import choreo 
import scipy
import math as m
import numba
import pyquickbench

if ("--no-show" in sys.argv):
    plt.show = (lambda : None)

if not(os.path.isdir(timings_folder)):
    os.makedirs(timings_folder)

basename_timings_filename = 'quad_lowlevel_bench'

# sphinx_gallery_end_ignore

# ForceBenchmark = True
ForceBenchmark = False

t_span = (0., np.pi/2)

nsteps = 10
quad_method = choreo.segm.multiprec_tables.ComputeQuadrature(nsteps, method="Gauss")

def test_from_fun(fun):
    
    return lambda nint : choreo.segm.quad.IntegrateOnSegment(
        fun                     ,
        ndim = 1                ,
        x_span = (0.,np.pi/2)   ,
        quad = quad_method      ,
        nint = nint             ,
    )

def Wallis7_py(x):
    res = np.empty((1), dtype=np.float64)
    res[0] = np.sin(x)
    return res

def Wallis7_c_double(x):
    return np.sin(x)

Wallis7_c_double_jit = choreo.segm.quad.nb_jit_double_double(Wallis7_c_double)

def Wallis7_c_inplace_array(x, res):
    res[0] = np.sin(x)
    
Wallis7_c_inplace_array_jit = choreo.segm.quad.nb_jit_inplace_double_array(Wallis7_c_inplace_array)

all_funs = {
    'Wallis7_py'                            : test_from_fun(Wallis7_py)                         ,
    'Wallis7_c_double_jit'                  : test_from_fun(Wallis7_c_double_jit)               ,
    'Wallis7_c_inplace_array_jit'           : test_from_fun(Wallis7_c_inplace_array_jit)        ,
    'Wallis7_c_inplace_array_cython'        : test_from_fun(scipy.LowLevelCallable.from_cython(
        choreo.segm.cython.test                                                                 ,
        "Wallis7_c_inplace_array_cython"                                                        ,
    ))                                                                                          ,
    'Wallis7_c_inplace_memoryview_cython'   : test_from_fun(scipy.LowLevelCallable.from_cython(
        choreo.segm.cython.test                                                                 ,
        "Wallis7_c_inplace_memoryview_cython"                                                   ,
    ))                                                                                          ,
}

all_nint = np.array([2**i for i in range(10)])

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

# %%
# 

relative_to_val = {pyquickbench.fun_ax_name:'Wallis7_c_inplace_array_cython'}

pyquickbench.plot_benchmark(
    all_times                           ,
    all_nint                            ,
    all_funs                            ,
    show = True                         ,
    relative_to_val = relative_to_val   ,
    title = 'Timings relative to Wallis7_c_inplace_array_cython'      ,
)

