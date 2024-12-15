"""
Benchmark of Error-Free Transforms for ODE IVP
==============================================
"""



# %%
# This benchmark compares accuracy and efficiency of several summation algorithms in floating point arithmetics for initial value problems of ordinary differential equations

# sphinx_gallery_start_ignore

import os
import sys
import itertools
import functools

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

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import math as m
import scipy

import choreo
import choreo.scipy_plus.precomputed_tables as precomputed_tables

import pyquickbench

if ("--no-show" in sys.argv):
    plt.show = (lambda : None)

bench_folder = os.path.join(__PROJECT_ROOT__,'examples','generated_files')

if not(os.path.isdir(bench_folder)):
    os.makedirs(bench_folder)
    
basename_bench_filename = 'EFT_RK_bench'

# ForceBenchmark = True
ForceBenchmark = False

# sphinx_gallery_end_ignore

eq_names = [
    "y'' = -y"          ,
    "y'' = - exp(y)"    ,
    "y'' = xy"          ,
    "y' = Az; z' = By"  ,
]

implicit_methods = {
    f'{rk_name} {order}' : choreo.scipy_plus.multiprec_tables.ComputeImplicitRKTable_Gauss(order, method=rk_name) for rk_name, order in itertools.product(["Gauss"], [2,4,6,8])
}

explicit_methods = {
    rk_name : getattr(globals()['precomputed_tables'], rk_name) for rk_name in [
            'McAte4'    ,
            'McAte5'    ,
            'KahanLi8'  ,
            'SofSpa10'  ,
    ]
}

# sphinx_gallery_start_ignore

all_nint = np.array([2**i for i in range(16)])

all_methods = {**explicit_methods, **implicit_methods}
all_bool = {"True":True, "False":False}

all_EFT = ["True", "False"]

all_funs = {
    rk_name : functools.partial(
        choreo.scipy_plus.test.ODE_cpte_error_on_test   ,
        rk_method = rk                                  ,
    ) for (rk_name, rk) in all_methods.items()
}

def setup(eq_name, nint, DoEFT):
    return {'eq_name':eq_name, 'nint': nint, 'DoEFT':all_bool[DoEFT]}

# sphinx_gallery_end_ignore

# %%
# The following plots give the measured relative error as a function of the number of quadrature subintervals

plot_ylim = [1e-17,1e1]

bench_filename = os.path.join(bench_folder, basename_bench_filename+'_error.npz')

all_args = {
    "eq_name" : eq_names                ,
    "nint" : all_nint                   ,
    "DoEFT" : all_EFT                   ,
}

all_errors = pyquickbench.run_benchmark(
    all_args                        ,
    all_funs                        ,  
    setup = setup                   ,
    mode = "scalar_output"          ,
    filename = bench_filename       ,
    ForceBenchmark = ForceBenchmark ,
)

plot_intent = {
    "eq_name" : "subplot_grid_y"                ,
    "nint" : "points"                           ,
    "DoEFT" : "curve_linestyle"                 ,
    pyquickbench.fun_ax_name : "curve_color"    ,
}

pyquickbench.plot_benchmark(
    all_errors                              ,
    all_args                                ,
    all_funs                                ,
    mode = "scalar_output"                  ,
    show = True                             ,
    plot_ylim = plot_ylim                   ,
    plot_intent = plot_intent               ,
    title = f'Relative error on integrand'  ,
)
    

# %%
# Error as a function of running time

bench_filename = os.path.join(bench_folder, basename_bench_filename+'_timings.npz')

all_timings = pyquickbench.run_benchmark(
    all_args                        ,
    all_funs                        ,  
    setup = setup                   ,
    filename = bench_filename       ,
    ForceBenchmark = ForceBenchmark ,
)

pyquickbench.plot_benchmark(
    all_errors                  ,
    all_args                    ,
    all_funs                    ,
    all_xvalues = all_timings   ,
    show = True                 ,
    plot_ylim = plot_ylim       ,
    plot_intent = plot_intent   ,
    xlabel = 'Time (s)'         ,
    ylabel = 'Relative error'   ,
)

