"""
Convergence analysis of integration methods on segment
======================================================
"""

# %%
# Evaluation of relative quadrature error with the following parameters:

# sphinx_gallery_start_ignore

import os
import sys
import itertools
import functools

try:
    __PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))

    if ':' in __PROJECT_ROOT__:
        __PROJECT_ROOT__ = os.getcwd()

except (NameError, ValueError): 

    __PROJECT_ROOT__ = os.path.abspath(os.path.join(os.getcwd(),os.pardir))

sys.path.append(__PROJECT_ROOT__)

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import matplotlib.pyplot as plt
import numpy as np
import math as m

import choreo
import pyquickbench

if ("--no-show" in sys.argv):
    plt.show = (lambda : None)

bench_folder = os.path.join(__PROJECT_ROOT__,'examples','generated_files')

if not(os.path.isdir(bench_folder)):
    os.makedirs(bench_folder)
    
basename_bench_filename = 'quad_cvg_bench_'

# ForceBenchmark = True
ForceBenchmark = False

# sphinx_gallery_end_ignore

all_integrands = [
    "exp",
]

methods = [
    'Gauss',
    'Radau_I',
    'Radau_II',
    'Lobatto_III',
    'Cheb_I',
    'Cheb_II',
    'ClenshawCurtis',
]

all_nsteps = [i for i in range(2,40)]
refinement_lvl = [1]

def setup(fun_name, quad_method, quad_nsteps, nint):
    return {'fun_name': fun_name, 'quad_method': quad_method, 'quad_nsteps': quad_nsteps, 'nint': nint}

all_args = {
    'integrand': all_integrands,
    'quad_method': methods,
    'quad_nsteps': all_nsteps,
    'nint': refinement_lvl,
}
    
all_funs = [choreo.scipy_plus.test.Quad_cpte_error_on_test]
    

bench_filename = os.path.join(bench_folder,basename_bench_filename+'.npz')

all_errors = pyquickbench.run_benchmark(
    all_args                        ,
    all_funs                        ,
    setup = setup                   ,
    mode = "scalar_output"          ,
    filename = bench_filename       ,
    ForceBenchmark = ForceBenchmark ,
    StopOnExcept = True             ,
    # pooltype = "process"            ,
)

plot_intent = {
    'integrand': "subplot_grid_x"       ,
    'quad_method': "curve_color"        ,
    'quad_nsteps': "points"             ,
    'nint': "subplot_grid_y"            ,
    pyquickbench.fun_ax_name : "same"   ,
}

fig, ax = pyquickbench.plot_benchmark(
    all_errors                              ,
    all_args                                ,
    all_funs                                ,
    plot_intent = plot_intent               ,
    logx_plot = False                        ,
    title = 'Absolute error of quadrature'  ,
)

plot_xlim = ax[0,0].get_xlim()
fig.tight_layout()
fig.show()
