"""
Convergence analysis of Runge-Kutta methods for ODE IVP
=======================================================
"""

# %%
# Evaluation of relative quadrature error with the following parameters:

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

sys.path.append(__PROJECT_ROOT__)

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import matplotlib.pyplot as plt
import numpy as np
import math as m
import scipy

import choreo
import choreo.segm.precomputed_tables as precomputed_tables

import pyquickbench

if ("--no-show" in sys.argv):
    plt.show = (lambda : None)

bench_folder = os.path.join(__PROJECT_ROOT__,'examples','generated_files')

if not(os.path.isdir(bench_folder)):
    os.makedirs(bench_folder)
    
basename_bench_filename = 'SelectedRK_ivp_cvg_bench_'

# ForceBenchmark = True
ForceBenchmark = False

def get_implicit_table_pair(rk_name, order):

    return choreo.segm.multiprec_tables.ComputeImplicitSymplecticRKTablePair(order)
    
    raise ValueError(f"Unknown {rk_name = }")

# sphinx_gallery_end_ignore

eq_names = [
    "y'' = -y"          ,
    "y'' = - exp(y)"    ,
    "y'' = xy"          ,
    "y' = Az; z' = By"  ,
]

implicit_methods = {
    f'{rk_name} {order}' : choreo.segm.multiprec_tables.ComputeImplicitSymplecticRKTablePair(order,method=rk_name) for rk_name, order in itertools.product(["Gauss"], [2,4,6,8])
    # f'{rk_name} {order}' : choreo.segm.multiprec_tables.ComputeImplicitSymplecticRKTablePair(order, method=rk_name) for rk_name, order in itertools.product(["Radau_IIA"], [2,3,4,5])
}

explicit_methods = {
    rk_name : getattr(globals()['precomputed_tables'], rk_name) for rk_name in [
            'McAte4'    ,
            'McAte5'    ,
            'KahanLi8'  ,
            'SofSpa10'  ,
    ]
}

scipy_methods = [
    "RK45" ,  
    "RK23" ,  
    "DOP853" ,  
    "Radau" ,  
    "BDF" ,  
    "LSODA" ,  
]


# sphinx_gallery_start_ignore

all_nint = np.array([2**i for i in range(12)])

all_benchs = {}
for eq_name in eq_names:
    bench = {}
    
    for method, (rk, rk_ad) in implicit_methods.items():
        
        bench[f'{method}'] = functools.partial(
            choreo.segm.test.ISPRK_ODE_cpte_error_on_test ,
            eq_name     ,
            rk          ,
            rk_ad       ,
        )  
            
    for method, rk in explicit_methods.items():
        
        bench[f'{method}'] = functools.partial(
            choreo.segm.test.ODE_cpte_error_on_test ,
            eq_name    ,
            rk
        )  
        
    for method in scipy_methods:
        
        bench[f'{method}'] = functools.partial(
            choreo.segm.test.scipy_ODE_cpte_error_on_test ,
            eq_name ,
            method  ,     
        )
    
    all_benchs[eq_name] = bench

all_nint = np.array([2**i for i in range(12)])

def setup(nint):
    return {'nint': nint}

n_bench = len(all_benchs)

dpi = 150
figsize = (1600/dpi, n_bench * 800 / dpi)

# sphinx_gallery_end_ignore

# %%
# The following plots give the measured relative error as a function of the number of quadrature subintervals

# sphinx_gallery_start_ignore

fig, axs = plt.subplots(
    nrows = n_bench,
    ncols = 1,
    sharex = True,
    sharey = False,
    figsize = figsize,
    dpi = dpi   ,
    squeeze = False,
)

i_bench = -1

plot_ylim = [1e-17,1e1]

for bench_name, all_funs in all_benchs.items():

    i_bench += 1
    
    bench_filename = os.path.join(bench_folder,basename_bench_filename+str(i_bench).zfill(2)+'_error.npy')
    
    all_errors = pyquickbench.run_benchmark(
        all_nint                        ,
        all_funs                        ,
        setup = setup                   ,
        mode = "scalar_output"          ,
        filename = bench_filename       ,
        ForceBenchmark = ForceBenchmark ,
    )

    pyquickbench.plot_benchmark(
        all_errors                                  ,
        all_nint                                    ,
        all_funs                                    ,
        fig = fig                                   ,
        ax = axs[i_bench,0]                         ,
        plot_ylim = plot_ylim                       ,
        title = f'Relative error on integrand {bench_name}' ,
    )
    
plot_xlim = axs[0,0].get_xlim()
    
plt.tight_layout()

# sphinx_gallery_end_ignore

plt.show()

# %%
# Error as a function of running time

# sphinx_gallery_start_ignore

plt.close()

fig, axs = plt.subplots(
    nrows = n_bench,
    ncols = 1,
    sharex = False,
    sharey = False,
    figsize = figsize,
    dpi = dpi   ,
    squeeze = False,
)

i_bench = -1

plot_ylim = [1e-17,1e1]

for bench_name, all_funs in all_benchs.items():

    i_bench += 1
    
    bench_filename = os.path.join(bench_folder,basename_bench_filename+str(i_bench).zfill(2)+'_error.npy') 

    all_errors = pyquickbench.run_benchmark(
        all_nint                        ,
        all_funs                        ,
        setup = setup                   ,
        mode = "scalar_output"          ,
        filename = bench_filename       ,
        ForceBenchmark = ForceBenchmark ,
    )
    
    timings_filename = os.path.join(bench_folder,basename_bench_filename+str(i_bench).zfill(2)+'_timings.npy') 
    
    all_times = pyquickbench.run_benchmark(
        all_nint                        ,
        all_funs                        ,
        setup = setup                   ,
        mode = "timings"                ,
        filename = timings_filename     ,
        ForceBenchmark = ForceBenchmark ,
    )
    
    pyquickbench.plot_benchmark(
        all_errors                                  ,
        all_nint                                    ,
        all_funs                                    ,
        all_xvalues = all_times                     ,
        logx_plot = True                            ,
        fig = fig                                   ,
        ax = axs[i_bench,0]                         ,
        plot_ylim = plot_ylim                       ,
        title = f'Relative error as a function of computational cost for equation {bench_name}' ,
    )

plt.tight_layout()
 
# sphinx_gallery_end_ignore

plt.show()


