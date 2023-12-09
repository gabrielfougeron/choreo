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
import choreo.scipy_plus.precomputed_tables as precomputed_tables

if ("--no-show" in sys.argv):
    plt.show = (lambda : None)

bench_folder = os.path.join(__PROJECT_ROOT__,'examples','generated_files')

if not(os.path.isdir(bench_folder)):
    os.makedirs(bench_folder)
    
basename_bench_filename = 'EFT_RK_bench_'

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

def EFT_str(DoEFT):
    return 'w/ EFT' if DoEFT else 'w/o EFT'

all_EFT = [True, False]

all_benchs = {
    eq_name : {
        f'{rk_name} {EFT_str(DoEFT)}' : functools.partial(
            choreo.scipy_plus.test.ODE_cpte_error_on_test ,
            eq_name         ,
            rk              ,
            DoEFT = DoEFT   ,
        ) for (rk_name, rk), DoEFT in itertools.product(all_methods.items(), all_EFT)
    } for eq_name in eq_names
}

color_list = [choreo.benchmark.default_color_list[i] for (i, rk), DoEFT in itertools.product(enumerate(all_methods), all_EFT)]

linestyle_loop = [ 'solid','dotted']
linestyle_list = [linestyle_loop[i] for rk, (i,DoEFT) in itertools.product(all_methods, enumerate(all_EFT))]

def setup(nint):
    return nint

def setup_timings(nint):
    return [(nint, 'nint')]


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
    
    all_errors = choreo.benchmark.run_benchmark(
        all_nint                        ,
        all_funs                        ,
        setup = setup                   ,
        mode = "scalar_output"          ,
        filename = bench_filename       ,
        ForceBenchmark = ForceBenchmark ,
    )

    choreo.plot_benchmark(
        all_errors                                  ,
        all_nint                                    ,
        all_funs                                    ,
        color_list = color_list                     ,
        linestyle_list = linestyle_list             ,
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

all_nint = np.array([2**i for i in range(12)])

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


for bench_name, all_funs in all_benchs.items():

    i_bench += 1
    
    timings_filename = os.path.join(bench_folder,basename_bench_filename+str(i_bench).zfill(2)+'_timings.npy') 
    
    all_times = choreo.benchmark.run_benchmark(
        all_nint                        ,
        all_funs                        ,
        setup = setup_timings           ,
        mode = "timings"                ,
        filename = timings_filename     ,
        ForceBenchmark = ForceBenchmark ,
    )
    
    choreo.plot_benchmark(
        all_times                                   ,
        all_nint                                    ,
        all_funs                                    ,
        color_list = color_list                     ,
        linestyle_list = linestyle_list             ,
        logx_plot = True                            ,
        fig = fig                                   ,
        ax = axs[i_bench,0]                         ,
        title = f'Computational cost (s) as a function of number of iterations for equation {bench_name}' ,
    )

plt.tight_layout()
 
# sphinx_gallery_end_ignore

plt.show()


