"""
Convergence analysis of explicit Runge-Kutta methods for ODE IVP
================================================================
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

import matplotlib.pyplot as plt
import numpy as np
import math as m
import scipy

import choreo
import choreo.scipy_plus.precomputed_tables as precomputed_tables

bench_folder = os.path.join(__PROJECT_ROOT__,'examples','generated_files')

if not(os.path.isdir(bench_folder)):
    os.makedirs(bench_folder)
    
basename_bench_filename = 'ExplicitRK_ivp_cvg_bench_'

# ForceBenchmark = True
ForceBenchmark = False

# sphinx_gallery_end_ignore

eq_names = [
    "y'' = -y"          ,
    "y'' = - exp(y)"    ,
    "y'' = xy"          ,
    "y' = Az; z' = By"  ,
]


all_methods = { name : getattr(precomputed_tables,name) for name in dir(precomputed_tables) if isinstance(getattr(precomputed_tables,name),choreo.scipy_plus.ODE.ExplicitSymplecticRKTable) }

method_order_hierarchy = {}
for name, rk in all_methods.items():

    order = rk.th_cvg_rate

    cur_same_order = method_order_hierarchy.get(order, {})
    cur_same_order[name] = rk
    method_order_hierarchy[order] = cur_same_order

sorted_method_order = sorted(method_order_hierarchy)

# sphinx_gallery_start_ignore


all_nint = np.array([2**i for i in range(10)])

all_benchs = {
    f'{eq_name} methods of order {order}' : {
        f'{rk_name}' : functools.partial(
            choreo.scipy_plus.test.ODE_cpte_error_on_test ,
            eq_name    ,
            rk,
            mode = 'VX'       ,
        ) for rk_name, rk in method_order_hierarchy[order].items()
    } for order, eq_name in itertools.product(sorted_method_order, eq_names)
}


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
        plot_ylim = plot_ylim                       ,
        fig = fig                                   ,
        ax = axs[i_bench,0]                         ,
        title = f'Relative error on integrand {bench_name}' ,
    )
    
plot_xlim = axs[0,0].get_xlim()
    
plt.tight_layout()

# sphinx_gallery_end_ignore

plt.show()


# %%
# The following plots give the measured convergence rate as a function of the number of quadrature subintervals.
# The dotted lines are theoretical convergence rates.

# sphinx_gallery_start_ignore

plt.close()

plot_ylim = [0,10]

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
        transform = "pol_cvgence_order"             ,
        plot_xlim = plot_xlim                       ,
        plot_ylim = plot_ylim                       ,
        logx_plot = True                            ,
        # clip_vals = True                            ,
        # stop_after_first_clip = True                ,
        fig = fig                                   ,
        ax = axs[i_bench,0]                         ,
        title = f'Approximate convergence rate on integrand {bench_name}' ,
    )
    
    for fun_name, fun in all_funs.items():
            
        rk = fun.args[1]
        
        th_order = rk.th_cvg_rate
        xlim = axs[i_bench,0].get_xlim()

        axs[i_bench,0].plot(xlim, [th_order, th_order], linestyle='dotted')
        
plt.tight_layout()
 
# sphinx_gallery_end_ignore

plt.show()

# %%
# We can see 3 distinct phases on these plots:
# 
# * A first pre-convergence phase, where the convergence rate is growing towards its theoretical value. the end of the pre-convergence phase occurs for a number of sub-intervals roughtly independant of the convergence order of the quadrature method.
# * A steady convergence phase where the convergence remains close to the theoretical value
# * A final phase, where the relative error stagnates arround 1e-15. The value of the integral is computed with maximal accuracy given floating point precision. The approximation of the convergence rate is dominated by seemingly random floating point errors.
# 

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

    all_errors = choreo.benchmark.run_benchmark(
        all_nint                        ,
        all_funs                        ,
        setup = setup                   ,
        mode = "scalar_output"          ,
        filename = bench_filename       ,
        ForceBenchmark = ForceBenchmark ,
    )
    
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
        all_errors                                  ,
        all_nint                                    ,
        all_funs                                    ,
        all_xvalues = all_times                     ,
        plot_ylim = plot_ylim                       ,
        logx_plot = True                            ,
        fig = fig                                   ,
        ax = axs[i_bench,0]                         ,
        title = f'Relative error as a function of computational cost for equation {bench_name}' ,
    )

plt.tight_layout()
 
# sphinx_gallery_end_ignore

plt.show()

# %%
# Error as a function of running time for different orders

# sphinx_gallery_start_ignore

plt.close()

best_method_by_order = {
    1: 'SymplecticEuler',
    2: 'McAte2',
    3: 'McAte3',
    4: 'McAte4',
    5: 'McAte5',
    6: 'KahanLi6',
    8: 'KahanLi8',
    10:'SofSpa10',
}


all_benchs = {
    f'{eq_name}' : {
        f'{rk_name}' : functools.partial(
            choreo.scipy_plus.test.ODE_cpte_error_on_test ,
            eq_name    ,
            all_methods[rk_name],
            mode = 'VX'       ,
        ) for (order, rk_name) in best_method_by_order.items()
    } for eq_name in eq_names
}

n_bench = len(all_benchs)
figsize = (1600/dpi, n_bench * 800 / dpi)

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
    
    bench_filename = os.path.join(bench_folder,basename_bench_filename+str(i_bench).zfill(2)+'_error_best.npy') 

    all_errors = choreo.benchmark.run_benchmark(
        all_nint                        ,
        all_funs                        ,
        setup = setup                   ,
        mode = "scalar_output"          ,
        filename = bench_filename       ,
        ForceBenchmark = ForceBenchmark ,
    )
    
    timings_filename = os.path.join(bench_folder,basename_bench_filename+str(i_bench).zfill(2)+'_timings_best.npy') 
    
    all_times = choreo.benchmark.run_benchmark(
        all_nint                        ,
        all_funs                        ,
        setup = setup_timings           ,
        mode = "timings"                ,
        filename = timings_filename     ,
        ForceBenchmark = ForceBenchmark ,
    )
    
    choreo.plot_benchmark(
        all_errors                                  ,
        all_nint                                    ,
        all_funs                                    ,
        all_xvalues = all_times                     ,
        plot_ylim = plot_ylim                       ,
        logx_plot = True                            ,
        fig = fig                                   ,
        ax = axs[i_bench,0]                         ,
        title = f'Relative error as a function of computational cost for equation {bench_name}' ,
    )

plt.tight_layout()
 
# sphinx_gallery_end_ignore

plt.show()


