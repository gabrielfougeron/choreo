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

import choreo

bench_folder = os.path.join(__PROJECT_ROOT__,'examples','generated_files')

if not(os.path.isdir(bench_folder)):
    os.makedirs(bench_folder)
    
basename_bench_filename = 'quad_cvg_bench_'

ForceBenchmark = True
# ForceBenchmark = False

# sphinx_gallery_end_ignore

fun_names = [
    "exp",
]

methods = [
    'Gauss'
]

all_nsteps = range(1,11)
refinement_lvl = np.array(range(1,100))
# refinement_lvl = np.array([2**i for i in range(18)])

# sphinx_gallery_start_ignore

def setup(nint):
    return nint
    
all_benchs = {
    fun_name : {
        f'{method} {nsteps}' : functools.partial(
            choreo.scipy_plus.test.Quad_cpte_error_on_test  ,
            fun_name    ,
            method      ,
            nsteps      ,
        ) for method, nsteps in itertools.product(methods, all_nsteps)
    } for fun_name in fun_names
}

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
    sharey = True,
    figsize = figsize,
    dpi = dpi   ,
    squeeze = False,
)

i_bench = -1

for bench_name, all_funs in all_benchs.items():

    i_bench += 1

    bench_filename = os.path.join(bench_folder,basename_bench_filename+bench_name.replace(' ','_')+'.npy')

    all_errors = choreo.benchmark.run_benchmark(
        refinement_lvl                  ,
        all_funs                        ,
        setup = setup                   ,
        mode = "scalar_output"          ,
        filename = bench_filename       ,
        ForceBenchmark = ForceBenchmark ,
    )

    choreo.plot_benchmark(
        all_errors                                  ,
        refinement_lvl                              ,
        all_funs                                    ,
        fig = fig                                   ,
        ax = axs[i_bench,0]                         ,
        title = f'Absolute error on integrand {bench_name}' ,
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

plot_ylim = [0,20]

fig, axs = plt.subplots(
    nrows = n_bench,
    ncols = 1,
    sharex = True,
    sharey = True,
    figsize = figsize,
    dpi = dpi   ,
    squeeze = False,
)

i_bench = -1

for bench_name, all_funs in all_benchs.items():

    i_bench += 1

    bench_filename = os.path.join(bench_folder,basename_bench_filename+bench_name.replace(' ','_')+'.npy')

    all_errors = choreo.benchmark.run_benchmark(
        refinement_lvl                  ,
        all_funs                        ,
        setup = setup                   ,
        mode = "scalar_output"          ,
        filename = bench_filename       ,
        ForceBenchmark = ForceBenchmark ,
    )

    choreo.plot_benchmark(
        all_errors                                  ,
        refinement_lvl                              ,
        all_funs                                    ,
        transform = "pol_cvgence_order"             ,
        plot_xlim = plot_xlim                       ,
        plot_ylim = plot_ylim                       ,
        logx_plot = True                            ,
        clip_vals = True                            ,
        stop_after_first_clip = True                ,
        fig = fig                                   ,
        ax = axs[i_bench,0]                         ,
        title = f'Approximate convergence rate on integrand {bench_name}' ,
    )
    
    for fun_name, fun in all_funs.items():
            
        quad_method = fun.args[1]
        quad_nsteps = fun.args[2]

        quad = choreo.scipy_plus.SegmQuad.ComputeQuadrature(quad_method, quad_nsteps)
        th_order = quad.th_cvg_rate
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

