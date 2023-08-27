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

import matplotlib.pyplot as plt
import numpy as np
import scipy
import math as m

import choreo

def cpte_error(
    fun_name,
    quad_method,
    quad_nsteps,
    nint,
):

    if fun_name == "exp" :
        # WOLFRAM
        # f(x) = y*exp(y*x)
        # F(x) = exp(y*x)

        test_ndim = 50

        fun = lambda x: np.array([y*m.exp(y*x) for y in range(test_ndim)])
        Fun = lambda x: np.array([m.exp(y*x) for y in range(test_ndim)])
        
        x_span = (0.,1.)
        exact = Fun(x_span[1]) - Fun(x_span[0])


    quad = choreo.scipy_plus.SegmQuad.ComputeQuadrature(quad_method, quad_nsteps)

    approx = choreo.scipy_plus.SegmQuad.IntegrateOnSegment(
        fun = fun       ,
        ndim = test_ndim,
        x_span = x_span ,
        quad = quad     ,
        nint = nint     ,
    )
    
    # print()
    # print(approx)
    # print(exact)
    # print(approx/exact)

    error = np.linalg.norm(approx-exact)/np.linalg.norm(exact)

    return error


# sphinx_gallery_end_ignore

fun_names = [
    "exp",
]

methods = [
    'Gauss'
]

all_nsteps = range(1,11)
refinement_lvl = np.array(range(1,100))

# sphinx_gallery_start_ignore

def setup(nint):
    return nint
    
all_benchs = {
    fun_name : {
        f'{method} {nsteps}' : functools.partial(
            cpte_error  ,
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
# The following plots give the measured convergence rate as a function of the number of quadrature subintervals

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

    all_errors = choreo.benchmark.run_benchmark(
        refinement_lvl          ,
        all_funs                ,
        setup = setup           ,
        mode = "scalar_output"  ,
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
# The following plots give the measured convergence rate as a function of the number of quadrature subintervals

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

    all_errors = choreo.benchmark.run_benchmark(
        refinement_lvl          ,
        all_funs                ,
        setup = setup           ,
        mode = "scalar_output"  ,
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
    
plt.tight_layout()

# sphinx_gallery_end_ignore

plt.show()