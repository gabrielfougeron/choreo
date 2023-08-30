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
import choreo.scipy_plus.precomputed_tables as precomputed_tables

bench_folder = os.path.join(__PROJECT_ROOT__,'examples','generated_files')

if not(os.path.isdir(bench_folder)):
    os.makedirs(bench_folder)
    
basename_bench_filename = 'SelectedRK_ivp_cvg_bench_'

# ForceBenchmark = True
ForceBenchmark = False

def cpte_error(
    eq_name,
    rk_method,
    nint,
):

    if eq_name == "y'' = -y" :
        # WOLFRAM
        # y'' = - y
        # y(x) = A cos(x) + B sin(x)

        test_ndim = 2

        ex_sol = lambda t : np.array( [ np.cos(t) , np.sin(t),-np.sin(t), np.cos(t) ]  )

        fun = lambda t,y:   np.asarray(y)
        gun = lambda t,x:  -np.asarray(x)

    if eq_name == "y'' = - exp(y)" :
        # WOLFRAM
        # y'' = - exp(y)
        # y(x) = - 2 * ln( cosh(t / sqrt(2) ))

        test_ndim = 1

        invsqrt2 = 1./np.sqrt(2.)
        sqrt2 = np.sqrt(2.)
        ex_sol = lambda t : np.array( [ -2*np.log(np.cosh(invsqrt2*t)) , -sqrt2*np.tanh(invsqrt2*t) ]  )

        fun = lambda t,y:  np.array(y)
        gun = lambda t,x: -np.exp(x)

    if eq_name == "y'' = xy" :

        # Solutions: Airy functions
        # Nonautonomous linear test case

        test_ndim = 2

        def ex_sol(t):

            ai, aip, bi, bip = scipy.special.airy(t)

            return np.array([ai,bi,aip,bip])

        fun = lambda t,y: np.array(y)
        gun = lambda t,x: np.array([t*x[0],t*x[1]],dtype=np.float64)
        
    if eq_name == "y' = Az; z' = By" :

        test_ndim = 10

        A = np.diag(np.array(range(test_ndim)))
        B = np.identity(test_ndim)

        AB = np.zeros((2*test_ndim,2*test_ndim))
        AB[0:test_ndim,test_ndim:2*test_ndim] = A
        AB[test_ndim:2*test_ndim,0:test_ndim] = B

        yo = np.array(range(test_ndim))
        zo = np.array(range(test_ndim))

        yzo = np.zeros(2*test_ndim)
        yzo[0:test_ndim] = yo
        yzo[test_ndim:2*test_ndim] = zo

        def ex_sol(t):

            return scipy.linalg.expm(t*AB).dot(yzo)

        fun = lambda t,z: A.dot(z)
        gun = lambda t,y: B.dot(y)

    t_span = (0.,np.pi)

    ex_init  = ex_sol(t_span[0])
    ex_final = ex_sol(t_span[1])

    x0 = ex_init[0          :  test_ndim].copy()
    v0 = ex_init[test_ndim  :2*test_ndim].copy()
    
    xf,vf = choreo.scipy_plus.ODE.SymplecticIVP(
        fun             ,
        gun             ,
        t_span          ,
        x0              ,
        v0              ,
        rk = rk_method  ,
        nint = nint     ,
    )
        
    sol = np.ascontiguousarray(np.concatenate((xf,vf),axis=0).reshape(2*test_ndim))
    error = np.linalg.norm(sol-ex_final)/np.linalg.norm(ex_final)

    return error

def get_implicit_table(rk_name, order):
    
    if rk_name == "Gauss":
        
        return choreo.scipy_plus.multiprec_tables.ComputeImplicitSymplecticRKTable_Gauss(order)
    
    raise ValueError(f"Unknown {rk_name = }")

# sphinx_gallery_end_ignore

eq_names = [
    "y'' = -y"          ,
    "y'' = - exp(y)"    ,
    "y'' = xy"          ,
    "y' = Az; z' = By"  ,
]

implicit_methods = {f'{rk_name} {order}' : get_implicit_table(rk_name, order) for rk_name, order in itertools.product(["Gauss"], [2,4,6,8])}

explicit_methods = {rk_name : getattr(globals()['precomputed_tables'], rk_name) for rk_name in [
    'McAte4'    ,
    'McAte5'    ,
    'KahanLi8'  ,
    'SofSpa10'  ,
]}


# sphinx_gallery_start_ignore

all_nint = np.array([2**i for i in range(14)])

all_methods = {**explicit_methods, **implicit_methods}

all_benchs = {
    eq_name : {
        f'{rk_name}' : functools.partial(
            cpte_error ,
            eq_name    ,
            rk
        ) for rk_name, rk in all_methods.items()
    } for eq_name in eq_names
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
        logx_plot = True                            ,
        fig = fig                                   ,
        ax = axs[i_bench,0]                         ,
        plot_ylim = plot_ylim                       ,
        title = f'Relative error as a function of computational cost for equation {bench_name}' ,
    )

plt.tight_layout()
 
# sphinx_gallery_end_ignore

plt.show()


