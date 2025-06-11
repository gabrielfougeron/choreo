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

import tests.test_config

if ("--no-show" in sys.argv):
    plt.show = (lambda : None)

bench_folder = os.path.join(__PROJECT_ROOT__,'examples','generated_files')

if not(os.path.isdir(bench_folder)):
    os.makedirs(bench_folder)
    
error_filename = os.path.join(bench_folder, 'RKConvergence_error.npz')
timings_filename = os.path.join(bench_folder, 'RKConvergence_timings.npz')

ForceBenchmark = True
# ForceBenchmark = False

# sphinx_gallery_end_ignore

def ODE_cpte_error_on_test(rk, ivp, nint, fun_type, DoEFT):

    vector_calls = not isinstance(rk, choreo.segm.ODE.ExplicitSymplecticRKTable)
    fun, gun = ivp["fgun"][(fun_type, vector_calls)]
    t_span = ivp["t_span"]
        
    x0 = ivp["x0"]
    v0 = ivp["v0"]
    
    if isinstance(rk, choreo.segm.ODE.ExplicitSymplecticRKTable):
        xf, vf = choreo.segm.ODE.ExplicitSymplecticIVP(
            fun             ,
            gun             ,
            t_span          ,
            x0              ,
            v0              ,
            rk = rk         ,
            nint = nint     ,
            DoEFT = DoEFT   ,
        )
    else:
        xf, vf = choreo.segm.ODE.ImplicitSymplecticIVP(
            fun                         ,
            gun                         ,
            t_span                      ,
            x0                          ,
            v0                          ,
            rk_x = rk[0]                ,
            rk_v = rk[1]                ,
            nint = nint                 ,
            DoEFT = DoEFT               ,
            vector_calls = vector_calls ,
            # eps = 1e-30                 ,
            # maxiter = 100                ,
        )
    
    xf_ex = ivp["ex_sol_x"]
    vf_ex =  ivp["ex_sol_v"]
    
    res = 2 * (np.linalg.norm(xf-xf_ex) + np.linalg.norm(vf-vf_ex)) / (np.linalg.norm(xf) + np.linalg.norm(vf) + np.linalg.norm(xf_ex) + np.linalg.norm(vf_ex))
    
    return res
    
def setup(rk, ivp, nint, fun_type, DoEFT):
    
    if rk in tests.test_config.Explicit_tables_dict:
        rk_ = tests.test_config.Explicit_tables_dict[rk]
    else:
        rk_ = choreo.segm.multiprec_tables.ComputeImplicitSymplecticRKTablePair(n=5, method=rk, dps=1000)
        
    ivp_ = tests.test_config.define_ODE_ivp(ivp)
    
    return {'rk': rk_, 'ivp': ivp_, 'nint': nint, 'fun_type': fun_type, 'DoEFT': DoEFT}

eq_names = [
    # "ypp=minus_y",
    # "ypp=xy",
    # "choreo_3C-Figure_eight"        ,
    # "choreo_3D-Heart"        ,
    # "choreo_5C3k-Double_lobes"        ,
    # "choreo_5C-Pinched_circle"        ,
    # "choreo_5D3k-Flower"        ,
    "choreo_complex_mass_charges"        ,
    # "choreo_2C2C1C-Double_blob"        ,
    # "choreo_2D-Circle"        ,
]


# "Gauss"             ,
# "Radau_IA"          ,
# "Radau_IIA"         ,
# "Radau_IB"          ,
# "Radau_IIB"         ,
# "Lobatto_IIIA"      ,
# "Lobatto_IIIB"      ,
# "Lobatto_IIIC"      ,
# "Lobatto_IIIC*"     ,
# "Lobatto_IIID"      ,            
# "Lobatto_IIIS"      ,     
# "Cheb_I"            ,
# "Cheb_II"           ,
# "ClenshawCurtis"    ,
# "NewtonCotesOpen"   ,
# "NewtonCotesClosed" ,

all_rk = [name for name, method in tests.test_config.Explicit_tables_dict.items() if method.th_cvg_rate <= 4]

# all_rk.extend(["Gauss", "Radau_IA", "Radau_IIA", "Radau_IB", "Radau_IIB", "Lobatto_IIIA", "Lobatto_IIIB"])

all_args = {
    # "rk" : [name for name, method in tests.test_config.Explicit_tables_dict.items()],
    # "rk" : [name for name, method in tests.test_config.Explicit_tables_dict.items() if method.th_cvg_rate <= 8],
    # "rk" : ["Gauss", "Radau_IA", "Radau_IIA", "Radau_IB", "Radau_IIB", "Lobatto_IIIA", "Lobatto_IIIB", "Lobatto_IIIC","Lobatto_IIIC*" , "Lobatto_IIID" , "Lobatto_IIIS", "Cheb_I", "Cheb_II", "ClenshawCurtis", "NewtonCotesOpen","NewtonCotesClosed"],
    "rk" : all_rk,
    # "rk" : ["Gauss", "SofSpa10"],
    "ivp" : eq_names,
    # "nint" : np.array([2**i for i in range(20)]) ,
    "nint" : np.array([2**i for i in range(7)]) ,
    "fun_type" : ["c_fun_memoryview"] ,
    "DoEFT" : [True] ,
    # "DoEFT" : [False, True] ,
}

plot_intent = {
    "rk" : 'curve_color'                 ,
    "ivp" : 'subplot_grid_y'               ,
    "nint" : 'points'               ,
    # "nint" : 'subplot_grid_y'               ,
    "fun_type" : 'subplot_grid_y'                   ,
    "DoEFT" : 'curve_linestyle'              ,
    pyquickbench.fun_ax_name : 'subplot_grid_y',
}

# %%
# The following plots give the measured relative error as a function of the number of quadrature subintervals

plot_ylim = None
# plot_ylim = [1e-17,1e1]

all_errors = pyquickbench.run_benchmark(
    all_args                        ,
    [ODE_cpte_error_on_test]        ,
    setup = setup                   ,
    mode = "scalar_output"          ,
    # pooltype = 'process'            ,
    filename = error_filename       ,
    StopOnExcept = True             ,
    ForceBenchmark = ForceBenchmark ,
)

for transform, ylabel in zip([
    None,
    "pol_cvgence_order",
],[
    "Relative error",
    "Order of convergence",
]):

    pyquickbench.plot_benchmark(
        all_errors                  ,
        all_args                    ,
        [ODE_cpte_error_on_test]    ,
        show = True                 ,
        plot_intent = plot_intent   ,
        plot_ylim = plot_ylim       ,
        transform = transform       ,
        ylabel = ylabel             ,
    )


# %%
# Error as a function of running time

time_per_test = 0.002
timeout = 0.1
MonotonicAxes = ["nint"]

all_timings = pyquickbench.run_benchmark(
    all_args                        ,
    [ODE_cpte_error_on_test]        ,
    setup = setup                   ,
    mode = "timings"                ,
    filename = timings_filename     ,
    StopOnExcept = True             ,
    ForceBenchmark = ForceBenchmark ,
    timeout = timeout               ,
    MonotonicAxes = MonotonicAxes   ,
    time_per_test = time_per_test   ,
)

pyquickbench.plot_benchmark(
    all_timings                 ,
    all_args                    ,
    [ODE_cpte_error_on_test]    ,
    logx_plot = True            ,
    plot_ylim = plot_ylim       ,
    show = True                 ,
    plot_intent = plot_intent   ,
)

pyquickbench.plot_benchmark(
    all_errors                  ,
    all_args                    ,
    [ODE_cpte_error_on_test]    ,
    all_xvalues = all_timings   ,
    logx_plot = True            ,
    plot_ylim = plot_ylim       ,
    show = True                 ,
    plot_intent = plot_intent   ,
    ylabel = "Relative error"   ,
    xlabel = "Time (s)"         ,
)

