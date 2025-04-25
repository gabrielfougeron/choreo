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
    
bench_filename = os.path.join(bench_folder, 'ExplicitRKConvergence.npz')

ForceBenchmark = True
# ForceBenchmark = False

# sphinx_gallery_end_ignore

def ODE_cpte_error_on_test(rk, ivp, nint, fun_type, DoEFT):

    vector_calls = not isinstance(rk, choreo.segm.ODE.ExplicitSymplecticRKTable)
    fun, gun = ivp["fgun"][(fun_type, vector_calls)]
    t_span = ivp["t_span"]
        
    x0 = ivp["x0"]
    v0 = ivp["v0"]
    
    reg_x0 = ivp["reg_x0"]
    reg_v0 = ivp["reg_v0"]
    
    # reg_init_freq = nint
    reg_init_freq = -1
    
    segm_store = reg_x0.shape[0]
    keep_freq = nint
    nint = (segm_store-1) * keep_freq
    
    if isinstance(rk, choreo.segm.ODE.ExplicitSymplecticRKTable):
        segmpos_ODE, segmmom_ODE = choreo.segm.ODE.ExplicitSymplecticIVP(
            fun                             ,
            gun                             ,
            t_span                          ,
            rk = rk                         ,
            nint = nint                     ,
            keep_freq = keep_freq           ,
            keep_init = True                ,
            DoEFT = DoEFT                   ,
            reg_x0 = reg_x0                 ,
            reg_v0 = reg_v0                 ,
            reg_init_freq = reg_init_freq   ,
        )
    else:
        segmpos_ODE, segmmom_ODE = choreo.segm.ODE.ImplicitSymplecticIVP(
            fun                             ,
            gun                             ,
            t_span                          ,
            rk_x = rk[0]                    ,
            rk_v = rk[1]                    ,
            nint = nint                     ,
            keep_freq = keep_freq           ,
            keep_init = True                ,
            DoEFT = DoEFT                   ,
            vector_calls = vector_calls     ,
            reg_x0 = reg_x0                 ,
            reg_v0 = reg_v0                 ,
            reg_init_freq = reg_init_freq   ,
            # eps = 1e-30                     ,
            # maxiter = 100                   ,
        )
    
    res = 2 * (np.linalg.norm(segmpos_ODE-reg_x0) + np.linalg.norm(segmmom_ODE-reg_v0)) / (np.linalg.norm(segmpos_ODE) + np.linalg.norm(segmmom_ODE) + np.linalg.norm(reg_x0) + np.linalg.norm(reg_v0))
    # 
    # res = 2 * (np.linalg.norm(segmpos_ODE[-1,:]-reg_x0[-1,:]) + np.linalg.norm(segmmom_ODE[-1,:]-reg_v0[-1,:])) / (np.linalg.norm(segmpos_ODE[-1,:]) + np.linalg.norm(segmmom_ODE[-1,:]) + np.linalg.norm(reg_x0[-1,:]) + np.linalg.norm(reg_v0[-1,:]))
    
    return res

def setup(rk, ivp, nint, fun_type, DoEFT):
    
    if rk in tests.test_config.Explicit_tables_dict:
        rk_ = tests.test_config.Explicit_tables_dict[rk]
    else:
        rk_ = choreo.segm.multiprec_tables.ComputeImplicitSymplecticRKTablePair(n=2, method=rk, dps=1000)
        
    ivp_ = tests.test_config.define_ODE_ivp(ivp)
    
    return {'rk': rk_, 'ivp': ivp_, 'nint': nint, 'fun_type': fun_type, 'DoEFT': DoEFT}

eq_names = [
    "choreo_3C-Figure_eight"        ,
    # "choreo_3D-Heart"        ,
    # "choreo_5C3k-Double_lobes"        ,
    # "choreo_5C-Pinched_circle"        ,
    # "choreo_5D3k-Flower"        ,
    # "choreo_complex_mass_charges"        ,
    # "choreo_2C2C1C-Double_blob"        ,
    # "choreo_2D-Circle"        ,
    # "choreo_1C1C-Ellipse"        ,
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

all_args = {
    # "rk" : [name for name, method in tests.test_config.Explicit_tables_dict.items()],
    "rk" : [name for name, method in tests.test_config.Explicit_tables_dict.items() if method.th_cvg_rate == 2],
    # "rk" : ["Gauss", "Radau_IA", "Radau_IIA", "Radau_IB", "Radau_IIB", "Lobatto_IIIA", "Lobatto_IIIB", "Lobatto_IIIC","Lobatto_IIIC*" , "Lobatto_IIID" , "Lobatto_IIIS", "Cheb_I", "Cheb_II", "ClenshawCurtis", "NewtonCotesOpen","NewtonCotesClosed"],
    # "rk" : ["Gauss", "Radau_IA", "Radau_IIA", "Radau_IB", "Radau_IIB", "Lobatto_IIIA", "Lobatto_IIIB", "Lobatto_IIIC","Lobatto_IIIC*" , "Lobatto_IIID" , "Lobatto_IIIS"],
    # "rk" : ["Gauss"],
    # "rk" : ["Gauss", "SofSpa10"],
    "ivp" : eq_names,
    "nint" : np.array([2**i for i in range(10)]) ,
    # "nint" : np.array([100]) ,
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

all_vals = pyquickbench.run_benchmark(
    all_args                        ,
    [ODE_cpte_error_on_test]        ,
    setup = setup                   ,
    mode = "scalar_output"          ,
    pooltype = 'process'            ,
    filename = bench_filename       ,
    StopOnExcept = True             ,
    ForceBenchmark = ForceBenchmark ,
)

for (transform, ylabel) in zip([
    None,
    "pol_cvgence_order",
], [
    "Relative error",
    "Convergence rate"
]):

    pyquickbench.plot_benchmark(
        all_vals                    ,
        all_args                    ,
        [ODE_cpte_error_on_test]    ,
        show = True                 ,
        plot_intent = plot_intent   ,
        plot_ylim = plot_ylim       ,
        transform = transform       ,
        ylabel = ylabel             ,
    )
