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

# def ODE_cpte_error_on_test(rk, ivp, nint, fun_type, DoEFT):
# 
#     fun, gun = ivp["fgun"][(fun_type, False)]
#     t_span = ivp["t_span"]
#         
#     x0 = ivp["x0"]
#     v0 = ivp["v0"]
#     
#     xf, vf = choreo.segm.ODE.ExplicitSymplecticIVP(
#         fun             ,
#         gun             ,
#         t_span          ,
#         x0              ,
#         v0              ,
#         rk = rk         ,
#         nint = nint     ,
#         DoEFT = DoEFT   ,
#     )
#     
#     xf_ex = ivp["ex_sol_x"]
#     vf_ex =  ivp["ex_sol_v"]
#     
#     res = 2 * (np.linalg.norm(xf-xf_ex) + np.linalg.norm(vf-vf_ex)) / (np.linalg.norm(xf) + np.linalg.norm(vf) + np.linalg.norm(xf_ex) + np.linalg.norm(vf_ex))
#     
#     return res

def ODE_cpte_error_on_test(rk, ivp, nint, fun_type, DoEFT, tend):

    fun, gun = ivp["fgun"][(fun_type, False)]
    t_span = (0., tend)
    nint = nint * tend
        
    x0 = ivp["ex_sol_fun_x"](t_span[0])
    v0 = ivp["ex_sol_fun_v"](t_span[0])
    
    xf, vf = choreo.segm.ODE.SymplecticIVP(
        fun             ,
        gun             ,
        t_span          ,
        x0              ,
        v0              ,
        rk = rk         ,
        nint = nint     ,
        DoEFT = DoEFT   ,
    )
    
    xf_ex = ivp["ex_sol_fun_x"](t_span[1])
    vf_ex = ivp["ex_sol_fun_v"](t_span[1])
    
    res = 2 * (np.linalg.norm(xf-xf_ex) + np.linalg.norm(vf-vf_ex)) / (np.linalg.norm(xf) + np.linalg.norm(vf) + np.linalg.norm(xf_ex) + np.linalg.norm(vf_ex))
    
    return res
    
def setup(rk, ivp, nint, fun_type, DoEFT, tend):
    
    if rk in tests.test_config.Explicit_tables_dict:
        rk_ = tests.test_config.Explicit_tables_dict[rk]
    else:
        rk_ = choreo.segm.multiprec_tables.ComputeImplicitRKTable(n=20, method=rk, dps=1000)
    ivp_ = tests.test_config.define_ODE_ivp(ivp)
    
    return {'rk': rk_, 'ivp': ivp_, 'nint': nint, 'fun_type': fun_type, 'DoEFT': DoEFT, 'tend': tend}

eq_names = [
    "ypp=minus_y",
    # "ypp=xy",
    # "choreo_3C-Figure_eight"        ,
    # "choreo_3D-Heart"        ,
    # "choreo_5C3k-Double_lobes"        ,
    # "choreo_5C-Pinched_circle"        ,
    # "choreo_5D3k-Flower"        ,
    # "choreo_complex_mass_charges"        ,
    # "choreo_2C2C1C-Double_blob"        ,
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
    # "rk" : [name for name, method in tests.test_config.Explicit_tables_dict.items() if method.th_cvg_rate == 8],
    "rk" : ["Gauss", "Radau_IA", "Radau_IIA", "Radau_IB", "Radau_IIB", "Lobatto_IIIA", "Lobatto_IIIB", "Lobatto_IIIC",],
    "ivp" : eq_names,
    # "nint" : np.array([2**i for i in range(20)]) ,
    "nint" : np.array([10]) ,
    "fun_type" : ["c_fun_memoryview"] ,
    # "fun_type" : ["py_fun","c_fun_memoryview","c_fun_pointer",] ,
    # "DoEFT" : [True] ,
    "DoEFT" : [False, True] ,
    "tend" : np.array([2**i for i in range(20)]) ,
}

plot_intent = {
    "rk" : 'curve_color'                 ,
    "ivp" : 'subplot_grid_y'               ,
    # "nint" : 'points'               ,
    "nint" : 'subplot_grid_y'               ,
    "fun_type" : 'subplot_grid_y'                   ,
    "DoEFT" : 'curve_linestyle'              ,
    pyquickbench.fun_ax_name : 'subplot_grid_y',
    # "tend" : 'subplot_grid_y'               ,
    "tend" : 'points' ,
}

# %%
# The following plots give the measured relative error as a function of the number of quadrature subintervals

plot_ylim = None
# plot_ylim = [1e-17,1e1]

pyquickbench.run_benchmark(
    all_args                        ,
    [ODE_cpte_error_on_test]        ,
    setup = setup                   ,
    mode = "scalar_output"          ,
    pooltype = 'process'            ,
    filename = bench_filename       ,
    ForceBenchmark = ForceBenchmark ,
    StopOnExcept = True             ,
    show = True                     ,
    plot_intent = plot_intent       ,
    plot_ylim = plot_ylim           ,
    # transform = "pol_cvgence_order" ,
    # transform = "pol_growth_order" ,

)


# # %%
# # Error as a function of running time
# 
# # sphinx_gallery_start_ignore
# 
# plt.close()
# 
# fig, axs = plt.subplots(
#     nrows = n_bench,
#     ncols = 1,
#     sharex = False,
#     sharey = False,
#     figsize = figsize,
#     dpi = dpi   ,
#     squeeze = False,
# )
# 
# i_bench = -1
# 
# plot_ylim = [1e-17,1e1]
# 
# for bench_name, all_funs in all_benchs.items():
# 
#     i_bench += 1
#     
#     bench_filename = os.path.join(bench_folder,basename_bench_filename+str(i_bench).zfill(2)+'_error.npy') 
# 
#     all_errors = pyquickbench.run_benchmark(
#         all_nint                        ,
#         all_funs                        ,
#         setup = setup                   ,
#         mode = "scalar_output"          ,
#         filename = bench_filename       ,
#         ForceBenchmark = ForceBenchmark ,
#     )
#     
#     timings_filename = os.path.join(bench_folder,basename_bench_filename+str(i_bench).zfill(2)+'_timings.npy') 
#     
#     all_times = pyquickbench.run_benchmark(
#         all_nint                        ,
#         all_funs                        ,
#         setup = setup                   ,
#         mode = "timings"                ,
#         filename = timings_filename     ,
#         ForceBenchmark = ForceBenchmark ,
#     )
#     
#     pyquickbench.plot_benchmark(
#         all_errors                                  ,
#         all_nint                                    ,
#         all_funs                                    ,
#         all_xvalues = all_times                     ,
#         logx_plot = True                            ,
#         fig = fig                                   ,
#         ax = axs[i_bench,0]                         ,
#         plot_ylim = plot_ylim                       ,
#         title = f'Relative error as a function of computational cost for equation {bench_name}' ,
#     )
# 
# plt.tight_layout()
#  
# # sphinx_gallery_end_ignore
# 
# plt.show()
# 
# 
