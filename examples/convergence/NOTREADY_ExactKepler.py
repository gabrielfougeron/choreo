"""

"""

import os
import sys
import matplotlib.pyplot as plt
import math
import threadpoolctl

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

threadpoolctl.threadpool_limits(limits=1).__enter__()

if ("--no-show" in sys.argv):
    plt.show = (lambda : None)

bench_folder = os.path.join(__PROJECT_ROOT__,'examples','generated_files')

if not(os.path.isdir(bench_folder)):
    os.makedirs(bench_folder)
    
bench_filename = os.path.join(bench_folder, 'Kepler.npz')

# %%

import numpy as np
import choreo
import pyquickbench

def kepler_error(nint_fac, eccentricity, method):
    
    nbody = 2
    mass = 1
    
    NBS, dict_res = choreo.NBodySyst.KeplerEllipse(nbody, nint_fac, mass, eccentricity)
    NBS.fft_backend = "ducc"
    # NBS.fft_backend = "scipy"
    segmpos = dict_res["segmpos"]
    
    res = np.nan
    
    norm_ord = np.inf
    # norm_ord = 1
    # norm_ord = 2
    
    if method == "spectral":
        
        params_buf = NBS.segmpos_to_params(segmpos)
        action_grad = NBS.segmpos_params_to_action_grad(segmpos, params_buf)

        res = np.linalg.norm(action_grad, ord = norm_ord)
        
    else:
        
        rk_method, nsteps = method.split('-')
        
        nsteps = int(nsteps)
        keep_freq = 1
        nint_ODE = (NBS.segm_store-1) * keep_freq
        
        rk = choreo.segm.multiprec_tables.ComputeImplicitRKTable(nsteps, method=rk_method)
        
        x0 = dict_res["reg_x0"][0,:]
        v0 = dict_res["reg_v0"][0,:]
        
        # segmpos_ODE, segmmom_ODE = choreo.segm.ODE.SymplecticIVP(
        #     fun = dict_res["fun"]       ,
        #     gun = dict_res["gun"]       ,
        #     t_span = dict_res["t_span"] ,
        #     rk = rk                     ,
        #     keep_freq = keep_freq       ,
        #     nint = nint_ODE             ,
        #     keep_init = True            ,
        #     x0 = x0                     ,
        #     v0 = v0                     ,
        #     vector_calls = True         ,
        #     # DoEFT = False               ,
        # )

        # segmpos_ODE = np.ascontiguousarray(segmpos_ODE.reshape((NBS.segm_store, NBS.nsegm, NBS.geodim)).swapaxes(0, 1))
        # segmmom_ODE = np.ascontiguousarray(segmmom_ODE.reshape((NBS.segm_store, NBS.nsegm, NBS.geodim)).swapaxes(0, 1))
        
        # res = np.linalg.norm((segmpos - segmpos_ODE).reshape(-1), ord = norm_ord) / np.linalg.norm(segmpos_ODE.reshape(-1), ord = norm_ord)
        # res = np.linalg.norm((segmpos - segmpos_ODE)[:,-1,:].reshape(-1), ord = norm_ord) 
        
        # / np.linalg.norm(segmpos_ODE[:,-1,:].reshape(-1), ord = norm_ord)
        
        

        x0 = dict_res["reg_x0"][0,:]
        v0 = dict_res["reg_v0"][0,:]
        
        xf, vf = choreo.segm.ODE.SymplecticIVP(
            fun = dict_res["fun"]       ,
            gun = dict_res["gun"]       ,
            t_span = dict_res["t_span"] ,
            rk = rk                     ,
            nint = nint_ODE             ,
            x0 = x0                     ,
            v0 = v0                     ,
            vector_calls = True         ,
            # DoEFT = False               ,
        )
        
        res = np.linalg.norm((xf - dict_res["reg_x0"][-1,:]).reshape(-1), ord = norm_ord) 
    
    return res

    
def setup(nint_fac, eccentricity, method):
    return {"nint_fac":nint_fac, "eccentricity":eccentricity, "method":method}

imax = 20
n_per_i = 16
num = n_per_i * imax + 1

meth_name_list =  [
    "Gauss",    
    "Radau_IA", 
    "Radau_IIA",    
    "Radau_IB", 
    "Radau_IIB",    
    # "Lobatto_IIIA", 
    # "Lobatto_IIIB", 
    # "Lobatto_IIIC", 
    # "Lobatto_IIIC*",    
    # "Lobatto_IIID", 
    # "Lobatto_IIIS", 
    # "Cheb_I",   
    # "Cheb_II",  
    # "ClenshawCurtis",   
    # "NewtonCotesOpen",  
    # "NewtonCotesClosed",    
]


meth_nit = [2,3,4]



all_args = {
    "nint_fac" : [math.floor(2**i) for i in np.linspace(0, imax, num = num, endpoint=True)],
    # "eccentricity" : [0., 0.01, 0.2, 0.5, 0.8, 0.95, 0.99, 0.999],
    # "eccentricity" : [0.999,0.9999],
    "eccentricity" : [0.8],
    # "method" : ["spectral", "Gauss-1", "Gauss-2", "Gauss-3", "Gauss-4", "Gauss-10"],
    "method" : [meth_name+"-"+str(i) for i in meth_nit for meth_name in meth_name_list],
    # "method" : ["spectral"],
}

plot_intent = {
    pyquickbench.fun_ax_name    : "subplot_grid_x" ,
    "nint_fac"   : "points"       ,
    # "eccentricity"  : "curve_color"  ,
    "eccentricity"  : "subplot_grid_y"  ,
    "method"       : "curve_color"  ,
    # pyquickbench.out_ax_name : "subplot_grid_y"        ,
}

res = pyquickbench.run_benchmark(
    all_args                    ,
    [kepler_error]              ,
    setup = setup               ,
    mode = "scalar_output"      ,
    filename = bench_filename   ,
    pooltype="process"          ,
    # nproc = 8                   ,
    ForceBenchmark = True       ,
    StopOnExcept = True         ,
)

# print(res)


pyquickbench.plot_benchmark(
    res                             ,
    all_args                        ,
    [kepler_error]                  ,
    plot_intent = plot_intent       ,
    show = True                     ,
    plot_ylim = [1e-18, 1.]         ,
    # transform = 'pol_cvgence_order' ,
)