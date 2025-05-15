import os
import sys
__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(__PROJECT_ROOT__)

import choreo 
import time
import pyquickbench
import json
import numpy as np
import scipy
import scipy.sparse
import itertools
import traceback
import tqdm
import concurrent.futures
import threadpoolctl
import math

import tests.test_config

np.set_printoptions(
    precision = 3,
    edgeitems = 10,
    # linewidth = 150,
    linewidth = 300,
    floatmode = "fixed",
)


def proj_to_zero(array, eps=1e-14):
    for idx in itertools.product(*[range(i)  for i in array.shape]):
        if abs(array[idx]) < eps:
            array[idx] = 0.

def main():
        
    TT = pyquickbench.TimeTrain(
        include_locs = False    ,
        align_toc_names = True  ,
        relative_timings = True  ,
    )
# 
#     for name in tests.test_config.Sols_dict:
#     # for name in ['5C-Pinched_circle']:
#     # for name in ['2D-Circle']:
#         print()
#         # print("  OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO  ")
#         print()
#         print(name)
#         # print()
# 
#         doit(name)
#                 
#         TT.toc(name)
# 
#     print()
#     print(TT)    
    

    n_threads = 16
    with threadpoolctl.threadpool_limits(limits=1):
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_threads) as executor:

            res = []
            for name in tests.test_config.Sols_dict:
                res.append(executor.submit(doit, name))
            
            # Useful ?    
            # concurrent.futures.wait(res, return_when=concurrent.futures.FIRST_EXCEPTION)
            # executor.shutdown(wait=False, cancel_futures=True)
        

def doit(name):
    
    NBS_in, params_buf_in = tests.test_config.Sols_dict[name]
    
    
    if NBS_in.nsegm == NBS_in.nbody:
        NBS = NBS_in
        params_buf = params_buf_in
        
    else:
                
        NBS = NBS_in.copy_nosym()

        segmpos = NBS_in.params_to_segmpos(params_buf_in)
        all_bodypos = NBS_in.segmpos_to_allbody_noopt(segmpos, pos = True )
        params_buf = NBS.segmpos_to_params(all_bodypos)

    NBS.ForceGreaterNStore = True
    
    segmpos = NBS.params_to_segmpos(params_buf)
    segmvel = NBS.params_to_segmvel(params_buf)
    xmin_arr, xmax_arr = NBS.DetectXlim(segmpos)
    xlim = [xmin_arr[0], xmax_arr[0], xmin_arr[1], xmax_arr[1]]
    
    diag = math.sqrt((xmax_arr[0]-xmin_arr[0])**2+(xmax_arr[1]-xmin_arr[1])**2)

    loop_len, bin_dx_min = NBS.segm_to_path_stats(segmpos, segmvel)
    Max_PathLength = loop_len.max()
    
    ODE_Syst = NBS.Get_ODE_def(params_buf, grad=True, regular_init = True)
    
    reg_init_freq = 1
    nint_ODE = (NBS.segm_store-1)*reg_init_freq

    nsteps = 20
    method = "Gauss"    
    rk = choreo.segm.multiprec_tables.ComputeImplicitRKTable(nsteps, method=method)
  
    segmpos_ODE, segmmom_ODE, segmpos_grad_ODE, segmmom_grad_ODE = choreo.segm.ODE.ImplicitSymplecticIVP(
        nint = nint_ODE                 ,
        rk_x = rk, rk_v = rk            ,
        reg_init_freq = reg_init_freq   ,
        **ODE_Syst                      ,
    )

    n = NBS.nsegm * NBS.geodim

    MonodromyMat = NBS.PropagateMonodromy(segmpos_grad_ODE, segmmom_grad_ODE)
    MonodromyMat_sq = MonodromyMat.reshape(2*n,2*n)
    
    Instability_magnitude, Instability_directions = choreo.scipy_plus.linalg.InstabilityDecomposition(MonodromyMat_sq)
    
    # print(Instability_magnitude)
    # print(Instability_magnitude)
    
    ODE_Syst = NBS.Get_ODE_def(params_buf)
    x0 = ODE_Syst['x0'].copy()
    v0 = ODE_Syst['v0'].copy()
    
    n_eig = min(6, 2*n)
    # n_eig = 6
    
    n_period = 2.
    
    all_filenames = []
    # for i_eig in tqdm.tqdm(range(n_eig)):
    for i_eig in range(n_eig):
        
        # alpha = 1./Instability_magnitude[i_eig]
        alpha = 3. /(Instability_magnitude[0]**n_period)
        
        if alpha < 1e-10:
            return
        
        alpha = min(alpha, 0.1*diag)
        
        ODE_Syst['x0'] = x0 + alpha * Instability_directions[i_eig,:n]
        ODE_Syst['v0'] = v0 + alpha * Instability_directions[i_eig,n:]
            
        keep_freq = 1
        nint_ODE = (NBS.segm_store-1)*NBS.nint_min
        ODE_Syst['t_span'] = (0., n_period)
        
        segmpos_ODE, segmmom_ODE = choreo.segm.ODE.ImplicitSymplecticIVP(
            nint = nint_ODE         ,
            rk_x = rk, rk_v = rk    ,
            keep_init = True        ,
            keep_freq = keep_freq   ,
            **ODE_Syst              ,
        )
        
        allpos = segmpos_ODE.reshape(( -1, NBS.nbody,NBS.geodim)).swapaxes(0,1)
        
        filename = os.path.join(__PROJECT_ROOT__, "scripts", "output", name + str(i_eig).zfill(3)+".mp4")
        all_filenames.append(filename)
        NBS.plot_all_2D_anim(allpos, filename, color="body", xlim=xlim, Max_PathLength=Max_PathLength*n_period, ShootingStars=True, Periodic=False)
        
    grid_filename = os.path.join(__PROJECT_ROOT__, "scripts", "output", f'{name}_instability_grid.mp4')
    choreo.tools.VideoGrid(all_filenames, grid_filename)

    for filename in all_filenames:
        os.remove(filename)

if __name__ == "__main__":
    main()
