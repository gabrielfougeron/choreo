import os
import concurrent.futures
import multiprocessing

import json
import shutil
import random
import time
import math
import numpy as np
import sys
import fractions
import functools

__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(__PROJECT_ROOT__)

import choreo 

store_folder = os.path.join(__PROJECT_ROOT__,'Sniff_all_sym','Simo_tests')
NT_init_filename = os.path.join(__PROJECT_ROOT__,'NumericalTank_data','Simo_init_cond.txt')
all_NT_init = np.loadtxt(NT_init_filename)

import pyquickbench

def Integrate(n_NT_init):
    
    print('')
    print('#####################################')
    print(f'    Simo {n_NT_init}')
    print('#####################################')
    print('')

    geodim = 2
    nbody = 3

    bodymass = np.ones(nbody)
    bodycharge = bodymass.copy()
    Sym_list = []


    BodyPerm = np.array([1,2,0], dtype=np.intp)
    SpaceRot = np.identity(geodim, dtype=np.float64)

    Sym_choreo = choreo.ActionSym(
        BodyPerm.copy() ,
        SpaceRot.copy() ,
        1               ,
        1               ,
        nbody           ,
    )
    
    BodyPerm = np.array([0,2,1], dtype=np.intp)
    SpaceRot = np.array([[1,0],[0,-1]], dtype=np.float64)
    
    Sym_reflexion = choreo.ActionSym(
        BodyPerm.copy() ,
        SpaceRot.copy() ,
        -1              ,
        0               ,
        1               ,
    )
    
    Sym_list = [Sym_choreo, Sym_reflexion]

    NBS = choreo.NBodySyst(geodim, nbody, bodymass, bodycharge, Sym_list)

    NBS.ForceGreaterNStore = True

    print(NBS.DescribeSystem())

    
    ODEinitparams = np.random.random((NBS.n_ODEinitparams))
    
    x0, v0 = NBS.ODE_params_to_initposmom(ODEinitparams)
    
    print(x0)
    print(v0)
    
    exit()



    file_basename = 'Simo_'+(str(n_NT_init).zfill(5))
    save_filename = os.path.join(store_folder, file_basename)

    ndof = NBS.geodim * NBS.nsegm

    x0 = np.zeros(ndof)
    v0 = np.zeros(ndof)

    # Initial positions: (-2 c1, 0), (c1, c2), (c1, -c2) * 6
    # Initial velocities: (0, -2 c4), (c3, c4), (-c3, c4)
    
    phys_exp = 1/(NBS.Homo_exp-1)
      
    a = 6

    x0[0] = -2 * all_NT_init[n_NT_init,0] * a
    x0[1] = 0
    x0[2] =   all_NT_init[n_NT_init,0]    * a
    x0[3] =   all_NT_init[n_NT_init,1]    * a
    x0[4] =   all_NT_init[n_NT_init,0]    * a
    x0[5] = - all_NT_init[n_NT_init,1]    * a

    v0[0] = 0
    v0[1] = -2 * all_NT_init[n_NT_init,3]
    v0[2] =   all_NT_init[n_NT_init,2]   
    v0[3] =   all_NT_init[n_NT_init,3]   
    v0[4] = - all_NT_init[n_NT_init,2]   
    v0[5] =   all_NT_init[n_NT_init,3]   

    T_NT = all_NT_init[n_NT_init,4] * 18

    rfac = (T_NT) ** phys_exp
    
    # print(f'{phys_exp = }')
    # print(f'{rfac = }')

    x0 *= rfac
    v0 *= rfac * T_NT 

    ODE_Syst = NBS.Get_ODE_def()
    
    NBS.nint_fac = 32
    
    dx = np.ones(NBS.geodim)
    
    while np.linalg.norm(dx) > 1e-6:
        
        NBS.nint_fac = NBS.nint_fac * 2

        nsteps = 10
        keep_freq = 1
        nint_ODE = (NBS.segm_store-1) * keep_freq
        method = "Gauss"
        
        rk = choreo.segm.multiprec_tables.ComputeImplicitRKTable(nsteps, method=method)

        xf, vf = choreo.segm.ODE.ImplicitSymplecticIVP(
            x0 = x0                 ,
            v0 = v0                 ,
            rk_x = rk               ,
            rk_v = rk               ,
            keep_freq = keep_freq   ,
            nint = nint_ODE         ,
            keep_init = True        ,
            **ODE_Syst              ,
        )
        
        xf = xf.reshape(NBS.segm_store, NBS.nsegm, NBS.geodim)
        vf = vf.reshape(NBS.segm_store, NBS.nsegm, NBS.geodim)
        
        n = (NBS.segm_store-1)
        
        dx = NBS.Compute_periodicity_default_pos(xf[0,:,:].reshape(-1), xf[n,:,:].reshape(-1))
        
        print(f'{NBS.nint_fac = }')
        print(f'error = {np.linalg.norm(dx)}')
    
    assert np.linalg.norm(dx) < 1e-6
# 
    # segmpos_ODE = np.ascontiguousarray(xf.reshape((NBS.segm_store, NBS.nsegm, NBS.geodim)).swapaxes(0, 1))
#     segmmom_ODE = np.ascontiguousarray(vf.reshape((NBS.segm_store, NBS.nsegm, NBS.geodim)).swapaxes(0, 1))

    # params_mom_buf = NBS.segmpos_to_params(segmpos_ODE)

    # allbodypos = NBS.segmpos_to_allbody_noopt(segmpos_ODE)
    # NBS.plot_all_2D_anim(allbodypos, save_filename + '.mp4')
    # NBS.plot_segmpos_2D(segmpos_ODE, save_filename + '.png')
    # np.save(save_filename + '.npy', segmpos_ODE)
    # NBS.Write_Descriptor(params_mom_buf=params_mom_buf, segmpos=segmpos_ODE, filename = save_filename+'.json')


the_NT_init = range(len(all_NT_init))
# the_NT_init = range(5)

# the_NT_init = [0]
# the_NT_init.extend(range(25,len(all_NT_init)))

# # 
# 
if __name__ == "__main__":

    TT = pyquickbench.TimeTrain()

    for n_NT_init in the_NT_init:
        
        Integrate(n_NT_init)
        TT.toc(n_NT_init)
        
    # print(TT)



#     # n = multiprocessing.cpu_count() // 2
#     n = 8
# 
#     print(f"Executing with {n} workers")
# 
#     with concurrent.futures.ProcessPoolExecutor(max_workers=n) as executor:
#         
#         res = []
#         for n_NT_init in range(len(all_NT_init)):
#             res.append(executor.submit(Integrate,n_NT_init))
# 

