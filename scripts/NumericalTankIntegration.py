import os
import concurrent.futures
import multiprocessing

os.environ['NUMBA_NUM_THREADS'] = str(multiprocessing.cpu_count()//2)
os.environ['OMP_NUM_THREADS'] = str(multiprocessing.cpu_count()//2)
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'


import json
import shutil
import random
import time
import math as m
import numpy as np
import scipy.linalg
import sys
import fractions
import scipy.integrate
import scipy.special
import functools
import inspect

import tqdm

__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(__PROJECT_ROOT__)

import choreo 


store_folder = os.path.join(__PROJECT_ROOT__,'Sniff_all_sym','NumericalTank_tests')
NT_init_filename = os.path.join(__PROJECT_ROOT__,'NumericalTank_data','init_data_np.txt')
all_NT_init = np.loadtxt(NT_init_filename)

def Integrate(n_NT_init):

    Save_img = True
    # Save_img = False

    # img_size = (12,12) # Image size in inches
    img_size = (8,8) # Image size in inches
    thumb_size = (2,2) # Image size in inches

    color = "body"
    # color = "loop"
    # color = "velocity"
    # color = "all"

    Save_anim = True
    # Save_anim = False

    vid_size = (8,8) # Image size in inches
    # nint_plot_anim = 2*2*2*3*3
    nint_plot_anim = 2*2*2*3*3*5
    dnint = 30

    nint_plot_img = nint_plot_anim * dnint

    min_n_steps_ode = 1*nint_plot_anim

    nperiod_anim = 1.

    Plot_trace_anim = True
    # Plot_trace_anim = False

    geodim = 2
    nbody = 3

    mass = np.ones(nbody)
    Sym_list = []

    # MomConsImposed = True
    MomConsImposed = False

    nint_small = 30
    n_reconverge_it_max_small = 0
    n_grad_change = 1.


    
    # Save_ODE_anim = True
    Save_ODE_anim = False



    ActionSyst_small = choreo.setup_changevar(geodim,nbody,nint_small,mass,n_reconverge_it_max_small,Sym_list=Sym_list,MomCons=MomConsImposed,n_grad_change=n_grad_change,CrashOnIdentity=False)

    # print(choreo.all_unique_SymplecticIntegrators.keys())

    # SymplecticMethod = 'SymplecticEuler'
    # SymplecticMethod = 'SymplecticStormerVerlet'
    # SymplecticMethod = 'SymplecticRuth3'
    # SymplecticMethod = 'SymplecticRuth4'
    # SymplecticMethod = 'SymplecticGauss1'
    # SymplecticMethod = 'SymplecticGauss2'
    # SymplecticMethod = 'SymplecticGauss3'
    # SymplecticMethod = 'SymplecticGauss5' 
    SymplecticMethod = 'SymplecticGauss10'

    SymplecticIntegrator = choreo.GetSymplecticIntegrator(SymplecticMethod)
    SymplecticTanIntegrator = choreo.GetSymplecticTanIntegrator(SymplecticMethod)

    fun,gun = ActionSyst_small.GetSymplecticODEDef()
    grad_fun,grad_gun = ActionSyst_small.GetSymplecticTanODEDef()
    ndof = nbody*ActionSyst_small.geodim

    w = np.zeros((2*ndof,2*ndof),dtype=np.float64)
    w[0:ndof,ndof:2*ndof] = np.identity(ndof)
    w[ndof:2*ndof,0:ndof] = -np.identity(ndof)



# 
#     print('')
#     print('#####################################')
#     print(f'    Numerical Tank {n_NT_init}')
#     print('#####################################')
#     print('')

    file_basename = 'NumericalTank_'+(str(n_NT_init).zfill(5))
    Info_filename = os.path.join(store_folder,file_basename + '.json')

    # if os.path.isfile(Info_filename):
    #     continue

    # print("Time forward integration")
    GoOn = True
    itry = -1


    x0 = np.zeros(ndof)
    v0 = np.zeros(ndof)

    # Initial positions: (-1, 0), (1, 0), (0, 0)
    # Initial velocities: (va, vb), (va, vb), (-2vb, -2vb)

    x0[0] = -1
    x0[2] = 1

    va = all_NT_init[n_NT_init,0]
    vb = all_NT_init[n_NT_init,1]
    T_NT = all_NT_init[n_NT_init,2]
    T_NT_s = all_NT_init[n_NT_init,3]

    v0[0] = va
    v0[1] = vb
    v0[2] = va
    v0[3] = vb
    v0[4] = -2*va
    v0[5] = -2*vb

    phys_exp = 1/(choreo.n-1)
    rfac = (T_NT) ** phys_exp

    x0 = x0 * rfac
    v0 = v0 * rfac * T_NT

    t_span = (0., 1.)

    nint_anim = nbody * 1024 * 128
    nint_sub = 16

    nint = nint_sub * nint_anim

    all_pos, all_v = SymplecticIntegrator(
        fun = fun,
        gun = gun,
        t_span = t_span,
        x0 = x0,
        v0 = v0,
        nint = nint,
        keep_freq = nint_sub
    )

    xf = all_pos[-1,:].copy()
    vf = all_v[-1,:].copy()

    dvx0 = np.concatenate((x0-xf,v0-vf)).reshape(2*ndof)
    period_err = np.linalg.norm(dvx0)
    print(f'Numerical Tank {n_NT_init:4d}. Error on periodicity : {period_err:.2e}')



    all_pos[1:,:] = all_pos[:-1,:].copy()
    all_pos[0,:] = x0.copy()

    all_pos = np.ascontiguousarray(all_pos.transpose().reshape(ActionSyst_small.nbody,ActionSyst_small.geodim,nint_anim))

# 
    if ActionSyst_small :
        filename_output = os.path.join(store_folder,file_basename + '_ode.png')

        ActionSyst_small.plot_given_2D(all_pos,filename_output,fig_size=img_size,color=color)

    if Save_ODE_anim:

        filename_output = os.path.join(store_folder,file_basename + '_ode.mp4')

        ActionSyst_small.plot_all_2D_anim(nint_plot=nint_anim,filename=filename_output,fig_size=vid_size,dnint=dnint,all_pos_trace=all_pos,all_pos_points=all_pos,color='body')


if __name__ == "__main__":

    # n = 5
    n = multiprocessing.cpu_count() // 2
    # n = 2
    
    print(f"Executing with {n} workers")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=n) as executor:
        
        res = []
        for n_NT_init in range(len(all_NT_init)):
        # for n_NT_init in range(4):
            res.append(executor.submit(Integrate,n_NT_init))
            time.sleep(0.01)