import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'

import concurrent.futures
import multiprocessing
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

__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(__PROJECT_ROOT__)

import choreo 


store_folder = os.path.join(__PROJECT_ROOT__,'NumericalTank_tests')
NT_init_filename = os.path.join(__PROJECT_ROOT__,'NumericalTank_data','init_data_np.txt')
all_NT_init = np.loadtxt(NT_init_filename)



Save_All_Pos = True
# Save_All_Pos = False

# Save_All_Coeffs = True
Save_All_Coeffs = False

# Save_All_Coeffs_No_Sym = True
Save_All_Coeffs_No_Sym = False

# Save_Newton_Error = True
Save_Newton_Error = False

Save_img = True
# Save_img = False
# 
# Save_thumb = True
Save_thumb = False

# img_size = (12,12) # Image size in inches
img_size = (8,8) # Image size in inches
thumb_size = (2,2) # Image size in inches

color = "body"
# color = "loop"
# color = "velocity"
# color = "all"

# Save_anim = True
Save_anim = False

Save_ODE_anim = True
# Save_ODE_anim = False

Plot_trace_anim = True

fps=60
# fps=30

vid_size = (8,8) # Image size in inches
# vid_size_perturb = (4,4) # Image size in inches
# vid_size_perturb = (8,8) # Image size in inches
# nint_plot_anim = 2*2*2*3*3
# nint_plot_anim = 2*2*2*3*3*5
dnint = 1




for n_NT_init in [0]:


    nbody = 3
    nint = nbody * 100

    mass = np.ones(nbody)
    Sym_list = []

    # MomConsImposed = True
    MomConsImposed = False

    n_reconverge_it_max = 0
    n_grad_change = 1.

    ActionSyst = choreo.setup_changevar(2,nbody,nint,mass,n_reconverge_it_max,Sym_list=Sym_list,MomCons=MomConsImposed,n_grad_change=n_grad_change,CrashOnIdentity=False)

    # print(choreo.all_unique_SymplecticIntegrators.keys())

    # SymplecticMethod = 'SymplecticEuler'
    # SymplecticMethod = 'SymplecticStormerVerlet'
    # SymplecticMethod = 'SymplecticRuth3'
    SymplecticMethod = 'SymplecticRuth4'

    SymplecticIntegrator = choreo.GetSymplecticIntegrator(SymplecticMethod)



    # nint_ODE_mul = 64
    nint_ODE_mul = 2**11
    nint_ODE = nint_ODE_mul*nint

    fun,gun = ActionSyst.GetSymplecticODEDef()

    ndof = nbody*ActionSyst.geodim

    # all_x = np.zeros((nint,ActionSyst.nbody,ActionSyst.geodim))
    all_x = np.zeros((ActionSyst.nbody,ActionSyst.geodim,nint))


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


    # T = 1. / T_NT
    # T = 1.
    T = T_NT

    xi = x0
    vi = v0

    xf = x0
    vf = v0

    for iint in range(nint):

        x0 = xf
        v0 = vf

        all_x[:,:,iint] = x0.reshape(ActionSyst.nbody,ActionSyst.geodim)

        # t_span = (T * iint / nint, T * (iint+1) / nint)
        t_span = (iint / nint, (iint+1) / nint)

        xf,vf = SymplecticIntegrator(fun,gun,t_span,x0,v0,nint_ODE_mul)


    period_err = np.linalg.norm(xi-xf) + np.linalg.norm(vi-vf)

    print(f'Error on Periodicity: {period_err}')






    bare_name = 'test_'+str(n_NT_init).zfill(5)

    filename_output = os.path.join(store_folder,bare_name)

    print('Saving solution as '+filename_output+'.*')

    # ActionSyst.Write_Descriptor(x,filename_output+'.json')

    # if Save_img :
    #     ActionSyst.plot_all_2D(x,nint_plot_img,filename_output+'.png',fig_size=img_size,color=color)
    # 
    # if Save_thumb :
    #     ActionSyst.plot_all_2D(x,nint_plot_img,filename_output+'_thumb.png',fig_size=thumb_size,color=color)
    #     
    # if Save_anim :
    #     ActionSyst.plot_all_2D_anim(x,nint_plot_anim,filename_output+'.mp4',nperiod_anim,Plot_trace=Plot_trace_anim,fig_size=vid_size,dnint=dnint,color_list=color_list,color=color)
    # 
    # if Save_Newton_Error :
    #     ActionSyst.plot_Newton_Error(x,filename_output+'_newton.png')
    # 
    # if Save_All_Coeffs:
    # 
    #     np.save(filename_output+'_coeffs.npy',all_coeffs)
    # 
    # if Save_All_Pos:
    # 
    #     np.save(filename_output+'.npy',all_pos)
    # 
    # if Save_All_Coeffs_No_Sym:
    #     
    #     all_coeffs_nosym = ActionSyst.RemoveSym(x)
    # 
    #     np.save(filename_output+'_nosym.npy',all_coeffs_nosym)

    if Save_ODE_anim:

        ActionSyst.plot_all_2D_anim(filename=filename_output+'_ode.mp4',Plot_trace=Plot_trace_anim,fig_size=vid_size,dnint=dnint,all_pos_trace=all_x,all_pos_points=all_x,fps=fps)



