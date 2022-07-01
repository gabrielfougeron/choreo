import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import concurrent.futures
import multiprocessing
import shutil
import random
import time
import math as m
import numpy as np
import sys
import fractions

__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(__PROJECT_ROOT__)

import choreo 

import datetime


def main():


    input_folder = os.path.join(__PROJECT_ROOT__,'Sniff_all_sym/keep/6/')
# # 
#     input_names_list = []
#     for file_path in os.listdir(input_folder):
#         file_path = os.path.join(store_folder, file_path)
#         file_root, file_ext = os.path.splitext(os.path.basename(file_path))
#         
#         if (file_ext == '.txt' ):
#             input_names_list.append(file_root)

    input_names_list = ['6_3']



    store_folder = os.path.join(__PROJECT_ROOT__,'Sniff_all_sym/keep/mod/')
    # store_folder = input_folder

    Save_All_Coeffs = True
    # Save_All_Coeffs = False

    # Save_Newton_Error = True
    Save_Newton_Error = False

    Save_img = True
    # Save_img = False

    Save_thumb = True
    # Save_thumb = False

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
    nint_plot_anim = 2*2*2*3*3*5*2
    # nperiod_anim = 1./nbody
    dnint = 30

    nint_plot_img = nint_plot_anim * dnint

    try:
        the_lcm
    except NameError:
        period_div = 1.
    else:
        period_div = the_lcm
# 
    nperiod_anim = 1.
    # nperiod_anim = 1./period_div

    Plot_trace_anim = True
    # Plot_trace_anim = False

    GradActionThresh = 1e-8

    for the_name in input_names_list:

        print('')
        print(the_name)

        input_filename = os.path.join(input_folder,the_name)
        input_filename = input_filename + '.npy'

        all_coeffs = np.load(input_filename)

        theta = 2*np.pi * 0.
        SpaceRevscal = 1.
        SpaceRot = np.array( [[SpaceRevscal*np.cos(theta) , SpaceRevscal*np.sin(theta)] , [-np.sin(theta),np.cos(theta)]])
        TimeRev = -1.
        TimeShiftNum = 0
        TimeShiftDen = 2

        all_coeffs = choreo.Transform_Coeffs(SpaceRot, TimeRev, TimeShiftNum, TimeShiftDen, all_coeffs)

        ncoeff_init = all_coeffs.shape[2]

        the_i = -1
        the_i_max = 10

        Gradaction_OK = False

        while (not(Gradaction_OK) and (the_i < the_i_max)):

            the_i += 1

            p_list = range(the_i_max)
            # p_list = [3]
            p = p_list[the_i%len(p_list)]

            nc = 6

            mm = 2

            nbpl=[nc]

            SymType = {
                'name'  : 'D',
                'n'     : nc,
                'm'     : mm,
                'l'     : 0,
                'k'     : 1,
                'p'     : p,
                'q'     : nc,
            }
            Sym_list = choreo.Make2DChoreoSym(SymType,range(nc))
            nbody = nc

            mass = np.ones((nbody),dtype=np.float64)

            # MomConsImposed = True
            MomConsImposed = False

            n_reconverge_it_max = 0
            n_grad_change = 1.

            callfun = choreo.setup_changevar(nbody,ncoeff_init,mass,n_reconverge_it_max,Sym_list=Sym_list,MomCons=MomConsImposed,n_grad_change=n_grad_change,CrashOnIdentity=False)

            x = choreo.Package_all_coeffs(all_coeffs,callfun)

            Action,Gradaction = choreo.Compute_action(x,callfun)

            Gradaction_OK = (np.linalg.norm(Gradaction) < GradActionThresh)

        if (the_i == the_i_max):
            raise(ValueError('Correct Symmetries not found'))

        filename_output = os.path.join(store_folder,the_name)

        print('Saving solution as '+filename_output+'.*')

        choreo.Write_Descriptor(x,callfun,filename_output+'.txt')
        
        if Save_img :
            choreo.plot_all_2D(x,nint_plot_img,callfun,filename_output+'.png',fig_size=img_size,color=color)
        
        if Save_thumb :
            choreo.plot_all_2D(x,nint_plot_img,callfun,filename_output+'_thumb.png',fig_size=thumb_size,color=color)
            
        if Save_anim :
            choreo.plot_all_2D_anim(x,nint_plot_anim,callfun,filename_output+'.mp4',nperiod_anim,Plot_trace=Plot_trace_anim,fig_size=vid_size,dnint=dnint)

        if Save_Newton_Error :
            choreo.plot_Newton_Error(x,callfun,filename_output+'_newton.png')
        
        if Save_All_Coeffs:

            np.save(filename_output+'.npy',all_coeffs)



if __name__ == "__main__":
    main()    
