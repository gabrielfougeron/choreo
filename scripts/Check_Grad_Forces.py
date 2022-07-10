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
import scipy.integrate

__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(__PROJECT_ROOT__)

import choreo 

import datetime


def main():


    input_folder = os.path.join(__PROJECT_ROOT__,'Sniff_all_sym/3/')
    # input_folder = os.path.join(__PROJECT_ROOT__,'Sniff_all_sym/copy/')
    # input_folder = os.path.join(__PROJECT_ROOT__,'Sniff_all_sym/keep/13')

#     ''' Include all files in tree '''
#     input_names_list = []
#     for root, dirnames, filenames in os.walk(input_folder):
# 
#         for filename in filenames:
#             file_path = os.path.join(root, filename)
#             file_root, file_ext = os.path.splitext(os.path.basename(file_path))
# 
#             if (file_ext == '.txt' ):
# 
#                 file_path = os.path.join(root, file_root)
#                 the_name = file_path[len(input_folder):]
#                 input_names_list.append(the_name)


    ''' Include all files in folder '''
    input_names_list = []
    for file_path in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_path)
        file_root, file_ext = os.path.splitext(os.path.basename(file_path))
        
        if (file_ext == '.txt' ):
            input_names_list.append(file_root)

    # input_names_list = ['00006']

    store_folder = os.path.join(__PROJECT_ROOT__,'Sniff_all_sym/mod')
    # store_folder = input_folder

    # Save_All_Coeffs = True
    Save_All_Coeffs = False

    # Save_All_Coeffs_No_Sym = True
    Save_All_Coeffs_No_Sym = False

    # Save_Newton_Error = True
    Save_Newton_Error = False

    # Save_img = True
    Save_img = False
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

    # Save_ODE_anim = True
    Save_ODE_anim = False

    # ODE_method = 'RK23'
    # ODE_method = 'RK45'
    ODE_method = 'DOP853'
    # ODE_method = 'Radau'
    # ODE_method = 'BDF'

    atol_ode = 1e-10
    rtol_ode = 1e-12

    vid_size = (8,8) # Image size in inches
    # nint_plot_anim = 2*2*2*3*3
    nint_plot_anim = 2*2*2*3*3*5
    dnint = 30

    nint_plot_img = nint_plot_anim * dnint


    min_n_steps_ode = 1*nint_plot_anim

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

        bare_name = the_name.split('/')[-1]

        all_coeffs = np.load(input_filename)

        theta = 2*np.pi * 0.5
        SpaceRevscal = 1.
        SpaceRot = np.array( [[SpaceRevscal*np.cos(theta) , SpaceRevscal*np.sin(theta)] , [-np.sin(theta),np.cos(theta)]])
        TimeRev = 1.
        TimeShiftNum = 0
        TimeShiftDen = 2

        all_coeffs = choreo.Transform_Coeffs(SpaceRot, TimeRev, TimeShiftNum, TimeShiftDen, all_coeffs)

        ncoeff_init = all_coeffs.shape[2]

        the_i = -1
        the_i_max = 20

        Gradaction_OK = False

        while (not(Gradaction_OK) and (the_i < the_i_max)):

            the_i += 1

            # p = 1
            p_list = range(the_i_max)
            # p_list = [3]
            p = p_list[the_i%len(p_list)]

            nc = 3

            mm = 1
            # mm_list = [1]
            # mm = mm_list[the_i%len(mm_list)]

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

        if not(Gradaction_OK):
            raise(ValueError('Correct Symmetries not found'))

        filename_output = os.path.join(store_folder,bare_name)

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

        if Save_All_Coeffs_No_Sym:
            
            all_coeffs_nosym = choreo.RemoveSym(x,callfun)

            np.save(filename_output+'_nosym.npy',all_coeffs_nosym)

        if Save_ODE_anim:
            
            y0 = choreo.Compute_init_pos_vel(x,callfun).reshape(-1)

            t_eval = np.array([i/nint_plot_img for i in range(nint_plot_img)])

            ode_res = scipy.integrate.solve_ivp(fun=choreo.Compute_ODE_RHS, t_span=(0.,1.), y0=y0, method=ODE_method, t_eval=t_eval, dense_output=False, events=None, vectorized=False, args=callfun,max_step=1./min_n_steps_ode,atol=atol_ode,rtol=rtol_ode)

            all_pos_vel = ode_res['y'].reshape(2,nbody,choreo.ndim,nint_plot_img)
            all_pos_ode = all_pos_vel[0,:,:,:]
            
            choreo.plot_all_2D_anim(x,nint_plot_anim,callfun,filename_output+'_ode.mp4',nperiod_anim,Plot_trace=Plot_trace_anim,fig_size=vid_size,dnint=dnint,all_pos_trace=all_pos_ode,all_pos_points=all_pos_ode)

        n_ddl = nbody*choreo.ndim*2
        yo = choreo.Compute_init_pos_vel(x,callfun).reshape(-1)

        dt_list = [10**(-i) for i in range(13)]
        dy = np.random.rand(n_ddl)

        fo = choreo.Compute_Auto_ODE_RHS(yo,callfun[0])

        for dt in dt_list:

            ya = yo + dt*dy
            fa = choreo.Compute_Auto_ODE_RHS(ya,callfun[0])
            yb = yo - dt*dy
            fb = choreo.Compute_Auto_ODE_RHS(yb,callfun[0])

            dfdy_difffin = (fa-fb)/(2*dt)

            dfdy_exact = choreo.Compute_Auto_JacMul_ODE_RHS(yo,dy,callfun[0])

            print('dt = ',dt)
            print('error = ',np.linalg.norm(dfdy_difffin-dfdy_exact))
            print('error_rel = ',np.linalg.norm(dfdy_difffin-dfdy_exact)/np.linalg.norm(dfdy_exact))
            print('')

        Jac_Linopt = choreo.Compute_Auto_JacMul_ODE_RHS_LinOpt(yo,callfun[0])
        Jac_Mat = choreo.Compute_Auto_JacMat_ODE_RHS(yo,callfun[0])

        df1 = Jac_Linopt.dot(dy)
        df2 = Jac_Mat.dot(dy)

        print('error = ',np.linalg.norm(df1-df2))
        print('error_rel = ',np.linalg.norm(df1-df2)/np.linalg.norm(df2))
        print('')




if __name__ == "__main__":
    main()    
