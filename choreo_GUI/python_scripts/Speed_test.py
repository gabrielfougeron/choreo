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
import json

import choreo 

def main():

    params_dict = js.ConfigDict.to_py()

    file_basename = ''
    
    CrashOnError_changevar = False

    LookForTarget = False
    
    n_make_loops = len(params_dict["Geom_Bodies"]["SymType"])

    nbpl = params_dict["Geom_Bodies"]["nbpl"]

    the_lcm = m.lcm(*nbpl)

    SymType = params_dict["Geom_Bodies"]["SymType"]

    Sym_list,nbody = choreo.Make2DChoreoSymManyLoops(nbpl=nbpl,SymType=SymType)

    mass = []
    for il in range(n_make_loops):
        mass.extend([params_dict["Geom_Bodies"]["mass"][il] for ib in range(nbpl[il])])

    mass = np.array(mass,dtype=np.float64)

    n_custom_sym = params_dict["Geom_Custom"]["n_custom_sym"]
    
    for isym in range(n_custom_sym):
        
        if (params_dict["Geom_Custom"]["CustomSyms"][isym]["Reflexion"] == "True"):
            s = -1
        elif (params_dict["Geom_Custom"]["CustomSyms"][isym]["Reflexion"] == "False"):
            s = 1
        else:
            raise ValueError("Reflexion must be True or False")
            
        rot_angle = (2*np.pi * params_dict["Geom_Custom"]["CustomSyms"][isym]["RotAngleNum"]) / params_dict["Geom_Custom"]["CustomSyms"][isym]["RotAngleDen"]

        if (params_dict["Geom_Custom"]["CustomSyms"][isym]["TimeRev"] == "True"):
            TimeRev = -1
        elif (params_dict["Geom_Custom"]["CustomSyms"][isym]["TimeRev"] == "False"):
            TimeRev = 1
        else:
            raise ValueError("TimeRev must be True or False")
        
        
        Sym_list.append(
            choreo.ChoreoSym(
                LoopTarget=params_dict["Geom_Custom"]["CustomSyms"][isym]["LoopTarget"],
                LoopSource=params_dict["Geom_Custom"]["CustomSyms"][isym]["LoopSource"],
                SpaceRot= np.array([[s*np.cos(rot_angle),-s*np.sin(rot_angle)],[np.sin(rot_angle),np.cos(rot_angle)]],dtype=np.float64),
                TimeRev=TimeRev,
                TimeShift=fractions.Fraction(
                    numerator=params_dict["Geom_Custom"]["CustomSyms"][isym]["TimeShiftNum"],
                    denominator=params_dict["Geom_Custom"]["CustomSyms"][isym]["TimeShiftDen"])
                ))



    MomConsImposed = params_dict['Geom_Bodies'] ['MomConsImposed']

    store_folder = 'Sniff_all_sym/'
    # store_folder = os.path.join(__PROJECT_ROOT__,'Sniff_all_sym/')
    store_folder = store_folder+str(nbody)
    if os.path.isdir(store_folder):
        shutil.rmtree(store_folder)
        os.makedirs(store_folder)
    else:
        os.makedirs(store_folder)

    # print("store_folder: ",store_folder)
    # print(os.path.isdir(store_folder))

    Use_exact_Jacobian = params_dict["Solver_Discr"]["Use_exact_Jacobian"]

    Look_for_duplicates = params_dict["Solver_Checks"]["Look_for_duplicates"]

    Check_Escape = params_dict["Solver_Checks"]["Check_Escape"]

    # Penalize_Escape = True
    Penalize_Escape = False

    # save_first_init = False
    save_first_init = True

    save_all_inits = False
    # save_all_inits = True

    # Save_img = True
    Save_img = False

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
#     nperiod_anim = 1.
    nperiod_anim = 1./period_div

    Plot_trace_anim = True
    # Plot_trace_anim = False

    # Save_Newton_Error = True
    Save_Newton_Error = False

    n_reconverge_it_max = params_dict["Solver_Discr"] ['n_reconverge_it_max'] 
    nint_init = params_dict["Solver_Discr"]["nint_init"]   

    disp_scipy_opt = (params_dict['Solver_Optim'] ['optim_verbose_lvl'] == "full")
    
    max_norm_on_entry = 1e20

    Newt_err_norm_max = params_dict["Solver_Optim"]["Newt_err_norm_max"]  
    Newt_err_norm_max_save = Newt_err_norm_max

    duplicate_eps = params_dict['Solver_Checks'] ['duplicate_eps'] 

    krylov_method = params_dict["Solver_Optim"]["krylov_method"]  

    line_search = params_dict["Solver_Optim"]["line_search"]  
    linesearch_smin = float(params_dict["Solver_Optim"]["line_search_smin"]  )

    gradtol_list =          params_dict["Solver_Loop"]["gradtol_list"]
    inner_maxiter_list =    params_dict["Solver_Loop"]["inner_maxiter_list"]
    maxiter_list =          params_dict["Solver_Loop"]["maxiter_list"]
    outer_k_list =          params_dict["Solver_Loop"]["outer_k_list"]
    store_outer_Av_list =   params_dict["Solver_Loop"]["store_outer_Av_list"]

    n_optim_param = len(gradtol_list)
    
    gradtol_max = gradtol_list[n_optim_param-1]
    foundsol_tol = 1e5

    escape_fac = 1e0

    escape_min_dist = 1
    escape_pow = 2.0

    n_grad_change = 1.

    coeff_ampl_min  = params_dict["Geom_Random"]["coeff_ampl_min"]
    coeff_ampl_o    = params_dict["Geom_Random"]["coeff_ampl_o"]
    k_infl          = params_dict["Geom_Random"]["k_infl"]
    k_max           = params_dict["Geom_Random"]["k_max"]

    freq_erase_dict = 100
    hash_dict = {}

    n_opt = 0
    n_opt_max = 100
    n_find_max = 1

    mul_coarse_to_fine = params_dict["Solver_Discr"]["mul_coarse_to_fine"]

    # Save_All_Coeffs = True
    Save_All_Coeffs = False

    # Save_Init_Pos_Vel_Sol = True
    Save_Init_Pos_Vel_Sol = False

    Save_All_Pos = True
    # Save_All_Pos = False

    plot_extend = 0.




    # n_test = 100000
    n_test = 200
# 
    all_kwargs = choreo.Pick_Named_Args_From_Dict(choreo.Speed_test,dict(globals(),**locals()))

    choreo.Speed_test(**all_kwargs)



# 
if __name__ == "__main__":
    main()
    

# if __name__ == "__main__":
# 
#     # n = 5
#     n = multiprocessing.cpu_count()
#     # n = 1
#     
#     print(f"Executing with {n} workers")
#     
#     with concurrent.futures.ProcessPoolExecutor(max_workers=n) as executor:
#         
#         res = []
#         for i in range(n):
#             res.append(executor.submit(main))
#             time.sleep(0.01)

 
