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

__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(__PROJECT_ROOT__)

import choreo 

def load_target_files(filename,Workspace_folder,target_speed):

    if (filename == "no file"):

        raise ValueError("A target file is missing")

    else:

        path_list = filename.split("/")

        path_beg = path_list.pop(0)
        path_end = path_list.pop( )

        if (path_beg == "Gallery"):

            json_filename = os.path.join(Workspace_folder,'Temp',target_speed+'.json')
            npy_filename  = os.path.join(Workspace_folder,'Temp',target_speed+'.npy' )

        elif (path_beg == "Workspace"):

            json_filename = os.path.join(Workspace_folder,*path_list,path_end+'.json')
            npy_filename  = os.path.join(Workspace_folder,*path_list,path_end+'.npy' )

        else:

             raise ValueError("Unknown path")

    with open(json_filename) as jsonFile:
        Info_dict = json.load(jsonFile)

    all_pos = np.load(npy_filename)

    return Info_dict, all_pos


def main():

    np.random.seed(int(time.time()*10000) % 5000)

    Workspace_folder = "Sniff_all_sym/"

    params_filename = os.path.join(Workspace_folder,"choreo_config.json")

    with open(params_filename) as jsonFile:
        params_dict = json.load(jsonFile)

    file_basename = ''
    
    CrashOnError_changevar = False

    LookForTarget = params_dict['Geom_Target'] ['LookForTarget']

    if (LookForTarget) :

        Rotate_fast_with_slow = params_dict['Geom_Target'] ['Rotate_fast_with_slow']
        Optimize_Init = params_dict['Geom_Target'] ['Optimize_Init']
        Randomize_Fast_Init =  params_dict['Geom_Target'] ['Randomize_Fast_Init']
            
        nT_slow = params_dict['Geom_Target'] ['nT_slow']
        nT_fast = params_dict['Geom_Target'] ['nT_fast']

        Info_dict_slow_filename = params_dict['Geom_Target'] ["slow_filename"]
        Info_dict_slow, all_pos_slow = load_target_files(Info_dict_slow_filename,Workspace_folder,"slow")

        all_coeffs_slow = choreo.AllPosToAllCoeffs(all_pos_slow,Info_dict_slow["n_int"],Info_dict_slow["n_Fourier"])
        choreo.Center_all_coeffs(all_coeffs_slow,Info_dict_slow["nloop"],Info_dict_slow["mass"],Info_dict_slow["loopnb"],np.array(Info_dict_slow["Targets"]),np.array(Info_dict_slow["SpaceRotsUn"]))

        Info_dict_fast_list = []
        all_coeffs_fast_list = []

        for i in range(len(nT_fast)) :

            Info_dict_fast_filename = params_dict['Geom_Target'] ["fast_filenames"] [i]
            Info_dict_fast, all_pos_fast = load_target_files(Info_dict_fast_filename,Workspace_folder,"fast"+str(i))
            Info_dict_fast_list.append(Info_dict_fast)

            all_coeffs_fast = choreo.AllPosToAllCoeffs(all_pos_fast,Info_dict_fast_list[i]["n_int"],Info_dict_fast_list[i]["n_Fourier"])
            choreo.Center_all_coeffs(all_coeffs_fast,Info_dict_fast_list[i]["nloop"],Info_dict_fast_list[i]["mass"],Info_dict_fast_list[i]["loopnb"],np.array(Info_dict_fast_list[i]["Targets"]),np.array(Info_dict_fast_list[i]["SpaceRotsUn"]))

            all_coeffs_fast_list.append(all_coeffs_fast)

        Sym_list, mass,il_slow_source,ibl_slow_source,il_fast_source,ibl_fast_source = choreo.MakeTargetsSyms(Info_dict_slow,Info_dict_fast_list)
        
        nbody = len(mass)

    else:

        n_make_loops = len(params_dict["Geom_Bodies"]["SymType"])

        nbpl = params_dict["Geom_Bodies"]["nbpl"]

        SymType = params_dict["Geom_Bodies"]["SymType"]

        Sym_list,nbody = choreo.Make2DChoreoSymManyLoops(nbpl=nbpl,SymType=SymType)

        mass = []
        for il in range(n_make_loops):
            mass.extend([params_dict["Geom_Bodies"]["mass"][il] for ib in range(nbpl[il])])

        mass = np.array(mass,dtype=np.float64)


    if ((LookForTarget) and not(params_dict['Geom_Target'] ['RandomJitterTarget'])) :

        coeff_ampl_min  = 1e-17
        coeff_ampl_o    = 1e-17
        k_infl          = 2
        k_max           = 3

    else:

        coeff_ampl_min  = params_dict["Geom_Random"]["coeff_ampl_min"]
        coeff_ampl_o    = params_dict["Geom_Random"]["coeff_ampl_o"]
        k_infl          = params_dict["Geom_Random"]["k_infl"]
        k_max           = params_dict["Geom_Random"]["k_max"]

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

    store_folder = os.path.join(Workspace_folder,str(nbody))
    if not(os.path.isdir(store_folder)):

        os.makedirs(store_folder)

    # print("store_folder: ",store_folder)
    # print(os.path.isdir(store_folder))

    Use_exact_Jacobian = params_dict["Solver_Discr"]["Use_exact_Jacobian"]

    Look_for_duplicates = params_dict["Solver_Checks"]["Look_for_duplicates"]

    Check_Escape = params_dict["Solver_Checks"]["Check_Escape"]

    # Penalize_Escape = True
    Penalize_Escape = False

    save_first_init = False
    # save_first_init = True

    save_all_inits = False
    # save_all_inits = True

    Save_img = True
    # Save_img = False

    # Save_thumb = True
    Save_thumb = False

    # img_size = (12,12) # Image size in inches
    img_size = (8,8) # Image size in inches
    thumb_size = (2,2) # Image size in inches
    
    color = params_dict["Animation_Colors"]["color_method_input"]

    color_list = params_dict["Animation_Colors"]["colorLookup"]

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

    nperiod_anim = 1.
    # nperiod_anim = 1./period_div

    Plot_trace_anim = True
    # Plot_trace_anim = False

    # Save_Newton_Error = True
    Save_Newton_Error = False

    n_reconverge_it_max = params_dict["Solver_Discr"] ['n_reconverge_it_max'] 
    ncoeff_init = params_dict["Solver_Discr"]["ncoeff_init"]   

    disp_scipy_opt =  (params_dict['Solver_Optim'] ['optim_verbose_lvl'] == "full")
    # disp_scipy_opt = False
    # disp_scipy_opt = True
    
    max_norm_on_entry = 1e20

    Newt_err_norm_max = params_dict["Solver_Optim"]["Newt_err_norm_max"]  
    # Newt_err_norm_max_save = Newt_err_norm_max*1000
    Newt_err_norm_max_save = 1e-9

    duplicate_eps = 1e-8

    krylov_method = params_dict["Solver_Optim"]["krylov_method"]  

    line_search = params_dict["Solver_Optim"]["line_search"]  
    linesearch_smin = params_dict["Solver_Optim"]["line_search_smin"]  

    gradtol_list =          params_dict["Solver_Loop"]["gradtol_list"]
    inner_maxiter_list =    params_dict["Solver_Loop"]["inner_maxiter_list"]
    maxiter_list =          params_dict["Solver_Loop"]["maxiter_list"]
    outer_k_list =          params_dict["Solver_Loop"]["outer_k_list"]
    store_outer_Av_list =   params_dict["Solver_Loop"]["store_outer_Av_list"]

    n_optim_param = len(gradtol_list)
    
    gradtol_max = 100*gradtol_list[n_optim_param-1]
    # foundsol_tol = 1000*gradtol_list[0]
    foundsol_tol = 1e10

    escape_fac = 1e0

    escape_min_dist = 1
    escape_pow = 2.0

    n_grad_change = 1.

    freq_erase_dict = 100

    n_opt = 0
    n_opt_max = 100
    n_find_max = 1

    mul_coarse_to_fine = 3

    # Save_All_Coeffs = True
    Save_All_Coeffs = False

    # Save_Init_Pos_Vel_Sol = True
    Save_Init_Pos_Vel_Sol = False

    n_save_pos = 'auto'
    Save_All_Pos = True
    # Save_All_Pos = False

    plot_extend = 0.

    n_opt = 0
    # n_opt_max = 1
    n_opt_max = 1e6
    n_find_max = 1e4

    ReconvergeSol = False
    AddNumberToOutputName = True
    
    all_kwargs = choreo.Pick_Named_Args_From_Dict(choreo.Find_Choreo,dict(globals(),**locals()))

    choreo.Find_Choreo(**all_kwargs)



if __name__ == "__main__":
# 
    Exec_Mul_Proc = True
    # Exec_Mul_Proc = False

    # n = 5
    # n = multiprocessing.cpu_count()
    n = multiprocessing.cpu_count() // 2
    # n = 1
    
    if Exec_Mul_Proc:

        print(f"Executing with {n} workers")
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=n) as executor:
            
            res = []
            for i in range(n):
                res.append(executor.submit(main))
                time.sleep(0.01)

    else :

        main()


