import os

import shutil
import random
import time
import math as m
import numpy as np
import sys
import fractions
import json
import choreo 
import asyncio


import js
import pyodide

def NPY_JS_to_py(npy_js):

    return np.asarray(npy_js["data"]).reshape(npy_js["shape"])

async def main(params_dict):

    geodim = 2

    TwoDBackend = (geodim == 2)
    ParallelBackend = False
    GradHessBackend = 'Cython'
    
    file_basename = ''
    
    CrashOnError_changevar = False

    LookForTarget = params_dict['Phys_Target'] ['LookForTarget']


    if (LookForTarget) : # IS LIKELY BROKEN !!!!

        Rotate_fast_with_slow = params_dict['Phys_Target'] ['Rotate_fast_with_slow']
        Optimize_Init = params_dict['Phys_Target'] ['Optimize_Init']
        Randomize_Fast_Init =  params_dict['Phys_Target'] ['Randomize_Fast_Init']
            
        nT_slow = params_dict['Phys_Target'] ['nT_slow']
        nT_fast = params_dict['Phys_Target'] ['nT_fast']

        Info_dict_slow = js.TargetSlow_PlotInfo.to_py()

        ncoeff_slow = Info_dict_slow["n_int"] // 2 + 1

        all_pos_slow = NPY_JS_to_py(js.TargetSlow_Pos.to_py())
        all_coeffs_slow = choreo.AllPosToAllCoeffs(all_pos_slow,ncoeff_slow)
        choreo.Center_all_coeffs(all_coeffs_slow,Info_dict_slow["nloop"],Info_dict_slow["mass"],Info_dict_slow["loopnb"],np.array(Info_dict_slow["Targets"]),np.array(Info_dict_slow["SpaceRotsUn"]))

        Info_dict_fast_list = js.TargetFast_PlotInfoList.to_py()
        all_pos_fast_js_list = js.TargetFast_PosList.to_py()
        all_coeffs_fast_list = []


        for (i,all_pos_fast_js) in enumerate(all_pos_fast_js_list) :

            ncoeff_fast = Info_dict_fast_list[i]["n_int"] // 2 + 1

            all_pos_fast = NPY_JS_to_py(all_pos_fast_js)
            all_coeffs_fast = choreo.AllPosToAllCoeffs(all_pos_fast,ncoeff_fast)
            choreo.Center_all_coeffs(all_coeffs_fast,Info_dict_fast_list[i]["nloop"],Info_dict_fast_list[i]["mass"],Info_dict_fast_list[i]["loopnb"],np.array(Info_dict_fast_list[i]["Targets"]),np.array(Info_dict_fast_list[i]["SpaceRotsUn"]))

            all_coeffs_fast_list.append(all_coeffs_fast)

        Sym_list, mass,il_slow_source,ibl_slow_source,il_fast_source,ibl_fast_source = choreo.MakeTargetsSyms(Info_dict_slow,Info_dict_fast_list)
        
        nbody = len(mass)


    nbody = params_dict["Phys_Bodies"]["nbody"]
    MomConsImposed = params_dict['Phys_Bodies'] ['MomConsImposed']
    nsyms = params_dict["Phys_Bodies"]["nsyms"]

    nloop = params_dict["Phys_Bodies"]["nloop"]
    mass = np.zeros(nbody)

    for il in range(nloop):
        for ilb in range(len(params_dict["Phys_Bodies"]["Targets"][il])):
            
            ib = params_dict["Phys_Bodies"]["Targets"][il][ilb]
            mass[ib] = params_dict["Phys_Bodies"]["mass"][il]

    Sym_list = []

    if (geodim == 2):

        for isym in range(nsyms):
            
            BodyPerm = np.array(params_dict["Phys_Bodies"]["AllSyms"][isym]["BodyPerm"],dtype=int)

            Reflexion = params_dict["Phys_Bodies"]["AllSyms"][isym]["Reflexion"]
            if (Reflexion == "True"):
                s = -1
            elif (Reflexion == "False"):
                s = 1
            else:
                raise ValueError("Reflexion must be True or False")

            rot_angle = 2 * np.pi * params_dict["Phys_Bodies"]["AllSyms"][isym]["RotAngleNum"] / params_dict["Phys_Bodies"]["AllSyms"][isym]["RotAngleDen"]

            SpaceRot = np.array(
                [   [ s*np.cos(rot_angle)   , -s*np.sin(rot_angle)  ],
                    [ np.sin(rot_angle)     , np.cos(rot_angle)     ]   ]
                , dtype = np.float64
            )

            TimeRev_str = params_dict["Phys_Bodies"]["AllSyms"][isym]["TimeRev"]
            if (TimeRev_str == "True"):
                TimeRev = -1
            elif (TimeRev_str == "False"):
                TimeRev = 1
            else:
                raise ValueError("TimeRev must be True or False")

            TimeShiftNum = params_dict["Phys_Bodies"]["AllSyms"][isym]["TimeShiftNum"]
            TimeShiftDen = params_dict["Phys_Bodies"]["AllSyms"][isym]["TimeShiftDen"]

            Sym_list.append(
                choreo.ActionSym(
                    BodyPerm = BodyPerm     ,
                    SpaceRot = SpaceRot     ,
                    TimeRev = TimeRev       ,
                    TimeShiftNum = TimeShiftNum   ,
                    TimeShiftDen = TimeShiftDen   ,
                )
            )


    else:

        raise ValueError("Only compatible with 2D right now")


    Sym_list = choreo.Make_ChoreoSymList_From_ActionSymList(Sym_list, nbody)


    if ((LookForTarget) and not(params_dict['Phys_Target'] ['RandomJitterTarget'])) :

        coeff_ampl_min  = 0
        coeff_ampl_o    = 0
        k_infl          = 2
        k_max           = 3

    else:

        coeff_ampl_min  = params_dict["Phys_Random"]["coeff_ampl_min"]
        coeff_ampl_o    = params_dict["Phys_Random"]["coeff_ampl_o"]
        k_infl          = params_dict["Phys_Random"]["k_infl"]
        k_max           = params_dict["Phys_Random"]["k_max"]


    store_folder = 'Sniff_all_sym/'

    store_folder = store_folder+str(nbody)
    if os.path.isdir(store_folder):
        shutil.rmtree(store_folder)
        os.makedirs(store_folder)
    else:
        os.makedirs(store_folder)

    Use_exact_Jacobian = params_dict["Solver_Discr"]["Use_exact_Jacobian"]

    Look_for_duplicates = params_dict["Solver_Checks"]["Look_for_duplicates"]

    Check_Escape = params_dict["Solver_Checks"]["Check_Escape"]

    # Penalize_Escape = True
    Penalize_Escape = False

    save_first_init = False
    # save_first_init = True

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

    nperiod_anim = 1.

    Plot_trace_anim = True
    # Plot_trace_anim = False

    # Save_Newton_Error = True
    Save_Newton_Error = False

    n_reconverge_it_max = params_dict["Solver_Discr"] ['n_reconverge_it_max'] 

    nint_init = params_dict["Solver_Discr"]["nint_init"]   

    disp_scipy_opt = False
    # disp_scipy_opt = True
    
    max_norm_on_entry = 1e20

    Newt_err_norm_max = params_dict["Solver_Optim"]["Newt_err_norm_max"]  
    Newt_err_norm_max_save = params_dict["Solver_Optim"]["Newt_err_norm_safe"]  

    duplicate_eps =  params_dict['Solver_Checks'] ['duplicate_eps'] 

    krylov_method = params_dict["Solver_Optim"]["krylov_method"]  

    line_search = params_dict["Solver_Optim"]["line_search"]  

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
    hash_dict = {}

    n_opt = 0
    n_opt_max = 1

    mul_coarse_to_fine = params_dict["Solver_Discr"]["mul_coarse_to_fine"]

    # Save_All_Coeffs = True
    Save_All_Coeffs = False

    # Save_Init_Pos_Vel_Sol = True
    Save_Init_Pos_Vel_Sol = False

    n_save_pos = 'auto'
    Save_All_Pos = True
    # Save_All_Pos = False

    plot_extend = 0.0
    
    all_kwargs = choreo.Pick_Named_Args_From_Dict(choreo.GenSymExample,dict(globals(),**locals()))

    success = choreo.GenSymExample(**all_kwargs)

    filename = 'init.json'

    if (success and os.path.isfile(filename)):

        with open(filename, 'rt') as fh:
            thefile = fh.read()
            
        blob = js.Blob.new([thefile], {type : 'application/text'})

        filename = 'init.npy'
        all_pos = np.load(filename)

        js.postMessage(
            funname = "Play_Loop_From_Python",
            args    = pyodide.ffi.to_js(
                {
                    "is_sol":False,
                    "solname":"Non-solution initial state",
                    "JSON_data":blob,
                    "NPY_data":all_pos.reshape(-1),
                    "NPY_shape":all_pos.shape,
                    "DoClearScreen":True,
                    "DoXMinMax":True,
                    "ResetRot":True,
                },
                dict_converter=js.Object.fromEntries
            )
        )

        print("Non solution initial state playing.\n")

    else :

        print("No valid initial state generated.\n")

        js.postMessage(
            funname = "Python_no_sol_found",
            args    = pyodide.ffi.to_js(
                {
                },
                dict_converter=js.Object.fromEntries
            )
        )

async def main_new(params_dict):

    extra_args_dict = {}

    store_folder = '/Workspace/GUI solutions'

    if not(os.path.isdir(store_folder)):

        store_folder = 'Sniff_all_sym'

        if os.path.isdir(store_folder):
            shutil.rmtree(store_folder)
            os.makedirs(store_folder)
        else:
            os.makedirs(store_folder)

    file_basename = ''
    params_dict["Solver_Optim"]["n_opt"] = 0

    Workspace_folder = '/Workspace'
    extra_args_dict['store_folder'] = store_folder
    extra_args_dict['file_basename'] = file_basename
    extra_args_dict['save_first_init'] = True
    extra_args_dict['Save_SegmPos'] = True
    
    choreo.find_new.ChoreoChooseParallelEnvAndFind(Workspace_folder, params_dict, extra_args_dict)

    filename_output = store_folder+'/_init'
    filename = filename_output+".json"
    
    if os.path.isfile(filename):

        with open(filename, 'rt') as fh:
            thefile = fh.read()
        os.remove(filename)
        
        blob = js.Blob.new([thefile], {type : 'application/text'})

        filename = filename_output+'.npy'
        all_pos = np.load(filename)
        os.remove(filename)

        js.postMessage(
            funname = "Play_Loop_From_Python",
            args    = pyodide.ffi.to_js(
                {
                    "is_sol":False,
                    "solname":"Non-solution initial state",
                    "JSON_data":blob,
                    "NPY_data":all_pos.reshape(-1),
                    "NPY_shape":all_pos.shape,
                    "DoClearScreen":True,
                    "DoXMinMax":True,
                    "ResetRot":True,
                },
                dict_converter=js.Object.fromEntries
            )
        )

    else:

        js.postMessage(
            funname = "Python_no_sol_found",
            args    = pyodide.ffi.to_js(
                {
                },
                dict_converter=js.Object.fromEntries
            )
        )
        
        
if __name__ == "__main__":
    
    params_dict = js.ConfigDict.to_py()
    
    if params_dict['Solver_CLI']['GUI_backend'] == "New":
        asyncio.create_task(main_new(params_dict))
    elif params_dict['Solver_CLI']['GUI_backend'] == "Old":
        asyncio.create_task(main(params_dict))


