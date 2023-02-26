import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'

import shutil
import asyncio
import random
import time
import math as m
import numpy as np
import sys
import fractions
import json

import choreo 

import js
import pyodide

def Send_init_PlotInfo():

    filename = "init.json"

    if os.path.isfile(filename):

        with open(filename, 'rt') as fh:
            thefile = fh.read()
            
        blob = js.Blob.new([thefile], {type : 'application/text'})

        js.postMessage(
            funname = "Set_PlotInfo_From_Python",
            args    = pyodide.ffi.to_js(
                {
                    "JSON_data":blob,
                },
                dict_converter=js.Object.fromEntries
            )
        )


def Plot_Loops_During_Optim(x,f,f_norm,ActionSyst):

    xmin,xmax,ymin,ymax = ActionSyst.HeuristicMinMax()

    hside = max(xmax-xmin,ymax-ymin)/2

    xmid = (xmin+xmax)/2
    ymid = (ymin+ymax)/2

    windowObject = {}

    windowObject["xMin"] = xmid - hside
    windowObject["xMax"] = xmid + hside

    windowObject["yMin"] = ymid - hside
    windowObject["yMax"] = ymid + hside

    js.postMessage(

        funname = "Plot_Loops_During_Optim_From_Python",
        args    = pyodide.ffi.to_js(
            {
                "NPY_data":ActionSyst.last_all_pos.reshape(-1),
                "NPY_shape":ActionSyst.last_all_pos.shape,
                "Current_PlotWindow":windowObject
            },
            dict_converter=js.Object.fromEntries
        )
    )

def ListenToNextFromGUI(x,f,f_norm,ActionSyst):

    AskForNext =  (js.AskForNext.to_py()[0] == 1)

    js.AskForNext[0] = 0

    return AskForNext

def NPY_JS_to_py(npy_js):

    return np.asarray(npy_js["data"]).reshape(npy_js["shape"])



async def main():

    params_dict = js.ConfigDict.to_py()

    geodim = 2
    
    CrashOnError_changevar = False

    LookForTarget = params_dict['Geom_Target'] ['LookForTarget']

    if (LookForTarget) :

        Rotate_fast_with_slow = params_dict['Geom_Target'] ['Rotate_fast_with_slow']
        Optimize_Init = params_dict['Geom_Target'] ['Optimize_Init']
        Randomize_Fast_Init =  params_dict['Geom_Target'] ['Randomize_Fast_Init']
            
        nT_slow = params_dict['Geom_Target'] ['nT_slow']
        nT_fast = params_dict['Geom_Target'] ['nT_fast']

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
    dnint = 30

    nint_plot_img = nint_plot_anim * dnint

    nperiod_anim = 1.

    Plot_trace_anim = True
    # Plot_trace_anim = False

    # Save_Newton_Error = True
    Save_Newton_Error = False

    n_reconverge_it_max = params_dict["Solver_Discr"] ['n_reconverge_it_max'] 
    nint_init = params_dict["Solver_Discr"]["nint_init"]   

    disp_scipy_opt = (params_dict['Solver_Optim'] ['optim_verbose_lvl'] == "full")
    
    max_norm_on_entry = 1e20

    Newt_err_norm_max = params_dict["Solver_Optim"]["Newt_err_norm_max"]  
    Newt_err_norm_max_save = params_dict["Solver_Optim"]["Newt_err_norm_safe"]  

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

    ReconvergeSol = False
    AddNumberToOutputName = False
    
    callback_after_init_list = []

    if params_dict['Animation_Search']['DisplayLoopsDuringSearch']:
        callback_after_init_list.append(Send_init_PlotInfo)

    optim_callback_list = [ListenToNextFromGUI]

    if params_dict['Animation_Search']['DisplayLoopsDuringSearch']:

        optim_callback_list.append(Plot_Loops_During_Optim)

    # await SyncDiskPromise

    store_folder = '/Workspace/GUI solutions'

    if not(os.path.isdir(store_folder)):

        store_folder = 'Sniff_all_sym/'

        if os.path.isdir(store_folder):
            shutil.rmtree(store_folder)
            os.makedirs(store_folder)
        else:
            os.makedirs(store_folder)

    file_basename = ''

    max_num_file = 0
    
    for filename in os.listdir(store_folder):
        file_path = os.path.join(store_folder, filename)
        file_root, file_ext = os.path.splitext(os.path.basename(file_path))
        
        if (file_basename in file_root) and (file_ext == '.json' ):

            file_root = file_root.replace(file_basename,"")

            try:
                max_num_file = max(max_num_file,int(file_root))
            except:
                pass

    max_num_file = max_num_file + 1

    file_basename = file_basename+str(max_num_file).zfill(5)

    all_kwargs = choreo.Pick_Named_Args_From_Dict(choreo.Find_Choreo,dict(globals(),**locals()))

    choreo.Find_Choreo(**all_kwargs)

    filename_output = store_folder+'/'+file_basename
    filename = filename_output+".json"

    if os.path.isfile(filename):

        with open(filename, 'rt') as fh:
            thefile = fh.read()

        # os.remove(filename)
            
        blob = js.Blob.new([thefile], {type : 'application/text'})

        filename = filename_output+'.npy'
        all_pos = np.load(filename)

        # os.remove(filename)

        js.postMessage(
            funname = "Play_Loop_From_Python",
            args    = pyodide.ffi.to_js(
                {
                    "is_sol":True,
                    "solname":"User generated solution: "+file_basename,
                    "JSON_data":blob,
                    "NPY_data":all_pos.reshape(-1),
                    "NPY_shape":all_pos.shape,
                    "DoClearScreen":not(params_dict['Animation_Search']['DisplayLoopsDuringSearch']),
                    "DoXMinMax":not(params_dict['Animation_Search']['DisplayLoopsDuringSearch']),
                    "ResetRot":False,
                },
                dict_converter=js.Object.fromEntries
            )
        )

    else:

        print("Solver did not find a solution.")

        js.postMessage(
            funname = "Python_no_sol_found",
            args    = pyodide.ffi.to_js(
                {
                },
                dict_converter=js.Object.fromEntries
            )
        )

if __name__ == "__main__":
    asyncio.create_task(main())
