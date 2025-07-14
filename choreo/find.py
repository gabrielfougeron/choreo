import sys
import os
try:
    import concurrent.futures
except:
    pass

import math as m
import numpy as np
import scipy
import json
import time
import inspect
import threadpoolctl

import choreo.metadata
import choreo.scipy_plus
from choreo.cython import ActionSym

import warnings 
from choreo.optional_pyfftw import PYFFTW_AVAILABLE
if PYFFTW_AVAILABLE:
    import pyfftw

def Find_Choreo(
    *,
    geodim,
    nbody,
    mass,
    charge,
    inter_law ,
    inter_law_str,
    inter_law_params,
    Sym_list,
    nint_fac_init,
    coeff_ampl_o,
    k_infl,
    k_max,
    coeff_ampl_min,
    n_opt_max,
    n_find_max,
    save_first_init,
    Look_for_duplicates,
    Duplicates_Hash,
    store_folder,
    freq_erase_dict,
    ReconvergeSol,
    segmpos_ini,
    LookForTarget,
    save_all_inits,
    file_basename,
    AddNumberToOutputName,
    Save_img,
    Save_thumb,
    Save_anim,
    plot_Mass_Scale,
    plot_body_size,
    plot_trail_width,
    callback_after_init_list,
    optim_callback_list,
    max_norm_on_entry,
    gradtol_list,
    inner_maxiter_list,
    maxiter_list,
    outer_k_list,
    store_outer_Av_list,
    n_optim_param,
    krylov_method,
    SpectralSolve,
    rk_explicit,
    rk_implicit_x,
    rk_implicit_v,
    pos_mom_scaling,
    init_pos_BB_size,
    init_mom_BB_size,
    Use_exact_Jacobian,
    disp_scipy_opt,
    line_search,
    linesearch_smin,
    Check_Escape,
    Newt_err_norm_max,
    Newt_err_norm_max_save,
    n_reconverge_it_max,
    plot_extend,
    mul_coarse_to_fine,
    duplicate_eps,
    Save_SegmPos,
    Save_Params,
    img_size,
    thumb_size,
    color,
    color_list,
    fftw_planner_effort,
    fftw_wisdom_only,
    fftw_nthreads,
    fft_backend,
):
    """

    Finds periodic solutions

    """
    
    NBS = choreo.NBodySyst(geodim, nbody, mass, charge, Sym_list, inter_law, inter_law_str, inter_law_params)

    NBS.fftw_planner_effort = fftw_planner_effort
    NBS.fftw_wisdom_only = fftw_wisdom_only
    NBS.fftw_nthreads = fftw_nthreads
    NBS.fft_backend = fft_backend

    NBS.nint_fac = nint_fac_init

    print(NBS.DescribeSystem())
    
    if SpectralSolve:
        print("Searching for periodic solutions using the spectral solver")
    else:
        print("Searching for periodic solutions using the Runge-Kutta solver")

    print()

    if SpectralSolve:
        x_min, x_max = NBS.Make_params_bounds(coeff_ampl_o, k_infl, k_max, coeff_ampl_min)
        x_ptp = x_max - x_min
        
        del x_max
        
    else:

        NBS.ODEperdef_eqproj_pos_mul = pos_mom_scaling
        
        x_ptp_x = np.ones(NBS.n_ODEinitparams_pos, dtype=np.float64) * init_pos_BB_size
        x_ptp_v = np.ones(NBS.n_ODEinitparams_mom, dtype=np.float64) * init_mom_BB_size
        
        x_ptp = np.ascontiguousarray(np.concatenate((x_ptp_x, x_ptp_v)))
        x_min = (-0.5)*x_ptp

    n_opt = 0
    n_find = 0

    if optim_callback_list is None:
        optim_callback_list = []
    
    n_optim_callback_list = len(optim_callback_list)    

    if callback_after_init_list is None:
        callback_after_init_list = []
    n_callback_after_init_list = len(callback_after_init_list)
    
    ForceFirstEntry = save_first_init
    
    hash_dict = {}
    action_dict = {}
    
    if not SpectralSolve:

        NBS.setup_params_to_periodicity_default(rk_explicit, rk_implicit_x, rk_implicit_v)

        # TODO remove this and adapt Choose_Init_ODE_params
        Implicit = (rk_explicit is None)
        ODE_Syst = NBS.Get_ODE_def(vector_calls = Implicit)
        if not Implicit:
            ODE_Syst.pop('vector_calls', None)
        
        min_size_fac = 0.2
        
        def Choose_Init_ODE_params():
            
            ODE_params = x_min + x_ptp * np.random.random((NBS.n_ODEinitparams))
            xo, vo = NBS.ODE_params_to_initposmom(ODE_params)
            
            fac = 10
            nint_ODE = fac * (NBS.segm_store-1)

            if Implicit:
            
                segmpos_ODE, segmmom_ODE = choreo.segm.ODE.ImplicitSymplecticIVP(
                    xo = xo                 ,
                    vo = vo                 ,
                    rk_x = rk_implicit_x    ,
                    rk_v = rk_implicit_v    ,
                    nint = nint_ODE         ,
                    keep_freq = 1           ,
                    keep_init = False       ,
                    **ODE_Syst              ,
                )
            
            else:
                
                segmpos_ODE, segmmom_ODE = choreo.segm.ODE.ExplicitSymplecticIVP(
                    xo = xo                 ,
                    vo = vo                 ,
                    rk = rk_explicit        ,
                    nint = nint_ODE         ,
                    keep_freq = 1           ,
                    keep_init = False       ,
                    **ODE_Syst              ,
                )
            
            ODEperdef = NBS.endposmom_to_perdef_bulk(xo, vo, segmpos_ODE, segmmom_ODE)
            
            min_size = int(min_size_fac * ODEperdef.shape[0])
            
            NBS.scale_ODEperdef_lin(ODEperdef)
            ODEperdef_norm = np.linalg.norm(ODEperdef[min_size:], axis=1)
            
            i_min = np.argmin(ODEperdef_norm) + min_size
            
            # Rescale period
            T = (i_min+1) / nint_ODE
            
            NBS.scale_init_period(T, xo, vo)
            ODEparams_ini = NBS.initposmom_to_ODE_params(xo, vo)
            
            return ODEparams_ini

    while (((n_opt < n_opt_max) and (n_find < n_find_max)) or ForceFirstEntry):
        
        NBS.nint_fac = nint_fac_init

        ForceFirstEntry = False
        AskedForNext = False

        if (Look_for_duplicates and ((n_opt % freq_erase_dict) == 0)):

            hash_dict = {}
            action_dict = {}
            UpdateHashDict(store_folder, hash_dict, action_dict)

        if (ReconvergeSol):
            
            assert segmpos_ini.shape[0] == NBS.nsegm
            assert segmpos_ini.shape[1] >= NBS.segm_store
            assert segmpos_ini.shape[2] == NBS.geodim

            segmpos = segmpos_ini[:,0:NBS.segm_store,:].copy()
            x = NBS.segmpos_to_params(segmpos)
            
        elif (LookForTarget):
            raise NotImplementedError

            # all_coeffs_avg = ActionSyst.Gen_init_avg_2D(nT_slow,nT_fast,Info_dict_slow,all_coeffs_slow,Info_dict_fast_list,all_coeffs_fast_list,il_slow_source,ibl_slow_source,il_fast_source,ibl_fast_source,Rotate_fast_with_slow,Optimize_Init,Randomize_Fast_Init)

            # x_min = ActionSyst.Package_all_coeffs(all_coeffs_avg)
        else:

            if SpectralSolve:
                x = x_min + x_ptp * np.random.random((NBS.nparams))
                segmpos = NBS.params_to_segmpos(x)
                f0 = NBS.segmpos_params_to_action_grad(segmpos, x)
                spectral_params = x
            else:
                # x = x_min + x_ptp * np.random.random((NBS.n_ODEinitparams))
                x = Choose_Init_ODE_params()
                f0 = NBS.params_to_periodicity_default(x)
                segmpos = NBS.segmpos.copy()
                spectral_params = NBS.segmpos_to_params(segmpos)
        
        if save_all_inits or (save_first_init and n_opt == 0):

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

            if (AddNumberToOutputName):   
                filename_output = os.path.join(store_folder, file_basename+'_init_'+str(max_num_file).zfill(5))

            else:
                filename_output = os.path.join(store_folder, file_basename+'_init')

            NBS.Write_Descriptor(params_mom_buf=spectral_params, segmpos=segmpos, filename=filename_output+'.json')

            if Save_img :
                NBS.plot_segmpos_2D(segmpos, filename_output+'.png', fig_size=img_size, color=color, color_list=color_list)     

            if Save_thumb :
                NBS.plot_segmpos_2D(segmpos, filename_output+'_thumb.png', fig_size=thumb_size, color=color, color_list=color_list)     
                
            if Save_anim :
                allbodypos = NBS.segmpos_to_allbody_noopt(segmpos)
                NBS.plot_all_2D_anim(allbodypos, filename_output+'.mp4', fig_size=img_size, color=color, color_list=color_list, Mass_Scale = plot_Mass_Scale, body_size = plot_body_size, trail_width = plot_trail_width)    
                
            if Save_SegmPos:
                np.save(filename_output+'.npy', segmpos)
            
            if Save_Params:
                np.save(filename_output+'_params.npy', spectral_params)

        best_sol = choreo.scipy_plus.nonlin.current_best(x, f0)

        if m.isnan(best_sol.f_norm):
            raise ValueError(f"Norm on entry is {best_sol.f_norm:.2e} which indicates a problem with constraints.")
            
        # print(f"Norm on entry is {best_sol.f_norm:.2e}")
            
        GoOn = (best_sol.f_norm < max_norm_on_entry)
        
        if not(best_sol.f_norm < max_norm_on_entry):
            print(f"Norm on entry is {best_sol.f_norm:.2e} which is too big.")
        
        for i in range(n_callback_after_init_list):
            callback_after_init_list[i]()
        
        i_optim_param = 0
        current_cvg_lvl = 0 
        n_opt += 1
        
        GoOn = GoOn and (n_opt <= n_opt_max)
        
        if (n_opt <= n_opt_max):
            print(f'Optimization attempt number: {n_opt}')
        # else:
        #     print('Reached max number of optimization attempts')

        while GoOn:
            # Set correct optim params
            
            x = np.copy(best_sol.x)
            
            inner_tol = 0.
            
            rdiff = None
            gradtol = gradtol_list[i_optim_param]
            inner_maxiter = inner_maxiter_list[i_optim_param]
            maxiter = maxiter_list[i_optim_param]
            outer_k = outer_k_list[i_optim_param]
            store_outer_Av = store_outer_Av_list[i_optim_param]

            ActionGradNormEnterLoop = best_sol.f_norm
            
            print(f'Action Grad Norm on entry: {ActionGradNormEnterLoop:.2e}')
            print(f'Optim level: {i_optim_param+1} / {n_optim_param}    Resize level: {current_cvg_lvl+1} / {n_reconverge_it_max+1}')
            
            inner_M = None

            if (krylov_method == 'lgmres'):
                jac_options = {'method':krylov_method,'rdiff':rdiff,'outer_k':outer_k,'inner_inner_m':inner_maxiter,'inner_store_outer_Av':store_outer_Av,'inner_tol':inner_tol,'inner_M':inner_M }
            elif (krylov_method == 'gmres'):
                jac_options = {'method':krylov_method,'rdiff':rdiff,'outer_k':outer_k,'inner_tol':inner_tol,'inner_M':inner_M }
            else:
                jac_options = {'method':krylov_method,'rdiff':rdiff,'outer_k':outer_k,'inner_tol':inner_tol,'inner_M':inner_M }

            jacobian = NBS.GetKrylovJacobian(Use_exact_Jacobian, SpectralSolve, jac_options)

            def optim_callback(x,f,f_norm):

                AskedForNext = False
                best_sol.update(x,f,f_norm)

                for i in range(n_optim_callback_list):
                    AskedForNext = (AskedForNext or optim_callback_list[i](x,f,f_norm,NBS,jacobian))

                return AskedForNext

            if SpectralSolve:
                F = NBS.params_to_action_grad
            else:
                F = NBS.params_to_periodicity_default

            try : 
                
                opt_result, info = choreo.scipy_plus.nonlin.nonlin_solve_pp(
                    F=F, x0=x, jacobian=jacobian, 
                    verbose=disp_scipy_opt, maxiter=maxiter, f_tol=gradtol,  line_search=line_search, callback=optim_callback, raise_exception=False, smin=linesearch_smin, full_output=True, tol_norm=np.linalg.norm)
                AskedForNext = (info['status'] == 0)

            except Exception as exc:
                
                print(exc)
                GoOn = False
                raise(exc)

            if (AskedForNext):
                print("Skipping at user's request")
                GoOn = False

            SaveSol = False

            if SpectralSolve:
                segmpos = NBS.params_to_segmpos(best_sol.x)
                spectral_params = best_sol.x
            else:
                NBS.params_to_periodicity_default(best_sol.x)
                segmpos = NBS.segmpos
                spectral_params = NBS.segmpos_to_params(segmpos)
                
            Hash_Action = NBS.segmpos_to_hash(segmpos)
            
            print(f'Opt Action Grad Norm: {best_sol.f_norm:.2e}')
            
            if (GoOn and Check_Escape):
                
                Escaped = NBS.DetectEscape(segmpos)

                if Escaped:
                    print('One loop escaped. Starting over.')    
                    
                GoOn = GoOn and not(Escaped)
                
            if (GoOn and Look_for_duplicates):
                
                Found_duplicate, file_path = Check_Duplicates(NBS, segmpos, spectral_params, hash_dict, action_dict, store_folder, duplicate_eps, Hash_Action=Hash_Action, Duplicates_Hash=Duplicates_Hash)
                
                if (Found_duplicate):
                
                    print('Found Duplicate!')   
                    print('Path: ',file_path)
                    
                GoOn = GoOn and not(Found_duplicate)
                
            if (GoOn):
                
                nint_fac_cur = NBS.nint_fac
                nint_fac = 2*nint_fac_cur
                
                if SpectralSolve:
                    x_fine = NBS.params_resize(spectral_params, nint_fac)
                    NBS.nint_fac = nint_fac
                    f_fine = NBS.params_to_action_grad(x_fine)
                    f_fine_norm = np.linalg.norm(f_fine)
                else:
                    NBS.nint_fac = nint_fac
                    f_fine = NBS.params_to_periodicity_default(best_sol.x)
                    f_fine_norm = np.linalg.norm(f_fine)
                
                print(f'Opt Action Grad Norm Refine : {f_fine_norm:.2e}')
                
                ParamsDivergence = (f_fine_norm > max_norm_on_entry)
                ParamPreciseEnough = (f_fine_norm < Newt_err_norm_max)
                ParamPreciseEnoughSave = (f_fine_norm < Newt_err_norm_max_save)
                CanChangeOptimParams = i_optim_param < (n_optim_param-1)
                CanRefine = (current_cvg_lvl < n_reconverge_it_max)
                NeedsRefinement = (f_fine_norm > mul_coarse_to_fine*best_sol.f_norm) and (f_fine_norm > ParamPreciseEnoughSave)
                OnCollisionCourse = (best_sol.f_norm < 1e3*Newt_err_norm_max) and (f_fine_norm > 1e6 * best_sol.f_norm) 
                
                NBS.nint_fac = nint_fac_cur

                if GoOn and ParamsDivergence:
                
                    GoOn = False
                    print('Stopping search: solver is diverging.')                    

                if GoOn and ParamPreciseEnough and not(NeedsRefinement):

                    GoOn = False
                    print("Stopping search: found solution.")
                    SaveSol = True

                if GoOn and ParamPreciseEnoughSave and not(CanChangeOptimParams) and (not(CanRefine) or not(NeedsRefinement)) :

                    GoOn = False
                    print("Stopping search: found approximate solution.")
                    SaveSol = True

                if GoOn and (not(CanRefine) or not(NeedsRefinement)) and not(CanChangeOptimParams):
                
                    GoOn = False
                    print('Stopping search: could not converge within prescibed optimizer and refinement parameters.')
                    
                if GoOn and (OnCollisionCourse):
                
                    GoOn = False
                    print('Stopping search: solver is likely narrowing in on a collision solution.')

                if SaveSol :
                    
                    GoOn  = False

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
                    n_find = max_num_file

                    if (AddNumberToOutputName):   
                        filename_output = os.path.join(store_folder, file_basename+str(max_num_file).zfill(5))
                    else:
                        filename_output = os.path.join(store_folder, file_basename)

                    print(f'Saving solution as {filename_output}.*.')
                    
                    xo = NBS.ComputeCenterOfMass(segmpos)
                    for idim in range(NBS.geodim):
                        segmpos[:,:,idim] -= xo[idim]
                        
                    spectral_params = NBS.segmpos_to_params(segmpos)
                        
                    NBS.Write_Descriptor(params_mom_buf=spectral_params , filename = filename_output+'.json', segmpos=segmpos, Gradaction=f_fine_norm, Hash_Action=Hash_Action, extend=plot_extend)

                    if Save_img :
                        NBS.plot_segmpos_2D(segmpos, filename_output+'.png', fig_size=img_size, color=color, color_list=color_list)
                    
                    if Save_thumb :
                        NBS.plot_segmpos_2D(segmpos, filename_output+'_thumb.png', fig_size=thumb_size, color=color, color_list=color_list)     
                    
                    if Save_anim :
                        allbodypos = NBS.segmpos_to_allbody_noopt(segmpos)
                        NBS.plot_all_2D_anim(allbodypos, filename_output+'.mp4', fig_size=img_size, color=color, color_list=color_list, Mass_Scale = plot_Mass_Scale, body_size = plot_body_size, trail_width = plot_trail_width)     

                    if Save_SegmPos:
                        np.save(filename_output+'.npy', segmpos)

                    if Save_Params:
                        np.save(filename_output+'_params.npy', spectral_params)
                    
#                     if Save_Init_Pos_Vel_Sol:
#                         all_pos_b = ActionSyst.Compute_init_pos_vel(spectral_params)
#                         np.save(filename_output+'_init.npy',all_coeffs)
#                
                if GoOn and NeedsRefinement and CanRefine:
                    
                    print('Resizing.')

                    NBS.nint_fac = 2*NBS.nint_fac

                    if SpectralSolve:
                        best_sol = choreo.scipy_plus.nonlin.current_best(x_fine, f_fine)

                    else:
                        x = best_sol.x
                        f0 = NBS.params_to_periodicity_default(x)
                        best_sol = choreo.scipy_plus.nonlin.current_best(x, f0)

                    current_cvg_lvl += 1
                     
                elif GoOn and CanChangeOptimParams:
                    
                    print('Changing optimizer parameters.')
                    
                    i_optim_param += 1

            print('')

        print('')

    print('Done!')

def ChoreoFindFromDict(params_dict, extra_args_dict, Workspace_folder):
    
    all_kwargs = ChoreoLoadFromDict(params_dict, Workspace_folder, callback = Find_Choreo, extra_args_dict=extra_args_dict)

    Find_Choreo(**all_kwargs)

def ChoreoLoadSymList(params_dict):
    
    geodim = params_dict['Phys_Gen']['geodim']
    nbody = params_dict["Phys_Bodies"]["nbody"]
    nsyms = params_dict["Phys_Bodies"]["nsyms"]
    Sym_list = []
    
    for isym in range(nsyms):
        
        BodyPerm = np.array(params_dict["Phys_Bodies"]["AllSyms"][isym]["BodyPerm"],dtype=int)

        SpaceRot = params_dict["Phys_Bodies"]["AllSyms"][isym].get("SpaceRot")

        if SpaceRot is None:
            
            if geodim == 2:
                    
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
                        [   np.sin(rot_angle)   ,    np.cos(rot_angle)  ]   ]
                    , dtype = np.float64
                )

            else:
                
                raise ValueError(f"Provided space dimension: {geodim}. Please give a SpaceRot matrix directly.")
            
        else:
            
            SpaceRot = np.array(SpaceRot, dtype = np.float64)
            
            if SpaceRot.shape != (geodim, geodim):
                raise  ValueError(f"Invalid SpaceRot dimension. {geodim =}, {SpaceRot.shape =}")
                
        TimeRev_str = params_dict["Phys_Bodies"]["AllSyms"][isym]["TimeRev"]
        if (TimeRev_str == "True"):
            TimeRev = -1
        elif (TimeRev_str == "False"):
            TimeRev = 1
        else:
            raise ValueError("TimeRev must be True or False")

        TimeShiftNum = int(params_dict["Phys_Bodies"]["AllSyms"][isym]["TimeShiftNum"])
        TimeShiftDen = int(params_dict["Phys_Bodies"]["AllSyms"][isym]["TimeShiftDen"])

        Sym_list.append(
            ActionSym(
                BodyPerm = BodyPerm     ,
                SpaceRot = SpaceRot     ,
                TimeRev = TimeRev       ,
                TimeShiftNum = TimeShiftNum   ,
                TimeShiftDen = TimeShiftDen   ,
            )
        )
        
    return Sym_list

def ChoreoLoadFromDict(params_dict, Workspace_folder, callback=None, args_list=None, extra_args_dict={}):

    def load_target_files(filename, Workspace_folder, target_speed):

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

    np.random.seed(int(time.time()*10000) % 5000) # ???

    geodim = params_dict['Phys_Gen'] ['geodim']

    TwoDBackend = (geodim == 2)
    ParallelBackend = (params_dict['Solver_CLI']['Exec_Mul_Proc'] == "MultiThread")

    file_basename = ''
    
    CrashOnError_changevar = False

    LookForTarget = params_dict['Phys_Target'] ['LookForTarget']

#     if (LookForTarget) : # IS LIKELY BROKEN !!!!
# 
#         Rotate_fast_with_slow = params_dict['Phys_Target'] ['Rotate_fast_with_slow']
#         Optimize_Init = params_dict['Phys_Target'] ['Optimize_Init']
#         Randomize_Fast_Init =  params_dict['Phys_Target'] ['Randomize_Fast_Init']
#             
#         nT_slow = params_dict['Phys_Target'] ['nT_slow']
#         nT_fast = params_dict['Phys_Target'] ['nT_fast']
# 
#         Info_dict_slow_filename = params_dict['Phys_Target'] ["slow_filename"]
#         Info_dict_slow, all_pos_slow = load_target_files(Info_dict_slow_filename,Workspace_folder,"slow")
# 
#         ncoeff_slow = Info_dict_slow["n_int"] // 2 + 1
# 
#         all_coeffs_slow = AllPosToAllCoeffs(all_pos_slow,ncoeff_slow)
#         Center_all_coeffs(all_coeffs_slow,Info_dict_slow["nloop"],Info_dict_slow["mass"],Info_dict_slow["loopnb"],np.array(Info_dict_slow["Targets"]),np.array(Info_dict_slow["SpaceRotsUn"]))
# 
#         Info_dict_fast_list = []
#         all_coeffs_fast_list = []
# 
#         for i in range(len(nT_fast)) :
# 
#             Info_dict_fast_filename = params_dict['Phys_Target'] ["fast_filenames"] [i]
#             Info_dict_fast, all_pos_fast = load_target_files(Info_dict_fast_filename,Workspace_folder,"fast"+str(i))
#             Info_dict_fast_list.append(Info_dict_fast)
# 
#             ncoeff_fast = Info_dict_fast["n_int"] // 2 + 1
# 
#             all_coeffs_fast = AllPosToAllCoeffs(all_pos_fast,ncoeff_fast)
#             Center_all_coeffs(all_coeffs_fast,Info_dict_fast_list[i]["nloop"],Info_dict_fast_list[i]["mass"],Info_dict_fast_list[i]["loopnb"],np.array(Info_dict_fast_list[i]["Targets"]),np.array(Info_dict_fast_list[i]["SpaceRotsUn"]))
# 
#             all_coeffs_fast_list.append(all_coeffs_fast)
# 
#         Sym_list, mass,il_slow_source,ibl_slow_source,il_fast_source,ibl_fast_source = MakeTargetsSyms(Info_dict_slow,Info_dict_fast_list)
#         
#         nbody = len(mass)

    nbody = params_dict["Phys_Bodies"]["nbody"]
    nloop = params_dict["Phys_Bodies"]["nloop"]
    
    mass = np.zeros(nbody)
    for il in range(nloop):
        for ilb in range(len(params_dict["Phys_Bodies"]["Targets"][il])):
            
            ib = params_dict["Phys_Bodies"]["Targets"][il][ilb]
            mass[ib] = params_dict["Phys_Bodies"]["mass"][il]    
            
    charge = np.zeros(nbody)
    try:
        for il in range(nloop):
            for ilb in range(len(params_dict["Phys_Bodies"]["Targets"][il])):                
                ib = params_dict["Phys_Bodies"]["Targets"][il][ilb]
                charge[ib] = params_dict["Phys_Bodies"]["charge"][il]
    except KeyError:
        # TODO: Remove this
        # Replaces charge with mass for backwards compatibility
        for il in range(nloop):
            for ilb in range(len(params_dict["Phys_Bodies"]["Targets"][il])):                
                ib = params_dict["Phys_Bodies"]["Targets"][il][ilb]
                charge[ib] = params_dict["Phys_Bodies"]["mass"][il]

    # TODO: Remove this
    # Backwards compatibility again
    try:
        inter_pow = params_dict["Phys_Inter"]["inter_pow"]
        inter_pm_in = params_dict["Phys_Inter"]["inter_pm"]
        
        if inter_pm_in == "plus":
            inter_pm = 1
        elif inter_pm_in == "minus":
            inter_pm = -1
        else:
            raise ValueError(f"Invalid inter_pm {inter_pm_in}")

    except KeyError:
        inter_pow = -1.
        inter_pm = 1

    Sym_list = ChoreoLoadSymList(params_dict)

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
        
    init_pos_BB_size = params_dict["Phys_Random"].get("init_pos_BB_size", 1.)
    init_mom_BB_size = params_dict["Phys_Random"].get("init_mom_BB_size", 1.)

    SpectralSolve = params_dict["Solver_Discr"].get("SolverType") == "Spectral"
    
    rk_method =  params_dict["Solver_Discr"].get("RK_method","Gauss")
    rk_explicit = getattr(choreo.segm.precomputed_tables, rk_method, None)
    
    if rk_explicit is None:
        rk_nsteps = params_dict["Solver_Discr"].get("rk_nsteps", 10)    
        rk_implicit_x , rk_implicit_v = choreo.segm.multiprec_tables.ComputeImplicitSymplecticRKTablePair(rk_nsteps, method=rk_method)
        
    pos_mom_scaling = params_dict["Solver_Discr"].get("pos_mom_scaling", 1.)    
    
    Use_exact_Jacobian = params_dict["Solver_Discr"]["Use_exact_Jacobian"]

    Look_for_duplicates = params_dict["Solver_Checks"]["Look_for_duplicates"]
    Duplicates_Hash = params_dict["Solver_Checks"].get("Duplicates_Hash", True) # Backward compatibility

    Check_Escape = params_dict["Solver_Checks"]["Check_Escape"]

    # Penalize_Escape = True
    Penalize_Escape = False

    save_first_init = (sys.platform == 'emscripten')
    # save_first_init = False
    # save_first_init = True

    save_all_inits = False
    # save_all_inits = True

    Save_img = params_dict['Solver_CLI'] ['SaveImage']

    # Save_thumb = True
    Save_thumb = False

    # img_size = (12,12) # Image size in inches
    img_size = (8,8) # Image size in inches
    thumb_size = (2,2) # Image size in inches
    
    color = params_dict["Animation_Colors"]["color_method_input"]

    color_list = params_dict["Animation_Colors"]["colorLookup"]

    Save_anim =  params_dict['Solver_CLI'] ['SaveVideo']

    vid_size = (8,8) # Image size in inches
    
    plot_Mass_Scale = params_dict['Animation_Size'] ['checkbox_Mass_Scale']
    plot_body_size = float(params_dict['Animation_Size'] ['input_body_radius'])
    plot_trail_width = float(params_dict['Animation_Size'] ['input_trail_width'])

    n_reconverge_it_max = params_dict["Solver_Discr"] ['n_reconverge_it_max'] 
    nint_init = params_dict["Solver_Discr"]["nint_init"]   
    nint_fac_init = nint_init

    disp_scipy_opt =  (params_dict['Solver_Optim'] ['optim_verbose_lvl'] == "full")
    # disp_scipy_opt = False
    # disp_scipy_opt = True
    
    max_norm_on_entry = 1e20

    Newt_err_norm_max = params_dict["Solver_Optim"]["Newt_err_norm_max"]  
    Newt_err_norm_max_save = params_dict["Solver_Optim"]["Newt_err_norm_safe"]  

    duplicate_eps =  params_dict['Solver_Checks'] ['duplicate_eps'] 

    krylov_method = params_dict["Solver_Optim"]["krylov_method"]  

    line_search = params_dict["Solver_Optim"]["line_search"]  
    linesearch_smin = params_dict["Solver_Optim"]["line_search_smin"]  

    gradtol_list =          params_dict["Solver_Loop"]["gradtol_list"]
    inner_maxiter_list =    params_dict["Solver_Loop"]["inner_maxiter_list"]
    maxiter_list =          params_dict["Solver_Loop"]["maxiter_list"]
    outer_k_list =          params_dict["Solver_Loop"]["outer_k_list"]
    store_outer_Av_list =   params_dict["Solver_Loop"]["store_outer_Av_list"]

    n_optim_param = len(gradtol_list)
    
    escape_fac = 1e0

    escape_min_dist = 1
    escape_pow = 2.0

    n_grad_change = 1.

    freq_erase_dict = 100

    n_opt = 0
    n_opt_max = 100

    mul_coarse_to_fine = params_dict["Solver_Discr"]["mul_coarse_to_fine"]

    # Save_All_Coeffs = True
    Save_All_Coeffs = False

    # Save_Init_Pos_Vel_Sol = True
    Save_Init_Pos_Vel_Sol = False

    n_save_pos = 'auto'

    Save_SegmPos = True
    
    # Save_Params = True
    Save_Params = False
    
    plot_extend = 0.

    n_opt = 0
    # n_opt_max = 1
    n_opt_max = params_dict["Solver_Optim"]["n_opt"]
    if sys.platform == 'emscripten':
        n_find_max = 1
    else:
        n_find_max = params_dict["Solver_Optim"]["n_opt"]
    
    fftw_planner_effort = params_dict['Solver_CLI'].get('fftw_planner_effort', 'FFTW_MEASURE')
    fftw_wisdom_only = params_dict['Solver_CLI'].get('fftw_wisdom_only', False)
    fftw_nthreads = 1
    fft_backend = params_dict['Solver_CLI'].get('fft_backend', 'scipy')
    
    ReconvergeSol = False
    AddNumberToOutputName = not(sys.platform == 'emscripten')
    
    if callback is None:
        if args_list is None:
            return dict(**locals())
        else:
            the_dict = dict(**locals())
            the_dict.update(extra_args_dict)
            return {key:the_dict[key] for key in args_list}
    else:
        the_dict = dict(**locals())
        the_dict.update(extra_args_dict)
        return Pick_Named_Args_From_Dict(callback, the_dict)

def Pick_Named_Args_From_Dict(fun, the_dict, MissingArgsAreNone=True):
    
    list_of_args = inspect.getfullargspec(fun).kwonlyargs

    if MissingArgsAreNone:
        all_kwargs = {k:the_dict.get(k) for k in list_of_args}
        
    else:
        all_kwargs = {k:the_dict[k] for k in list_of_args}
    
    return all_kwargs

def ChoreoReadDictAndFind(Workspace_folder, store_folder = None, config_filename="choreo_config.json"):
    
    params_filename = os.path.join(Workspace_folder, config_filename)

    with open(params_filename) as jsonFile:
        params_dict = json.load(jsonFile)
        
    if store_folder is None:
        if sys.platform == 'emscripten':
            store_folder = os.path.join(Workspace_folder, "GUI solutions")
        else:
            store_folder = os.path.join(Workspace_folder, "CLI solutions")
        extra_args_dict = {"store_folder" : store_folder}
    else:
        extra_args_dict = {}
        
    if not os.path.isdir(store_folder):
        os.makedirs(store_folder)
        
    ChoreoChooseParallelEnvAndFind(Workspace_folder, params_dict, extra_args_dict)
    
def ChoreoChooseParallelEnvAndFind(Workspace_folder, params_dict, extra_args_dict={}):
    
    if sys.platform == 'emscripten':
        params_dict['Solver_CLI']['Exec_Mul_Proc'] = "No"
        params_dict['Solver_CLI']['SaveImage'] = False
        params_dict['Solver_CLI']['SaveVideo'] = False
        params_dict['Solver_CLI']['fft_backend'] = "scipy" 

    Exec_Mul_Proc = params_dict['Solver_CLI']['Exec_Mul_Proc']
    n_threads = params_dict['Solver_CLI']['nproc']

    if Exec_Mul_Proc == "MultiProc":

        print(f"Executing with {n_threads} workers")
        
        with threadpoolctl.threadpool_limits(limits=1):
            with concurrent.futures.ProcessPoolExecutor(max_workers=n_threads) as executor:

                res = []
                for i in range(n_threads):
                    res.append(executor.submit(ChoreoFindFromDict, params_dict, extra_args_dict, Workspace_folder))
                    time.sleep(0.01)
                
                # Useful ?    
                concurrent.futures.wait(res, return_when=concurrent.futures.FIRST_EXCEPTION)
                executor.shutdown(wait=False, cancel_futures=True)

    elif Exec_Mul_Proc == "MultiThread":
        
        with threadpoolctl.threadpool_limits(limits=n_threads):
            ChoreoFindFromDict(params_dict, extra_args_dict, Workspace_folder)

    elif Exec_Mul_Proc == "No":

        with threadpoolctl.threadpool_limits(limits=1):
            ChoreoFindFromDict(params_dict, extra_args_dict, Workspace_folder)
    else :

        raise ValueError(f'Unknown {Exec_Mul_Proc = }. Accepted values : "MultiProc", "MultiThread" or "No"')

def UpdateHashDict(store_folder, hash_dict, action_dict):
    # Creates a list of possible duplicates based on value of the action and hashes

    file_path_list = []
    for file_path in os.listdir(store_folder):

        file_path = os.path.join(store_folder, file_path)
        file_root, file_ext = os.path.splitext(os.path.basename(file_path))
        
        if (file_ext == '.json' ):
            
            This_Action_Hash = hash_dict.get(file_root)
            This_Action = action_dict.get(file_root)
            
            if (This_Action_Hash is None) :

                This_hash = ReadHashFromFile(file_path) 

                if not(This_hash is None):

                    action_dict[file_root] = This_hash[0]
                    hash_dict[file_root] = This_hash[1]

def ReadHashFromFile(filename):

    with open(filename,'r') as jsonFile:
        Info_dict = json.load(jsonFile)

    if Info_dict.get("choreo_version") == choreo.metadata.__version__:
        
        the_hash = Info_dict.get("Hash")
        the_action = Info_dict.get("Action")

        if the_hash is None:
            return None
        else:
            return the_action, np.array(the_hash)
    else:
        return None

def Check_Duplicates(NBS, segmpos, params, hash_dict, action_dict, store_folder, duplicate_eps, Action=None, Hash_Action=None, Duplicates_Hash=True):
    r"""
    Checks whether there is a duplicate of a given trajecory in the provided folder
    """
    
    UpdateHashDict(store_folder, hash_dict, action_dict)
    
    if Duplicates_Hash:

        if Hash_Action is None:
            Hash_Action = NBS.segmpos_to_hash(segmpos)
        
        for file_path, found_hash in hash_dict.items():
            
            IsCandidate = NBS.TestHashSame(Hash_Action, found_hash, duplicate_eps)

            if IsCandidate:
                return True, file_path
    else:

        if Action is None:
            Action = NBS.segmpos_params_to_action(segmpos, params)
        
        for file_path, found_action in action_dict.items():
            
            IsCandidate = NBS.TestActionSame(Action, found_action, duplicate_eps)

            if IsCandidate:
                return True, file_path

    return False, None

def Load_wisdom_file(Wisdom_file):
    
    if not(PYFFTW_AVAILABLE):
        warnings.warn("The package pyfftw could not be loaded. Please check your local install.", stacklevel=2)
    else:

        if os.path.isfile(Wisdom_file):
            with open(Wisdom_file,'r') as jsonFile:
                Wis_dict = json.load(jsonFile)
        
            wis = (
                Wis_dict["double"].encode('utf-8'),
                Wis_dict["single"].encode('utf-8'),
                Wis_dict["long"]  .encode('utf-8'),
            )

            pyfftw.import_wisdom(wis)
        
def Write_wisdom_file(Wisdom_file): 
       
    if not(PYFFTW_AVAILABLE):
        warnings.warn("The package pyfftw could not be loaded. Please check your local install.", stacklevel=2)
    else:
        wis = pyfftw.export_wisdom()
        
        Wis_dict = {
            "double": wis[0].decode('utf-8') , 
            "single": wis[1].decode('utf-8') ,
            "long":   wis[2].decode('utf-8') ,
        }
            
        with open(Wisdom_file, "w") as jsonFile:
            jsonString = json.dumps(Wis_dict, indent=4, sort_keys=False)
            jsonFile.write(jsonString)

def FindTimeRevSymmetry(NBS, semgpos, ntries = 1, hit_tol = 1e-7, refl_dim = [0], return_best = False):
    
    if isinstance(refl_dim, int):
        refl_dim = [refl_dim]
    
    IsReflexionInvariant = False
    for Sym in NBS.Sym_list:
        IsReflexionInvariant = IsReflexionInvariant or (Sym.TimeRev == -1)
    
    if IsReflexionInvariant:
        # I want at most one TimeRev == -1 symmetry
        return 
    
    params_ini = NBS.segmpos_to_params(semgpos)
    
    def Compute_Sym(SymParams, *args):
        
        dt = SymParams[0]
        rot = ActionSym.SurjectiveDirectSpaceRot(SymParams[1:])
        refl = np.identity(NBS.geodim)
        
        for idim in refl_dim:
            if idim >= 0:
                refl[idim,idim] = -1

        Sym = ActionSym(
            args[0] ,
            refl    ,
            -1      ,
            0       ,
            1       ,
        )

        all_coeffs_dense = NBS.params_to_all_coeffs_dense_noopt(params_ini, dt=dt) 
        
        for i in range(NBS.nloop):
            all_coeffs_dense[i] = np.matmul(all_coeffs_dense[i] , rot)

        params_dt = NBS.all_coeffs_dense_to_params_noopt(all_coeffs_dense)
        segmpos_dt = NBS.params_to_segmpos(params_dt)
        
        return Sym, segmpos_dt
    
    def EvalSym(SymParams, *args):
        
        Sym, segmpos_dt = Compute_Sym(SymParams, *args)

        return NBS.ComputeSymDefault(segmpos_dt, Sym, lnorm = 22, full=False)
    
    n_SymParams = 1 + (NBS.geodim * (NBS.geodim-1) // 2)
    
    best_sol = choreo.scipy_plus.nonlin.current_best((np.zeros(n_SymParams,dtype=np.float64),np.array(range(NBS.nbody))), np.inf)
    
    for itry in range(ntries):

        for BodyPerm in ActionSym.InvolutivePermutations(NBS.nbody):
            
            x0 = np.random.random(n_SymParams)
    
            # method = "BFGS"
            method = "L-BFGS-B"
            # method = "SLSQP"
            opt_res = scipy.optimize.minimize(EvalSym, x0, args=(BodyPerm,), method=method, tol=1e-8, callback=None, options={"maxiter":100})

            best_sol.update((opt_res.x, BodyPerm), opt_res.fun)

            if opt_res.fun < hit_tol:
                
                return Compute_Sym(opt_res.x, BodyPerm)

    if return_best:            
        x, f, f_norm = best_sol.get_best()    
        return Compute_Sym(x[0], x[1])
 
def FindTimeDirectSymmetry(NBS, semgpos, ntries = 1, refl_dim = [0], hit_tol = 1e-7, return_best = False):

    def Compute_Sym(SymParams, *args):
        
        rot = ActionSym.SurjectiveDirectSpaceRot(SymParams)
        
        for idim in refl_dim:
            if idim >= 0:
                rot[idim,idim] = -1

        Sym = ActionSym(
            args[0] ,
            rot     ,
            -1      ,
            0       ,
            1       ,
        )

        return Sym
    
    def EvalSym(SymParams, *args):
        
        Sym, segmpos_dt = Compute_Sym(SymParams, *args)

        return NBS.ComputeSymDefault(segmpos_dt, Sym, lnorm = 22, full=False)
    
    n_SymParams = 1 + (NBS.geodim * (NBS.geodim-1) // 2)
    
    best_sol = choreo.scipy_plus.nonlin.current_best((np.zeros(n_SymParams,dtype=np.float64),np.array(range(NBS.nbody))), np.inf)
    
    for itry in range(ntries):

        for BodyPerm in ActionSym.InvolutivePermutations(NBS.nbody):
            
            if random_init:
                x0 = np.random.random(n_SymParams)
            else:
                x0 = np.zeros(n_SymParams,dtype=np.float64)
                
            # method = "BFGS"
            method = "L-BFGS-B"
            # method = "SLSQP"
            opt_res = scipy.optimize.minimize(EvalSym, x0, args=(BodyPerm,), method=method, tol=1e-8, callback=None, options={"maxiter":100})

            best_sol.update((opt_res.x, BodyPerm), opt_res.fun)

            if opt_res.fun < hit_tol:
                
                return Compute_Sym(opt_res.x, BodyPerm)

    if return_best:            
        x, f, f_norm = best_sol.get_best()    
        return Compute_Sym(x[0], x[1])
 
