import os
try:
    import multiprocessing
    import concurrent.futures
except:
    pass


import sys,argparse
import random
import numpy as np
import math as m
import scipy.optimize as opt

import copy
import shutil
import time
import builtins

try:
    import pyamg
except:
    pass

import choreo.scipy_plus
from choreo.funs import *
from choreo.funs_new import *
from choreo.helper import *

from choreo.cython.funs_new import ActionSym

def Find_Choreo(
    geodim,
    TwoDBackend,
    ParallelBackend,
    GradHessBackend,
    nbody,
    n_reconverge_it_max,
    nint_init,
    mass,
    Sym_list,
    MomConsImposed,
    n_grad_change,
    store_folder,
    nT_slow,
    nT_fast,
    Info_dict_slow,
    all_coeffs_slow,
    Info_dict_fast_list,
    all_coeffs_fast_list,
    il_slow_source,
    ibl_slow_source,
    il_fast_source,
    ibl_fast_source,
    Rotate_fast_with_slow,
    Optimize_Init,
    Randomize_Fast_Init,
    save_all_inits,
    save_first_init,
    Save_img,
    Save_thumb,
    nint_plot_img,
    img_size,
    thumb_size,
    color,
    Save_Newton_Error,
    Save_GradientAction_Error,
    Save_coeff_profile,
    gradtol_list,
    inner_maxiter_list,
    maxiter_list,
    outer_k_list,
    store_outer_Av_list,
    n_optim_param,
    krylov_method,
    Use_exact_Jacobian,
    disp_scipy_opt,
    line_search,
    Check_Escape,
    Look_for_duplicates,
    duplicate_eps,
    foundsol_tol,
    gradtol_max,
    Newt_err_norm_max,
    Newt_err_norm_max_save,
    Save_anim,
    nint_plot_anim,
    nperiod_anim,
    Plot_trace_anim,
    vid_size,
    n_opt_max,
    freq_erase_dict,
    coeff_ampl_o,
    k_infl,
    k_max,
    coeff_ampl_min,
    LookForTarget,
    dnint,
    file_basename,
    max_norm_on_entry,
    n_save_pos,
    Save_All_Coeffs,
    Save_All_Pos,
    Save_Init_Pos_Vel_Sol,
    mul_coarse_to_fine,
    n_find_max,
    plot_extend,
    CrashOnError_changevar,
    color_list,
    optim_callback_list,
    callback_after_init_list,
    linesearch_smin,
    ReconvergeSol,
    all_coeffs_init,
    AddNumberToOutputName,
    SkipCheckRandomMinDist,
    CurrentlyDeveloppingNewStuff,
):
    """

    Finds periodic solutions

    """

    if CurrentlyDeveloppingNewStuff:

        print("WARNING : ENTERING DEV GROUNDS. Proceed with caution")

        ActionSyst = setup_changevar_new(
            geodim                                      ,
            nbody                                       ,
            nint_init                                   ,
            mass                                        ,
            n_reconverge_it_max                         ,
            Sym_list = Sym_list                         ,
            MomCons = MomConsImposed                    ,
            n_grad_change = n_grad_change               ,
            CrashOnIdentity = CrashOnError_changevar    ,
        )

        return

    
    print(f'Searching periodic solutions of {nbody:d} bodies.')

    print(f'Processing symmetries for {(n_reconverge_it_max+1):d} convergence levels.')

    ActionSyst = setup_changevar(
        geodim                                      ,
        nbody                                       ,
        nint_init                                   ,
        mass                                        ,
        n_reconverge_it_max                         ,
        Sym_list = Sym_list                         ,
        MomCons = MomConsImposed                    ,
        n_grad_change = n_grad_change               ,
        CrashOnIdentity = CrashOnError_changevar    ,
    )

    ActionSyst.SetBackend(parallel=ParallelBackend,TwoD=TwoDBackend,GradHessBackend=GradHessBackend)

    start_cvg_lvl = 0
    ActionSyst.current_cvg_lvl = start_cvg_lvl

    print('')

    nbi_tot = 0
    for il in range(ActionSyst.nloop):
        for ilp in range(il+1,ActionSyst.nloop):
            nbi_tot += ActionSyst.loopnb[il]*ActionSyst.loopnb[ilp]
        nbi_tot += ActionSyst.loopnbi[il]
    nbi_naive = (nbody*(nbody-1))//2

    print('Imposed constraints lead to the detection of:')
    print(f'    {ActionSyst.nloop:d} independant loops')
    print(f'    {nbi_tot:d} binary interactions')
    print(f'    ==> Reduction of {100*(1-nbi_tot/nbi_naive):.2f} % wrt the {nbi_naive:d} naive binary iteractions')
    print('')


    if ActionSyst.MatrixFreeChangevar:
        print('Matrix-free change of variables')
    else:
        print('Sparse change of variables')
    print('')

    ncoeff = ActionSyst.ncoeff
    nint = ActionSyst.nint
    nparams_before = 2 * ncoeff * ActionSyst.nloop * geodim
    nparams_after = ActionSyst.nparams

    print(f'Convergence attempt number: 1')
    print(f"    Number of Fourier coeffs: {ncoeff}")
    print(f"    Number of scalar parameters before constraints: {nparams_before}")
    print(f"    Number of scalar parameters after  constraints: {nparams_after}")
    print(f"    ==> Reduction of {100*(1-nparams_after/nparams_before):.2f} %")
    print('')

    all_coeffs_min,all_coeffs_max = Make_Init_bounds_coeffs(ActionSyst.nloop,ActionSyst.geodim,ncoeff,coeff_ampl_o,k_infl,k_max,coeff_ampl_min)
    
    x_min = ActionSyst.Package_all_coeffs(all_coeffs_min)
    x_max = ActionSyst.Package_all_coeffs(all_coeffs_max)

    rand_eps = coeff_ampl_min
    rand_dim = 0
    for i in range(ActionSyst.nparams):

        if (abs(x_max[i] - x_min[i]) > rand_eps):
            rand_dim +=1

    print(f'Number of initialization dimensions: {rand_dim}')

    sampler = UniformRandom(d=rand_dim)

    if not(SkipCheckRandomMinDist):

        x0 = np.random.random(ActionSyst.nparams)
        xmin = ActionSyst.Compute_MinDist(x0)
        if (xmin < 1e-5):
            print("")
            print(f"Init minimum inter body distance too low: {xmin}")
            print("There is likely something wrong with constraints.")
            print("")

            n_opt_max = -1

    n_opt = 0
    n_find = 0
    
    if optim_callback_list is None:
        optim_callback_list = []
    
    n_optim_callback_list = len(optim_callback_list)    

    if callback_after_init_list is None:
        callback_after_init_list = []
    
    n_callback_after_init_list = len(callback_after_init_list)

    ForceFirstEntry = save_first_init

    while (((n_opt < n_opt_max) and (n_find < n_find_max)) or ForceFirstEntry):

        ForceFirstEntry = False
        AskedForNext = False

        if (Look_for_duplicates and ((n_opt % freq_erase_dict) == 0)):
            
            hash_dict = {}
            _ = SelectFiles_Action(store_folder,hash_dict)

        ActionSyst.current_cvg_lvl = start_cvg_lvl

        ncoeff = ActionSyst.ncoeff
        nint = ActionSyst.nint

        if (ReconvergeSol):

            x_avg = ActionSyst.Package_all_coeffs(all_coeffs_init)
        
        elif (LookForTarget):

            all_coeffs_avg = ActionSyst.Gen_init_avg_2D(nT_slow,nT_fast,Info_dict_slow,all_coeffs_slow,Info_dict_fast_list,all_coeffs_fast_list,il_slow_source,ibl_slow_source,il_fast_source,ibl_fast_source,Rotate_fast_with_slow,Optimize_Init,Randomize_Fast_Init)

            x_avg = ActionSyst.Package_all_coeffs(all_coeffs_avg)

        else:
            
            x_avg = np.zeros((ActionSyst.nparams),dtype=np.float64)

        x0 = np.zeros((ActionSyst.nparams),dtype=np.float64)
        
        xrand = sampler.random()
        
        x0 = PopulateRandomInit(
            ActionSyst.nparams,
            x_avg   ,  
            x_min   ,  
            x_max   ,
            xrand   ,
            rand_eps
        )

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

                filename_output = os.path.join(store_folder,file_basename+'_init_'+str(max_num_file).zfill(5))

            else:

                filename_output = os.path.join(store_folder,file_basename+'_init')

            print(f'Saving initial state as {filename_output}.*.')

            ActionSyst.Write_Descriptor(x0,filename_output+'.json')

            if Save_img :
                ActionSyst.plot_all_2D(x0,nint_plot_img,filename_output+'.png',fig_size=img_size,color=color,color_list=color_list)        

            if Save_thumb :
                ActionSyst.plot_all_2D(x0,nint_plot_img,filename_output+'_thumb.png',fig_size=thumb_size,color=color,color_list=color_list)        
                
            if Save_anim :
                ActionSyst.plot_all_2D_anim(x0,nint_plot_anim,filename_output+'.mp4',nperiod_anim,Plot_trace=Plot_trace_anim,fig_size=vid_size,dnint=dnint,color_list=color_list,color=color)
            
            if Save_Newton_Error :
                ActionSyst.plot_Newton_Error(x0,filename_output+'_newton.png')

            if Save_GradientAction_Error :
                ActionSyst.plot_GradientAction_Error(x0,filename_output+'_gradaction.png')

            if Save_coeff_profile:
                ActionSyst.plot_coeff_profile(x0,filename_output+'_coeff_profile.png')

            if Save_All_Coeffs:

                all_coeffs = ActionSyst.Unpackage_all_coeffs(x0)
                np.save(filename_output+'_coeffs.npy',all_coeffs)

            if Save_All_Pos:

                if n_save_pos is None:
                    all_pos = ActionSyst.ComputeAllLoopPos(x0)
                elif n_save_pos == 'auto':
                    # TODO : implement auto
                    all_pos = ActionSyst.ComputeAllLoopPos(x0)
                else:
                    all_pos = ActionSyst.ComputeAllLoopPos(x0,n_save_pos)

                np.save(filename_output+'.npy',all_pos)

        for i in range(n_callback_after_init_list):
            callback_after_init_list[i]()

        f0 = ActionSyst.Compute_action_onlygrad(x0)
        best_sol = current_best(x0,f0)

        GoOn = (best_sol.f_norm < max_norm_on_entry)
        
        if not(GoOn):
            print(f"Norm on entry is {best_sol.f_norm:.2e} which is too big.")
        
        i_optim_param = 0

        n_opt += 1

        GoOn = (n_opt <= n_opt_max)
        
        if GoOn:
            print(f'Optimization attempt number: {n_opt}')
        else:
            print('Reached max number of optimization attempts')

        while GoOn:
            # Set correct optim params

            x0 = np.copy(best_sol.x)
            
            inner_tol = 0.
            
            rdiff = None
            gradtol = gradtol_list[i_optim_param]
            inner_maxiter = inner_maxiter_list[i_optim_param]
            maxiter = maxiter_list[i_optim_param]
            outer_k = outer_k_list[i_optim_param]
            store_outer_Av = store_outer_Av_list[i_optim_param]

            ActionGradNormEnterLoop = best_sol.f_norm
            
            print(f'Action Grad Norm on entry: {ActionGradNormEnterLoop:.2e}')
            print(f'Optim level: {i_optim_param+1} / {n_optim_param}    Resize level: {ActionSyst.current_cvg_lvl+1} / {n_reconverge_it_max+1}')
            
            F = lambda x : ActionSyst.Compute_action_onlygrad(x)
            
            inner_M = None

            # inner_M = ActionSyst.GetAMGPreco(x0,krylov_method=krylov_method,cycle='V')
            # inner_M = ActionSyst.GetAMGPreco(x0,krylov_method=krylov_method,cycle='W')
            # inner_M = ActionSyst.GetAMGPreco(x0,krylov_method=krylov_method,cycle='F')
            # inner_M = ActionSyst.GetAMGPreco(x0,krylov_method=krylov_method,cycle='AMLI')

            if (krylov_method == 'lgmres'):
                jac_options = {'method':krylov_method,'rdiff':rdiff,'outer_k':outer_k,'inner_inner_m':inner_maxiter,'inner_store_outer_Av':store_outer_Av,'inner_tol':inner_tol,'inner_M':inner_M }
            elif (krylov_method == 'gmres'):
                jac_options = {'method':krylov_method,'rdiff':rdiff,'outer_k':outer_k,'inner_tol':inner_tol,'inner_M':inner_M }
            else:
                jac_options = {'method':krylov_method,'rdiff':rdiff,'outer_k':outer_k,'inner_tol':inner_tol,'inner_M':inner_M }

            jacobian = ActionSyst.GetKrylovJacobian(Use_exact_Jacobian, jac_options)

            def optim_callback(x,f,f_norm):

                AskedForNext = False
                
                best_sol.update(x,f,f_norm)

                for i in range(n_optim_callback_list):

                    AskedForNext = (AskedForNext or optim_callback_list[i](x,f,f_norm,ActionSyst))

                return AskedForNext

            try : 
                
                opt_result , info = nonlin_solve_pp(F=F,x0=x0,jacobian=jacobian,verbose=disp_scipy_opt,maxiter=maxiter,f_tol=gradtol,line_search=line_search,callback=optim_callback,raise_exception=False,smin=linesearch_smin,full_output=True,tol_norm=np.linalg.norm)

                AskedForNext = (info['status'] == 0)

            except Exception as exc:
                
                print(exc)
                print("Value Error occured, skipping.")
                GoOn = False
                raise(exc)
                
            if (AskedForNext):
                print("Skipping at user's request")
                GoOn = False

            SaveSol = False

            Gradaction = best_sol.f_norm

            Hash_Action = None
            Action = None
            
            if (GoOn and Check_Escape):
                
                Escaped,_ = ActionSyst.Detect_Escape(best_sol.x)

                if Escaped:
                    print('One loop escaped. Starting over.')    
                    
                GoOn = GoOn and not(Escaped)
                
            if (GoOn and Look_for_duplicates):

                Found_duplicate,file_path = ActionSyst.Check_Duplicates(best_sol.x,hash_dict,store_folder,duplicate_eps)
                
                if (Found_duplicate):
                
                    print('Found Duplicate!')   
                    print('Path: ',file_path)
                    
                GoOn = GoOn and not(Found_duplicate)
                
            if (GoOn):
                
                ParamFoundSol = (best_sol.f_norm < foundsol_tol)
                ParamPreciseEnough = (best_sol.f_norm < gradtol_max)
                # print(f'Opt Action Grad Norm : {best_sol.f_norm} from {ActionGradNormEnterLoop}')
                print(f'Opt Action Grad Norm: {best_sol.f_norm:.2e}')
            
                Newt_err = ActionSyst.Compute_Newton_err(best_sol.x)
                Newt_err_norm = np.linalg.norm(Newt_err)/(nint*nbody)
                NewtonPreciseGood = (Newt_err_norm < Newt_err_norm_max)
                NewtonPreciseEnough = (Newt_err_norm < Newt_err_norm_max_save)
                print(f'Newton Error: {Newt_err_norm:.2e}')
                
                CanChangeOptimParams = i_optim_param < (n_optim_param-1)
                
                CanRefine = (ActionSyst.current_cvg_lvl < n_reconverge_it_max)
                
                if CanRefine :

                    x_fine = ActionSyst.TransferParamBtwRefinementLevels(best_sol.x)

                    ActionSyst.current_cvg_lvl += 1
                        

                    f_fine = ActionSyst.Compute_action_onlygrad(x_fine)
                    f_fine_norm = np.linalg.norm(f_fine)
                    
                    NeedsRefinement = (f_fine_norm > mul_coarse_to_fine*best_sol.f_norm)
                    
                    ActionSyst.current_cvg_lvl += -1
                
                else:
                    
                    NeedsRefinement = False

                # NeedsChangeOptimParams = GoOn and CanChangeOptimParams and not(ParamPreciseEnough) and not(NewtonPreciseGood) and not(NeedsRefinement)
                NeedsChangeOptimParams = GoOn and CanChangeOptimParams and not(NewtonPreciseGood) and not(NeedsRefinement)
                
                # print("ParamFoundSol ",ParamFoundSol)
                # print("ParamPreciseEnough ",ParamPreciseEnough)
                # print("NewtonPreciseEnough ",NewtonPreciseEnough)
                # print("NewtonPreciseGood ",NewtonPreciseGood)
                # print("NeedsChangeOptimParams ",NeedsChangeOptimParams)
                # print("CanChangeOptimParams ",CanChangeOptimParams)
                # print("NeedsRefinement ",NeedsRefinement)
                # print("CanRefine ",CanRefine)
                # 
                if GoOn and not(ParamFoundSol):
                
                    GoOn = False
                    print('Optimizer could not zero in on a solution.')

                if GoOn and not(ParamPreciseEnough) and not(NewtonPreciseEnough) and not(CanChangeOptimParams):
                
                    GoOn = False

                    print('Newton Error too high, discarding solution.')
                
                if GoOn and ParamPreciseEnough and not(NewtonPreciseEnough) and not(NeedsRefinement):

                    GoOn=False
                    print("Stopping search: there might be something wrong with the constraints.")
                    # SaveSol = True
                
                if GoOn and NewtonPreciseGood :

                    GoOn = False
                    print("Stopping search: found solution.")
                    SaveSol = True
                    
                if GoOn and NewtonPreciseEnough and not(CanChangeOptimParams) :

                    GoOn = False
                    print("Stopping search: found approximate solution.")
                    SaveSol = True

                if GoOn and not(NeedsRefinement) and not(NeedsChangeOptimParams):
                
                    GoOn = False
                    print('Could not converge within prescibed optimizer and refinement parameters.')

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

                        filename_output = os.path.join(store_folder,file_basename+str(max_num_file).zfill(5))

                    else:

                        filename_output = os.path.join(store_folder,file_basename)

                    print(f'Saving solution as {filename_output}.*.')
             
                    ActionSyst.Write_Descriptor(best_sol.x,filename_output+'.json',Action=Action,Gradaction=Gradaction,Newt_err_norm=Newt_err_norm,Hash_Action=Hash_Action,extend=plot_extend)

                    if Save_img :
                        ActionSyst.plot_all_2D(best_sol.x,nint_plot_img,filename_output+'.png',fig_size=img_size,color=color,color_list=color_list)
                    
                    if Save_thumb :
                        ActionSyst.plot_all_2D(best_sol.x,nint_plot_img,filename_output+'_thumb.png',fig_size=thumb_size,color=color,color_list=color_list)
                        
                    if Save_anim :
                        ActionSyst.plot_all_2D_anim(best_sol.x,nint_plot_anim,filename_output+'.mp4',nperiod_anim,Plot_trace=Plot_trace_anim,fig_size=vid_size,dnint=dnint,color_list=color_list,color=color)

                    if Save_Newton_Error :
                        ActionSyst.plot_Newton_Error(best_sol.x,filename_output+'_newton.png')

                    if Save_GradientAction_Error :
                        ActionSyst.plot_GradientAction_Error(best_sol.x,filename_output+'_gradaction.png')

                    if Save_coeff_profile:
                        ActionSyst.plot_coeff_profile(best_sol.x,filename_output+'_coeff_profile.png')

                    if Save_All_Coeffs:
                        all_coeffs = ActionSyst.Unpackage_all_coeffs(best_sol.x)
                        np.save(filename_output+'_coeffs.npy',all_coeffs)

                    if Save_All_Pos:
                        if n_save_pos is None:
                            all_pos = ActionSyst.ComputeAllLoopPos(best_sol.x)
                        elif n_save_pos == 'auto':
                            # TODO : implement auto
                            all_pos = ActionSyst.ComputeAllLoopPos(best_sol.x)
                        else:
                            all_pos = ActionSyst.ComputeAllLoopPos(best_sol.x,n_save_pos)

                        np.save(filename_output+'.npy',all_pos)

                    if Save_Init_Pos_Vel_Sol:
                        all_pos_b = ActionSyst.Compute_init_pos_vel(best_sol.x)
                        np.save(filename_output+'_init.npy',all_coeffs)
               
                if GoOn and NeedsRefinement:
                    
                    print('Resizing.')
                    
                    best_sol = current_best(x_fine,f_fine)
                    ActionSyst.current_cvg_lvl += 1
                    
                    ncoeff = ActionSyst.ncoeff
                    nint = ActionSyst.nint    
                    
                if GoOn and NeedsChangeOptimParams:
                    
                    print('Changing optimizer parameters.')
                    
                    i_optim_param += 1
                
            print('')

        print('')

    print('Done!')

def GenSymExample(
    geodim,
    ParallelBackend,
    TwoDBackend,
    GradHessBackend,
    nbody,
    nint_init,
    mass,
    Sym_list,
    MomConsImposed,
    n_grad_change,
    coeff_ampl_o,
    k_infl,
    k_max,
    coeff_ampl_min,
    LookForTarget,
    nT_slow,
    nT_fast,
    Info_dict_slow,
    all_coeffs_slow,
    Info_dict_fast_list,
    all_coeffs_fast_list,
    il_slow_source,
    ibl_slow_source,
    il_fast_source,
    ibl_fast_source,
    Rotate_fast_with_slow,
    Optimize_Init,
    Randomize_Fast_Init,
    Save_img,
    nint_plot_img,
    img_size,
    color,
    Save_anim,
    nint_plot_anim,
    nperiod_anim,
    Plot_trace_anim,
    vid_size,
    dnint,
    Save_All_Coeffs,
    Save_All_Pos,
    n_save_pos,
    plot_extend,
    CrashOnError_changevar,
    color_list,
):

    print(f'Building an initial state with {nbody:d} bodies.')
    print('')

    n_reconverge_it_max = 0
    n_grad_change = 1

    ActionSyst = setup_changevar(geodim,nbody,nint_init,mass,n_reconverge_it_max,Sym_list=Sym_list,MomCons=MomConsImposed,n_grad_change=n_grad_change,CrashOnIdentity=CrashOnError_changevar)

    ActionSyst.SetBackend(parallel=ParallelBackend,TwoD=TwoDBackend,GradHessBackend=GradHessBackend)
    
    nbi_tot = 0
    for il in range(ActionSyst.nloop):
        for ilp in range(il+1,ActionSyst.nloop):
            nbi_tot += ActionSyst.loopnb[il]*ActionSyst.loopnb[ilp]
        nbi_tot += ActionSyst.loopnbi[il]
    nbi_naive = (nbody*(nbody-1))//2

    print('Imposed constraints lead to the detection of:')
    print(f'    {ActionSyst.nloop:d} independant loops')
    print(f'    {nbi_tot:d} binary interactions')
    print(f'    ==> Reduction of {100*(1-nbi_tot/nbi_naive):.2f} % wrt the {nbi_naive:d} naive binary iteractions')
    print('')

    ncoeff = ActionSyst.ncoeff
    nint = ActionSyst.nint
    nparams_before = 2 * ncoeff * ActionSyst.nloop * geodim
    nparams_after = ActionSyst.nparams

    print(f'Convergence attempt number: 1')
    print(f"    Number of Fourier coeffs: {ncoeff}")
    print(f"    Number of scalar parameters before constraints: {nparams_before}")
    print(f"    Number of scalar parameters after  constraints: {nparams_after}")
    print(f"    ==> Reduction of {100*(1-nparams_after/nparams_before):.2f} %")
    print('')

    x0 = np.random.random(ActionSyst.nparams)
    xmin = ActionSyst.Compute_MinDist(x0)
    if (xmin < 1e-5):

        print("")
        print(f"Init minimum inter body distance too low : {xmin:.2e}.")
        print("There is likely something wrong with constraints.")
        print("")

        # return False

    all_coeffs_min,all_coeffs_max = Make_Init_bounds_coeffs(ActionSyst.nloop,ActionSyst.geodim,ncoeff,coeff_ampl_o,k_infl,k_max,coeff_ampl_min)

    x_min = ActionSyst.Package_all_coeffs(all_coeffs_min)
    x_max = ActionSyst.Package_all_coeffs(all_coeffs_max)

    rand_eps = coeff_ampl_min
    rand_dim = 0
    for i in range(ActionSyst.nparams):
        if (abs(x_max[i] - x_min[i]) > rand_eps):
            rand_dim +=1

    sampler = UniformRandom(d=rand_dim)

    if (LookForTarget):
        
        all_coeffs_avg = ActionSyst.Gen_init_avg_2D(nT_slow,nT_fast,Info_dict_slow,all_coeffs_slow,Info_dict_fast_list,all_coeffs_fast_list,il_slow_source,ibl_slow_source,il_fast_source,ibl_fast_source,Rotate_fast_with_slow,Optimize_Init,Randomize_Fast_Init)    

        x_avg = ActionSyst.Package_all_coeffs(all_coeffs_avg)
    
    else:
        
        x_avg = np.zeros((ActionSyst.nparams),dtype=np.float64)


    x0 = np.zeros((ActionSyst.nparams),dtype=np.float64)
    
    xrand = sampler.random()

    x0 = PopulateRandomInit(
        ActionSyst.nparams,
        x_avg   ,  
        x_min   ,  
        x_max   ,
        xrand   ,
        rand_eps
    )

    ActionSyst.Write_Descriptor(x0,'init.json',extend=plot_extend)

    if Save_img :
        ActionSyst.plot_all_2D(x0,nint_plot_img,'init.png',fig_size=img_size,color=color,color_list=color_list)        

    if Save_anim :
        ActionSyst.plot_all_2D_anim(x0,nint_plot_anim,'init.mp4',nperiod_anim,Plot_trace=Plot_trace_anim,fig_size=vid_size,dnint=dnint,color_list=color_list,color=color)

    if Save_All_Coeffs:
        all_coeffs = ActionSyst.Unpackage_all_coeffs(x0)
        np.save('init_coeffs.npy',all_coeffs)

    if Save_All_Pos:
        if n_save_pos is None:
            all_pos_b = ActionSyst.ComputeAllLoopPos(x0)
        elif n_save_pos == 'auto':
            # TODO : implement auto
            all_pos_b = ActionSyst.ComputeAllLoopPos(x0)
        else:
            all_pos_b = ActionSyst.ComputeAllLoopPos(x0,n_save_pos)

        np.save('init.npy',all_pos_b)

    return True

def Speed_test(
    geodim,
    ParallelBackend,
    TwoDBackend,
    GradHessBackend,
    grad_backend,
    hess_backend,
    nbody,
    n_reconverge_it_max,
    nint_init,
    mass,
    Sym_list,
    MomConsImposed,
    n_grad_change,
    coeff_ampl_o,
    k_infl,
    k_max,
    coeff_ampl_min,
    CrashOnError_changevar,
    n_test,
    LookForTarget,
    nT_slow,
    nT_fast,
    Info_dict_slow,
    all_coeffs_slow,
    Info_dict_fast_list,
    all_coeffs_fast_list,
    il_slow_source,
    ibl_slow_source,
    il_fast_source,
    ibl_fast_source,
    Rotate_fast_with_slow,
    Optimize_Init,
    Randomize_Fast_Init
):

    """

    Helper function to profile code

    """

    ActionSyst = setup_changevar(geodim,nbody,nint_init,mass,n_reconverge_it_max,Sym_list=Sym_list,MomCons=MomConsImposed,n_grad_change=n_grad_change,CrashOnIdentity=CrashOnError_changevar)

    # ActionSyst.SetBackend(parallel=ParallelBackend,TwoD=TwoDBackend,GradHessBackend=GradHessBackend)

    if not(grad_backend is None):
        ActionSyst.ComputeGradBackend = grad_backend
    if not(hess_backend is None):
        ActionSyst.ComputeHessBackend = hess_backend

    # start_cvg_lvl = 0
    start_cvg_lvl = n_reconverge_it_max
    ActionSyst.current_cvg_lvl = start_cvg_lvl


    ncoeff = ActionSyst.ncoeff
    nint = ActionSyst.nint

    all_coeffs_min,all_coeffs_max = Make_Init_bounds_coeffs(ActionSyst.nloop,ActionSyst.geodim,ncoeff,coeff_ampl_o,k_infl,k_max,coeff_ampl_min)
    
    x_min = ActionSyst.Package_all_coeffs(all_coeffs_min)
    x_max = ActionSyst.Package_all_coeffs(all_coeffs_max)

    rand_eps = coeff_ampl_min
    rand_dim = 0
    for i in range(ActionSyst.nparams):

        if (abs(x_max[i] - x_min[i]) > rand_eps):
            rand_dim +=1

    sampler = UniformRandom(d=rand_dim)

    x0 = np.random.random(ActionSyst.nparams)
    xmin = ActionSyst.Compute_MinDist(x0)

    if (xmin < 1e-5):
        # print(xmin)
        # raise ValueError("Init inter body distance too low. There is something wrong with constraints")
        print("")
        print(f"Init minimum inter body distance too low: {xmin}")
        print("There is likely something wrong with constraints.")
        print("")
    
    tot_time = 0.

    if (LookForTarget):
        
        all_coeffs_avg = ActionSyst.Gen_init_avg_2D(nT_slow,nT_fast,Info_dict_slow,all_coeffs_slow,Info_dict_fast_list,all_coeffs_fast_list,il_slow_source,ibl_slow_source,il_fast_source,ibl_fast_source,Rotate_fast_with_slow,Optimize_Init,Randomize_Fast_Init)    

        x_avg = ActionSyst.Package_all_coeffs(all_coeffs_avg)
    
    else:
        
        x_avg = np.zeros((ActionSyst.nparams),dtype=np.float64)

        
    x0 = np.zeros((ActionSyst.nparams),dtype=np.float64)
    
    xrand = sampler.random()
    
    x0 = PopulateRandomInit(
        ActionSyst.nparams,
        x_avg   ,  
        x_min   ,  
        x_max   ,
        xrand   ,
        rand_eps
    )

    dx = np.random.random(x0.shape)

    beg = time.perf_counter()
    for itest in range(n_test):

        xrand = sampler.random()

        x0 = PopulateRandomInit(
            ActionSyst.nparams,
            x_avg   ,  
            x_min   ,  
            x_max   ,
            xrand   ,
            rand_eps
        )

        f0 = ActionSyst.Compute_action_onlygrad(x0)
        hess = ActionSyst.Compute_action_hess_mul(x0,dx)

    end = time.perf_counter()
    tot_time += (end-beg)

    print(f'Total time s : {tot_time}')
        
def ChoreoFindFromDict(params_dict,Workspace_folder):

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

    np.random.seed(int(time.time()*10000) % 5000)

    geodim = params_dict['Geom_Gen'] ['geodim']

    TwoDBackend = (geodim == 2)
    ParallelBackend = (params_dict['Solver_CLI']['Exec_Mul_Proc'] == "MultiThread")
    GradHessBackend = params_dict['Solver_CLI']['GradHess_backend']

    file_basename = ''
    
    CrashOnError_changevar = False

    LookForTarget = params_dict['Geom_Target'] ['LookForTarget']

    if (LookForTarget) : # IS LIKELY BROKEN !!!!

        Rotate_fast_with_slow = params_dict['Geom_Target'] ['Rotate_fast_with_slow']
        Optimize_Init = params_dict['Geom_Target'] ['Optimize_Init']
        Randomize_Fast_Init =  params_dict['Geom_Target'] ['Randomize_Fast_Init']
            
        nT_slow = params_dict['Geom_Target'] ['nT_slow']
        nT_fast = params_dict['Geom_Target'] ['nT_fast']

        Info_dict_slow_filename = params_dict['Geom_Target'] ["slow_filename"]
        Info_dict_slow, all_pos_slow = load_target_files(Info_dict_slow_filename,Workspace_folder,"slow")

        ncoeff_slow = Info_dict_slow["n_int"] // 2 + 1

        all_coeffs_slow = AllPosToAllCoeffs(all_pos_slow,ncoeff_slow)
        Center_all_coeffs(all_coeffs_slow,Info_dict_slow["nloop"],Info_dict_slow["mass"],Info_dict_slow["loopnb"],np.array(Info_dict_slow["Targets"]),np.array(Info_dict_slow["SpaceRotsUn"]))

        Info_dict_fast_list = []
        all_coeffs_fast_list = []

        for i in range(len(nT_fast)) :

            Info_dict_fast_filename = params_dict['Geom_Target'] ["fast_filenames"] [i]
            Info_dict_fast, all_pos_fast = load_target_files(Info_dict_fast_filename,Workspace_folder,"fast"+str(i))
            Info_dict_fast_list.append(Info_dict_fast)

            ncoeff_fast = Info_dict_fast["n_int"] // 2 + 1

            all_coeffs_fast = AllPosToAllCoeffs(all_pos_fast,ncoeff_fast)
            Center_all_coeffs(all_coeffs_fast,Info_dict_fast_list[i]["nloop"],Info_dict_fast_list[i]["mass"],Info_dict_fast_list[i]["loopnb"],np.array(Info_dict_fast_list[i]["Targets"]),np.array(Info_dict_fast_list[i]["SpaceRotsUn"]))

            all_coeffs_fast_list.append(all_coeffs_fast)

        Sym_list, mass,il_slow_source,ibl_slow_source,il_fast_source,ibl_fast_source = MakeTargetsSyms(Info_dict_slow,Info_dict_fast_list)
        
        nbody = len(mass)



    nbody = params_dict["Geom_Bodies"]["nbody"]
    MomConsImposed = params_dict['Geom_Bodies'] ['MomConsImposed']
    nsyms = params_dict["Geom_Bodies"]["nsyms"]

    # TODO: change this
    nloop = nbody
    mass = np.ones(nloop)

    Sym_list = []

    if (geodim == 2):

        for isym in range(nsyms):
            
            BodyPerm = np.array(params_dict["Geom_Bodies"]["AllSyms"][isym]["BodyPerm"],dtype=int)

            Reflexion = params_dict["Geom_Bodies"]["AllSyms"][isym]["Reflexion"]
            if (Reflexion == "True"):
                s = -1
            elif (Reflexion == "False"):
                s = 1
            else:
                raise ValueError("Reflexion must be True or False")

            rot_angle = 2 * np.pi * params_dict["Geom_Bodies"]["AllSyms"][isym]["RotAngleNum"] / params_dict["Geom_Bodies"]["AllSyms"][isym]["RotAngleDen"]

            SpaceRot = np.array(
                [   [ s*np.cos(rot_angle)   , -s*np.sin(rot_angle)  ],
                    [ np.sin(rot_angle)     , np.cos(rot_angle)     ]   ]
                , dtype = np.float64
            )

            TimeRev_str = params_dict["Geom_Bodies"]["AllSyms"][isym]["TimeRev"]
            if (TimeRev_str == "True"):
                TimeRev = -1
            elif (TimeRev_str == "False"):
                TimeRev = 1
            else:
                raise ValueError("TimeRev must be True or False")

            TimeShiftNum = int(params_dict["Geom_Bodies"]["AllSyms"][isym]["TimeShiftNum"])
            TimeShiftDen = int(params_dict["Geom_Bodies"]["AllSyms"][isym]["TimeShiftDen"])

            Sym_list.append(
                ActionSym(
                    BodyPerm = BodyPerm     ,
                    SpaceRot = SpaceRot     ,
                    TimeRev = TimeRev       ,
                    TimeShiftNum = TimeShiftNum   ,
                    TimeShiftDen = TimeShiftDen   ,
                )
            )

    else:

        raise ValueError("Only compatible with 2D right now")


    if ((LookForTarget) and not(params_dict['Geom_Target'] ['RandomJitterTarget'])) :

        coeff_ampl_min  = 0
        coeff_ampl_o    = 0
        k_infl          = 2
        k_max           = 3

    else:

        coeff_ampl_min  = params_dict["Geom_Random"]["coeff_ampl_min"]
        coeff_ampl_o    = params_dict["Geom_Random"]["coeff_ampl_o"]
        k_infl          = params_dict["Geom_Random"]["k_infl"]
        k_max           = params_dict["Geom_Random"]["k_max"]

    CurrentlyDeveloppingNewStuff = params_dict.get("CurrentlyDeveloppingNewStuff",False)

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

    mul_coarse_to_fine = params_dict["Solver_Discr"]["mul_coarse_to_fine"]

    # Save_All_Coeffs = True
    Save_All_Coeffs = False

    # Save_Init_Pos_Vel_Sol = True
    Save_Init_Pos_Vel_Sol = False

    n_save_pos = 'auto'

    Save_All_Pos = True
    
    plot_extend = 0.

    n_opt = 0
    # n_opt_max = 1
    n_opt_max = params_dict["Solver_Optim"]["n_opt"]
    n_find_max = params_dict["Solver_Optim"]["n_opt"]

    ReconvergeSol = False
    AddNumberToOutputName = True
    
    all_kwargs = Pick_Named_Args_From_Dict(Find_Choreo,dict(globals(),**locals()))

    Find_Choreo(**all_kwargs)

def ChoreoFindFromDict_old(params_dict,Workspace_folder):

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

    np.random.seed(int(time.time()*10000) % 5000)

    geodim = params_dict['Geom_Gen'] ['geodim']

    TwoDBackend = (geodim == 2)
    ParallelBackend = (params_dict['Solver_CLI']['Exec_Mul_Proc'] == "MultiThread")
    GradHessBackend = params_dict['Solver_CLI']['GradHess_backend']

    file_basename = ''
    
    CrashOnError_changevar = False

    LookForTarget = params_dict['Geom_Target'] ['LookForTarget']

    if (LookForTarget) : # IS LIKELY BROKEN !!!!

        Rotate_fast_with_slow = params_dict['Geom_Target'] ['Rotate_fast_with_slow']
        Optimize_Init = params_dict['Geom_Target'] ['Optimize_Init']
        Randomize_Fast_Init =  params_dict['Geom_Target'] ['Randomize_Fast_Init']
            
        nT_slow = params_dict['Geom_Target'] ['nT_slow']
        nT_fast = params_dict['Geom_Target'] ['nT_fast']

        Info_dict_slow_filename = params_dict['Geom_Target'] ["slow_filename"]
        Info_dict_slow, all_pos_slow = load_target_files(Info_dict_slow_filename,Workspace_folder,"slow")

        ncoeff_slow = Info_dict_slow["n_int"] // 2 + 1

        all_coeffs_slow = AllPosToAllCoeffs(all_pos_slow,ncoeff_slow)
        Center_all_coeffs(all_coeffs_slow,Info_dict_slow["nloop"],Info_dict_slow["mass"],Info_dict_slow["loopnb"],np.array(Info_dict_slow["Targets"]),np.array(Info_dict_slow["SpaceRotsUn"]))

        Info_dict_fast_list = []
        all_coeffs_fast_list = []

        for i in range(len(nT_fast)) :

            Info_dict_fast_filename = params_dict['Geom_Target'] ["fast_filenames"] [i]
            Info_dict_fast, all_pos_fast = load_target_files(Info_dict_fast_filename,Workspace_folder,"fast"+str(i))
            Info_dict_fast_list.append(Info_dict_fast)

            ncoeff_fast = Info_dict_fast["n_int"] // 2 + 1

            all_coeffs_fast = AllPosToAllCoeffs(all_pos_fast,ncoeff_fast)
            Center_all_coeffs(all_coeffs_fast,Info_dict_fast_list[i]["nloop"],Info_dict_fast_list[i]["mass"],Info_dict_fast_list[i]["loopnb"],np.array(Info_dict_fast_list[i]["Targets"]),np.array(Info_dict_fast_list[i]["SpaceRotsUn"]))

            all_coeffs_fast_list.append(all_coeffs_fast)

        Sym_list, mass,il_slow_source,ibl_slow_source,il_fast_source,ibl_fast_source = MakeTargetsSyms(Info_dict_slow,Info_dict_fast_list)
        
        nbody = len(mass)


    nbody = params_dict["Geom_Bodies"]["nbody"]
    MomConsImposed = params_dict['Geom_Bodies'] ['MomConsImposed']
    nsyms = params_dict["Geom_Bodies"]["nsyms"]

    nloop = params_dict["Geom_Bodies"]["nloop"]
    mass = np.zeros(nbody)

    for il in range(nloop):
        for ilb in range(len(params_dict["Geom_Bodies"]["Targets"][il])):
            
            ib = params_dict["Geom_Bodies"]["Targets"][il][ilb]
            mass[ib] = params_dict["Geom_Bodies"]["mass"][il]

    Sym_list = []

    if (geodim == 2):

        for isym in range(nsyms):
            
            BodyPerm = np.array(params_dict["Geom_Bodies"]["AllSyms"][isym]["BodyPerm"],dtype=int)

            Reflexion = params_dict["Geom_Bodies"]["AllSyms"][isym]["Reflexion"]
            if (Reflexion == "True"):
                s = -1
            elif (Reflexion == "False"):
                s = 1
            else:
                raise ValueError("Reflexion must be True or False")

            rot_angle = 2 * np.pi * params_dict["Geom_Bodies"]["AllSyms"][isym]["RotAngleNum"] / params_dict["Geom_Bodies"]["AllSyms"][isym]["RotAngleDen"]

            SpaceRot = np.array(
                [   [ s*np.cos(rot_angle)   , -s*np.sin(rot_angle)  ],
                    [ np.sin(rot_angle)     , np.cos(rot_angle)     ]   ]
                , dtype = np.float64
            )

            TimeRev_str = params_dict["Geom_Bodies"]["AllSyms"][isym]["TimeRev"]
            if (TimeRev_str == "True"):
                TimeRev = -1
            elif (TimeRev_str == "False"):
                TimeRev = 1
            else:
                raise ValueError("TimeRev must be True or False")

            TimeShiftNum = int(params_dict["Geom_Bodies"]["AllSyms"][isym]["TimeShiftNum"])
            TimeShiftDen = int(params_dict["Geom_Bodies"]["AllSyms"][isym]["TimeShiftDen"])

            Sym_list.append(
                ActionSym(
                    BodyPerm = BodyPerm     ,
                    SpaceRot = SpaceRot     ,
                    TimeRev = TimeRev       ,
                    TimeShiftNum = TimeShiftNum   ,
                    TimeShiftDen = TimeShiftDen   ,
                )
            )

    else:

        raise ValueError("Only compatible with 2D right now")

    Sym_list = Make_ChoreoSymList_From_ActionSymList(Sym_list, nbody)

    if ((LookForTarget) and not(params_dict['Geom_Target'] ['RandomJitterTarget'])) :

        coeff_ampl_min  = 0
        coeff_ampl_o    = 0
        k_infl          = 2
        k_max           = 3

    else:

        coeff_ampl_min  = params_dict["Geom_Random"]["coeff_ampl_min"]
        coeff_ampl_o    = params_dict["Geom_Random"]["coeff_ampl_o"]
        k_infl          = params_dict["Geom_Random"]["k_infl"]
        k_max           = params_dict["Geom_Random"]["k_max"]

    CurrentlyDeveloppingNewStuff = params_dict.get("CurrentlyDeveloppingNewStuff",False)

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

    mul_coarse_to_fine = params_dict["Solver_Discr"]["mul_coarse_to_fine"]

    # Save_All_Coeffs = True
    Save_All_Coeffs = False

    # Save_Init_Pos_Vel_Sol = True
    Save_Init_Pos_Vel_Sol = False

    n_save_pos = 'auto'

    Save_All_Pos = True
    
    plot_extend = 0.

    n_opt = 0
    # n_opt_max = 1
    n_opt_max = params_dict["Solver_Optim"]["n_opt"]
    n_find_max = params_dict["Solver_Optim"]["n_opt"]

    ReconvergeSol = False
    AddNumberToOutputName = True
    
    all_kwargs = Pick_Named_Args_From_Dict(Find_Choreo,dict(globals(),**locals()))

    Find_Choreo(**all_kwargs)




def ChoreoReadDictAndFind(Workspace_folder,dict_name="choreo_config.json"):

    params_filename = os.path.join(Workspace_folder,dict_name)

    with open(params_filename) as jsonFile:
        params_dict = json.load(jsonFile)

    Exec_Mul_Proc = params_dict['Solver_CLI']['Exec_Mul_Proc']

    n_threads = params_dict['Solver_CLI']['nproc']

    if Exec_Mul_Proc == "MultiProc":

        os.environ['OMP_NUM_THREADS'] = str(1)
        numba.set_num_threads(1)

        print(f"Executing with {n_threads} workers")
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_threads) as executor:
            
            res = []
            for i in range(n_threads):
                res.append(executor.submit(ChoreoFindFromDict,params_dict,Workspace_folder))
                time.sleep(0.01)

    elif Exec_Mul_Proc == "MultiThread":

        os.environ['OMP_NUM_THREADS'] = str(n_threads)
        numba.set_num_threads(n_threads)
        ChoreoFindFromDict(params_dict,Workspace_folder)

    else :

        os.environ['OMP_NUM_THREADS'] = str(1)

        ChoreoFindFromDict(params_dict,Workspace_folder)

def ChoreoReadDictAndFind_old(Workspace_folder,dict_name="choreo_config.json"):

    params_filename = os.path.join(Workspace_folder,dict_name)

    with open(params_filename) as jsonFile:
        params_dict = json.load(jsonFile)

    Exec_Mul_Proc = params_dict['Solver_CLI']['Exec_Mul_Proc']

    n_threads = params_dict['Solver_CLI']['nproc']

    if Exec_Mul_Proc == "MultiProc":

        os.environ['OMP_NUM_THREADS'] = str(1)
        numba.set_num_threads(1)

        print(f"Executing with {n_threads} workers")
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_threads) as executor:
            
            res = []
            for i in range(n_threads):
                res.append(executor.submit(ChoreoFindFromDict_old,params_dict,Workspace_folder))
                time.sleep(0.01)

    elif Exec_Mul_Proc == "MultiThread":

        os.environ['OMP_NUM_THREADS'] = str(n_threads)
        numba.set_num_threads(n_threads)
        ChoreoFindFromDict_old(params_dict,Workspace_folder)

    else :

        os.environ['OMP_NUM_THREADS'] = str(1)
        ChoreoFindFromDict_old(params_dict,Workspace_folder)


