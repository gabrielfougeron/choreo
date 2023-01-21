import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import sys,argparse
import random
import numpy as np
import math as m
import scipy.optimize as opt

import copy
import shutil
import time
import builtins

from choreo.Choreo_scipy_plus import *
from choreo.Choreo_funs import *

def Find_Choreo(
    nbody,
    n_reconverge_it_max,
    ncoeff_init,
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
):
    """

    Finds periodic solutions

    """
    
    print(f'Searching periodic solutions of {nbody:d} bodies.')

    print(f'Processing symmetries for {(n_reconverge_it_max+1):d} convergence levels.')
    ActionSyst = setup_changevar(nbody,ncoeff_init,mass,n_reconverge_it_max,Sym_list=Sym_list,MomCons=MomConsImposed,n_grad_change=n_grad_change,CrashOnIdentity=CrashOnError_changevar)

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

    ncoeff = ActionSyst.ncoeff()
    nint = ActionSyst.nint()

    print(f'Convergence attempt number: {i+1}')
    print(f"    Number of Fourier coeffs: {ncoeff}")
    print(f"    Number of scalar parameters before constraints: {ActionSyst.coeff_to_param().shape[1]}")
    print(f"    Number of scalar parameters after  constraints: {ActionSyst.coeff_to_param().shape[0]}")
    print(f"    ==> Reduction of {100*(1-ActionSyst.coeff_to_param().shape[0]/ActionSyst.coeff_to_param().shape[1]):.2f} %")
    print('')

    all_coeffs_min,all_coeffs_max = Make_Init_bounds_coeffs(nloop,ncoeff,coeff_ampl_o,k_infl,k_max,coeff_ampl_min)

    x_min = ActionSyst.Package_all_coeffs(all_coeffs_min)
    x_max = ActionSyst.Package_all_coeffs(all_coeffs_max)

    rand_eps = coeff_ampl_min
    rand_dim = 0
    for i in range(ActionSyst.coeff_to_param().shape[0]):

        if (abs(x_max[i] - x_min[i]) > rand_eps):
            rand_dim +=1

    print(f'Number of initialization dimensions: {rand_dim}')

    sampler = UniformRandom(d=rand_dim)

    x0 = np.random.random(ActionSyst.param_to_coeff()[0].shape[1])
    xmin = ActionSyst.Compute_MinDist(x0)

    if (xmin < 1e-5):
        # print(xmin)
        # raise ValueError("Init inter body distance too low. There is something wrong with constraints")
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

    while ((n_opt < n_opt_max) and (n_find < n_find_max)):

        AskedForNext = False

        if (Look_for_duplicates and ((n_opt % freq_erase_dict) == 0)):
            
            hash_dict = {}
            _ = SelectFiles_Action(store_folder,hash_dict)

        n_opt += 1
        
        print(f'Optimization attempt number: {n_opt}')

        ActionSyst.current_cvg_lvl = 0
        ncoeff = ActionSyst.ncoeff()
        nint = ActionSyst.nint()

        if (ReconvergeSol):

            x_avg = ActionSyst.Package_all_coeffs(all_coeffs_init)
        
        elif (LookForTarget):

            all_coeffs_avg = ActionSyst.Gen_init_avg_2D(nT_slow,nT_fast,Info_dict_slow,all_coeffs_slow,Info_dict_fast_list,all_coeffs_fast_list,il_slow_source,ibl_slow_source,il_fast_source,ibl_fast_source,Rotate_fast_with_slow,Optimize_Init,Randomize_Fast_Init)

            x_avg = ActionSyst.Package_all_coeffs(all_coeffs_avg)

        else:
            
            x_avg = np.zeros((ActionSyst.coeff_to_param().shape[0]),dtype=np.float64)

        x0 = np.zeros((ActionSyst.coeff_to_param().shape[0]),dtype=np.float64)
        
        xrand = sampler.random()
        
        rand_dim = 0
        for i in range(ActionSyst.coeff_to_param().shape[0]):
            if (abs(x_max[i] - x_min[i]) > rand_eps):
                x0[i] = x_avg[i] + x_min[i] + (x_max[i] - x_min[i])*xrand[rand_dim]
                rand_dim +=1
            else:
                x0[i] = x_avg[i]

        if save_all_inits or (save_first_init and n_opt == 1):

            ActionSyst.Write_Descriptor(x0,'init.json')

            if Save_img :
                ActionSyst.plot_all_2D(x0,nint_plot_img,'init.png',fig_size=img_size,color=color,color_list=color_list)        

            if Save_thumb :
                ActionSyst.plot_all_2D(x0,nint_plot_img,'init_thumb.png',fig_size=thumb_size,color=color,color_list=color_list)        
                
            if Save_anim :
                ActionSyst.plot_all_2D_anim(x0,nint_plot_anim,'init.mp4',nperiod_anim,Plot_trace=Plot_trace_anim,fig_size=vid_size,dnint=dnint,color_list=color_list,color=color)
            
            if Save_Newton_Error :
                ActionSyst.plot_Newton_Error(x0,'init_newton.png')

            if Save_All_Coeffs:

                all_coeffs = ActionSyst.Unpackage_all_coeffs(x0)
                np.save('init_coeffs.npy',all_coeffs)

            if Save_All_Pos:

                if n_save_pos is None:
                    all_pos = ActionSyst.ComputeAllLoopPos(x0)
                elif n_save_pos == 'auto':
                    # TODO : implement auto
                    all_pos = ActionSyst.ComputeAllLoopPos(x0)
                else:
                    all_pos = ActionSyst.ComputeAllLoopPos(x0,n_save_pos)

                np.save('init.npy',all_pos)

            for i in range(n_callback_after_init_list):
                callback_after_init_list[i]()
            
        f0 = ActionSyst.Compute_action_onlygrad(x0)
        best_sol = current_best(x0,f0)

        GoOn = (best_sol.f_norm < max_norm_on_entry)
        
        if not(GoOn):
            print(f"Norm on entry is {best_sol.f_norm:.2e} which is too big.")
        
        i_optim_param = 0

        while GoOn:
            # Set correct optim params
            
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

            if (krylov_method == 'lgmres'):
                jac_options = {'method':krylov_method,'rdiff':rdiff,'outer_k':outer_k,'inner_inner_m':inner_maxiter,'inner_store_outer_Av':store_outer_Av,'inner_tol':inner_tol,'inner_M':inner_M }
            elif (krylov_method == 'gmres'):
                jac_options = {'method':krylov_method,'rdiff':rdiff,'outer_k':outer_k,'inner_tol':inner_tol,'inner_M':inner_M }
            else:
                jac_options = {'method':krylov_method,'rdiff':rdiff,'outer_k':outer_k,'inner_tol':inner_tol,'inner_M':inner_M }
 
            if (Use_exact_Jacobian):

                FGrad = lambda x,dx : ActionSyst.Compute_action_hess_mul(x,dx)
                jacobian = ExactKrylovJacobian(exactgrad=FGrad,**jac_options)

            else: 
                jacobian = scipy.optimize.nonlin.KrylovJacobian(**jac_options)


            def optim_callback(x,f,f_norm):

                AskedForNext = False
                
                best_sol.update(x,f,f_norm)

                for i in range(n_optim_callback_list):

                    AskedForNext = (AskedForNext or optim_callback_list[i](x,f,f_norm,ActionSyst))

                return AskedForNext

            try : 
                
                x0 = np.copy(best_sol.x)
                opt_result , info = nonlin_solve_pp(F=F,x0=x0,jacobian=jacobian,verbose=disp_scipy_opt,maxiter=maxiter,f_tol=gradtol,line_search=line_search,callback=optim_callback,raise_exception=False,smin=linesearch_smin,full_output=True)

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
                    
                    all_coeffs_coarse = ActionSyst.Unpackage_all_coeffs(best_sol.x)
                    ncoeff_coarse = ActionSyst.ncoeff()
                    
                    ActionSyst.current_cvg_lvl += 1
                    ncoeff_fine = ActionSyst.ncoeff()

                    all_coeffs_fine = np.zeros((ActionSyst.nloop,ndim,ncoeff_fine,2),dtype=np.float64)
                    for k in range(ncoeff_coarse):
                        all_coeffs_fine[:,:,k,:] = all_coeffs_coarse[:,:,k,:]
                        
                    x_fine = ActionSyst.Package_all_coeffs(all_coeffs_fine)
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
                    
                    ncoeff = ActionSyst.ncoeff()
                    nint = ActionSyst.nint()    
                    
                if GoOn and NeedsChangeOptimParams:
                    
                    print('Changing optimizer parameters.')
                    
                    i_optim_param += 1
                
            print('')
        
        print('')

    print('Done!')

def GenSymExample(
    nbody,
    ncoeff_init,
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

    ActionSyst = setup_changevar(nbody,ncoeff_init,mass,n_reconverge_it_max,Sym_list=Sym_list,MomCons=MomConsImposed,n_grad_change=n_grad_change,CrashOnIdentity=CrashOnError_changevar)

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

    ncoeff = ActionSyst.ncoeff()
    nint = ActionSyst.nint()

    print(f'Convergence attempt number: {i+1}')
    print(f"    Number of Fourier coeffs: {ncoeff}")
    print(f"    Number of scalar parameters before constraints: {ActionSyst.coeff_to_param().shape[1]}")
    print(f"    Number of scalar parameters after  constraints: {ActionSyst.coeff_to_param().shape[0]}")
    print(f"    ==> Reduction of {100*(1-ActionSyst.coeff_to_param().shape[0]/ActionSyst.coeff_to_param().shape[1]):.2f} %")
    print('')

    x0 = np.random.random(ActionSyst.param_to_coeff().shape[1])
    xmin = ActionSyst.Compute_MinDist(x0)
    if (xmin < 1e-5):
        # print(xmin)
        # raise ValueError("Init inter body distance too low. There is something wrong with constraints")
        print("")
        print(f"Init minimum inter body distance too low : {xmin:.2e}.")
        print("There is likely something wrong with constraints.")
        print("")

        return False

    all_coeffs_min,all_coeffs_max = Make_Init_bounds_coeffs(ActionSyst.nloop,ncoeff,coeff_ampl_o,k_infl,k_max,coeff_ampl_min)

    x_min = ActionSyst.Package_all_coeffs(all_coeffs_min)
    x_max = ActionSyst.Package_all_coeffs(all_coeffs_max)

    rand_eps = coeff_ampl_min
    rand_dim = 0
    for i in range(ActionSyst.coeff_to_param().shape[0]):
        if (abs(x_max[i] - x_min[i]) > rand_eps):
            rand_dim +=1

    sampler = UniformRandom(d=rand_dim)

    if (LookForTarget):
        
        all_coeffs_avg = ActionSyst.Gen_init_avg_2D(nT_slow,nT_fast,Info_dict_slow,all_coeffs_slow,Info_dict_fast_list,all_coeffs_fast_list,il_slow_source,ibl_slow_source,il_fast_source,ibl_fast_source,Rotate_fast_with_slow,Optimize_Init,Randomize_Fast_Init)    

        x_avg = ActionSyst.Package_all_coeffs(all_coeffs_avg)
    
    else:
        
        x_avg = np.zeros((ActionSyst.coeff_to_param().shape[0]),dtype=np.float64)


    x0 = np.zeros((ActionSyst.coeff_to_param().shape[0]),dtype=np.float64)
    
    xrand = sampler.random()
    
    rand_dim = 0
    for i in range(ActionSyst.coeff_to_param().shape[0]):
        if (abs(x_max[i] - x_min[i]) > rand_eps):
            x0[i] = x_avg[i] + x_min[i] + (x_max[i] - x_min[i])*xrand[rand_dim]
            rand_dim +=1
        else:
            x0[i] = x_avg[i]

    ActionSyst.Write_Descriptor(x0,'init.json',extend=plot_extend)

    if Save_img :
        ActionSyst.plot_all_2D(x0,nint_plot_img,'init.png',fig_size=img_size,color=color,color_list=color_list)        

    if Save_anim :
        ActionSyst.plot_all_2D_anim(x0,nint_plot_anim,'init.mp4',nperiod_anim,Plot_trace=Plot_trace_anim,fig_size=vid_size,dnint=dnint,color_list=color_list,color=color)

    if Save_All_Coeffs:
        ActionSyst.all_coeffs = Unpackage_all_coeffs(x0)
        np.save('init_coeffs.npy',all_coeffs)

    if Save_All_Pos:
        if n_save_pos is None:
            ActionSyst.all_pos_b = ComputeAllLoopPos(x0)
        elif n_save_pos == 'auto':
            # TODO : implement auto
            ActionSyst.all_pos_b = ComputeAllLoopPos(x0)
        else:
            ActionSyst.all_pos_b = ComputeAllLoopPos(x0,n_save_pos)

        np.save('init.npy',all_pos_b)

    return True

def Speed_test(
    nbody,
    n_reconverge_it_max,
    ncoeff_init,
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
    
    ActionSyst = setup_changevar(nbody,ncoeff_init,mass,n_reconverge_it_max,Sym_list=Sym_list,MomCons=MomConsImposed,n_grad_change=n_grad_change,CrashOnIdentity=CrashOnError_changevar)

    nbi_tot = 0
    for il in range(ActionSyst.nloop):
        for ilp in range(il+1,ActionSyst.nloop):
            nbi_tot += ActionSyst.loopnb[il]*ActionSyst.loopnb[ilp]
        nbi_tot += ActionSyst.loopnbi[il]
    nbi_naive = (nbody*(nbody-1))//2

    ActionSyst.current_cvg_lvl = n_reconverge_it_max
    ncoeff = ActionSyst.ncoeff()
    nint = ActionSyst.nint()

    all_coeffs_min,all_coeffs_max = Make_Init_bounds_coeffs(ActionSyst.nloop,ncoeff,coeff_ampl_o,k_infl,k_max,coeff_ampl_min)

    x_min = ActionSyst.Package_all_coeffs(all_coeffs_min)
    x_max = ActionSyst.Package_all_coeffs(all_coeffs_max)

    rand_eps = coeff_ampl_min
    rand_dim = 0
    for i in range(ActionSyst.coeff_to_param().shape[0]):

        if (abs(x_max[i] - x_min[i]) > rand_eps):
            rand_dim +=1

    sampler = UniformRandom(d=rand_dim)

    x0 = np.random.random(ActionSyst.coeff_to_param().shape[1])
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
        
        x_avg = np.zeros((ActionSyst.coeff_to_param().shape[0]),dtype=np.float64)

        
    x0 = np.zeros((ActionSyst.coeff_to_param().shape[0]),dtype=np.float64)
    
    xrand = sampler.random()
    
    rand_dim = 0
    for i in range(ActionSyst.coeff_to_param().shape[0]):
        if (abs(x_max[i] - x_min[i]) > rand_eps):
            x0[i] = x_avg[i] + x_min[i] + (x_max[i] - x_min[i])*xrand[rand_dim]
            rand_dim +=1
        else:
            x0[i] = x_avg[i]


    dx = np.random.random(x0.shape)

    beg = time.perf_counter()
    for itest in range(n_test):

        f0 = ActionSyst.Compute_action_onlygrad(x0)
        hess = ActionSyst.Compute_action_hess_mul(x0,dx)
        # toto = ActionSyst.Unpackage_all_coeffs(x0) 

    end = time.perf_counter()
    tot_time += (end-beg)

    print(f'Total time s : {tot_time}')
        
