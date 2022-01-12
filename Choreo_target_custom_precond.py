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
import scipy.sparse.linalg
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import cnames
from matplotlib import animation
import copy
import shutil
import time

from Choreo_funs import *
from scipy_root_plus import ExactKrylovJacobian,inv_op



def main(preprint_msg=''):
    
    def print(*args, **kwargs):
        """My custom print() function."""
        # Adding new arguments to the print function signature 
        # is probably a bad idea.
        # Instead consider testing if custom argument keywords
        # are present in kwargs
        __builtins__.print(preprint_msg,end='')
        return __builtins__.print(*args, **kwargs)
    

    # ~ slow_base_filename = './data/2_cercle.npy'
    # ~ slow_base_filename = './data/3_cercle.npy'
    slow_base_filename = './data/3_huit.npy'
    # ~ slow_base_filename = './data/3_heart.npy'

    # ~ fast_base_filename = './data/1_lone_wolf.npy'
    fast_base_filename = './data/2_cercle.npy'
    # ~ fast_base_filename = './data/3_cercle.npy'
    # ~ fast_base_filename = './data/3_huit.npy'
    # ~ fast_base_filename = './data/3_heart.npy'
    # ~ fast_base_filename = './data/3_dbl_heart.npy'

    nTf = 101
    # ~ nTf = 37
    nbs = 3
    nbf = 2

    # ~ Rotate_fast_with_slow = True
    Rotate_fast_with_slow = False

    # ~ Optimize_Init = True
    Optimize_Init = False

    # ~ Randomize_Fast_Init = True
    Randomize_Fast_Init = False

    all_coeffs_slow_load = np.load(slow_base_filename)
    all_coeffs_fast_load = np.load(fast_base_filename)
    all_coeffs_lone_wolf = np.load('./data/1_lone_wolf.npy')

    if (all_coeffs_slow_load.shape[0] != 1):
        raise ValueError("Several loops in slow base")

    if (all_coeffs_fast_load.shape[0] != 1):
        raise ValueError("Several loops in fast base")

    if (all_coeffs_slow_load.shape[0] != all_coeffs_fast_load.shape[0] ):
        raise ValueError("Fast and slow have different number of loops")

    nbody =  nbs * nbf

    mass = np.ones((nbody))

    Sym_list = []

    nbpl = [nbody]
    the_lcm = m.lcm(*nbpl)
    SymName = None
    Sym_list,nbody = Make2DChoreoSymManyLoops(nbpl=nbpl,SymName=SymName)

    # ~ rot_angle = twopi * nbf /  nTf
    # ~ s = 1

    # ~ Sym_list.append(ChoreoSym(
        # ~ LoopTarget=0,
        # ~ LoopSource=0,
        # ~ SpaceRot = np.array([[s*np.cos(rot_angle),-s*np.sin(rot_angle)],[np.sin(rot_angle),np.cos(rot_angle)]],dtype=np.float64),
        # ~ TimeRev=1,
        # ~ TimeShift=fractions.Fraction(numerator=1,denominator=nTf)
        # ~ ))


    MomConsImposed = True
    # ~ MomConsImposed = False

    store_folder = './Target_res/'
    store_folder = store_folder+str(nbody)
    if not(os.path.isdir(store_folder)):
        os.makedirs(store_folder)

    Use_exact_Jacobian = True
    # ~ Use_exact_Jacobian = False

    # ~ Use_deflation = True
    Use_deflation = False

    Look_for_duplicates = True
    # ~ Look_for_duplicates = False

    Check_loop_dist = True
    # ~ Check_loop_dist = False

    # ~ Penalize_Escape = True
    Penalize_Escape = False

    save_init = False
    # ~ save_init = True

    save_approx = False
    # ~ save_approx = True

    Save_img = True
    # ~ Save_img = False

    # ~ img_size = (12,12) # Image size in inches
    img_size = (8,8) # Image size in inches

    nint_plot_img = 10000

    Save_anim = True
    # ~ Save_anim = False

    vid_size = (8,8) # Image size in inches
    nint_plot_anim = 2*2*2*3*3*5 * 6 *3
    # ~ nperiod_anim = 1./nbody

    color = "body"
    # ~ color = "loop"
    # ~ color = "velocity"
    # ~ color = "all"

    try:
        the_lcm
    except NameError:
        period_div = 1.
    else:
        period_div = the_lcm


    nperiod_anim = 1./period_div

    Plot_trace_anim = True
    # ~ Plot_trace_anim = False

    n_reconverge_it_max = 3
    # ~ n_reconverge_it_max = 1

    # ~ ncoeff_init = 102
    # ~ ncoeff_init = 800
    # ~ ncoeff_init = 201   
    # ~ ncoeff_init = 300   
    # ~ ncoeff_init = 600
    ncoeff_init = 900
    # ~ ncoeff_init = 1800
    # ~ ncoeff_init = 1206
    # ~ ncoeff_init = 90
    
    ncoeff_precond = 300

    # ~ disp_scipy_opt = False
    disp_scipy_opt = True

    Newt_err_norm_max = 1e-9
    Newt_err_norm_max_save = Newt_err_norm_max * 100

    # ~ Save_Bad_Sols = True
    Save_Bad_Sols = False

    duplicate_eps = 1e-8

    krylov_method = 'lgmres'
    # ~ krylov_method = 'gmres'
    # ~ krylov_method = 'bicgstab'
    # ~ krylov_method = 'cgs'
    # ~ krylov_method = 'minres'

    # ~ line_search = 'armijo'
    line_search = 'wolfe'

    escape_fac = 1e0
    # ~ escape_fac = 1e-1
    # ~ escape_fac = 1e-2
    # ~ escape_fac = 1e-3
    # ~ escape_fac = 1e-4
    # ~ escape_fac = 1e-5
    # ~ escape_fac = 0
    escape_min_dist = 1
    escape_pow = 2.0
    # ~ escape_pow = 2.5
    # ~ escape_pow = 1.5
    # ~ escape_pow = 0.5

    n_grad_change = 1.
    # ~ n_grad_change = 1.5

    print('Searching periodic solutions of {:d} bodies'.format(nbody))
    # ~ print('Processing symmetries for {:d} convergence levels ...'.format(n_reconverge_it_max+1))

    print('Processing symmetries for {0:d} convergence levels'.format(n_reconverge_it_max+1))
    callfun = setup_changevar(nbody,ncoeff_init,mass,n_reconverge_it_max,Sym_list=Sym_list,MomCons=MomConsImposed,n_grad_change=n_grad_change)
    
    
    nbody_precond =  nbs
    mass_precond = np.ones((nbody_precond))*nbf
    Sym_list_precond = []
    nbpl_precond = [nbody_precond]
    SymName_precond = None
    Sym_list_precond,nbody_precond = Make2DChoreoSymManyLoops(nbpl=nbpl_precond,SymName=SymName_precond)
    callfun_precond = setup_changevar(nbody_precond,ncoeff_precond,mass_precond,0,Sym_list=Sym_list_precond,MomCons=MomConsImposed,n_grad_change=n_grad_change)

    print('')

    args = callfun[0]
    args['escape_fac'] = escape_fac
    args['escape_min_dist'] = escape_min_dist
    args['escape_pow'] = escape_pow

    nloop = args['nloop']
    loopnb = args['loopnb']
    loopnbi = args['loopnbi']
    nbi_tot = 0
    for il in range(nloop):
        for ilp in range(il+1,nloop):
            nbi_tot += loopnb[il]*loopnb[ilp]
        nbi_tot += loopnbi[il]
    nbi_naive = (nbody*(nbody-1))//2

    print('Imposed constraints lead to the detection of :')
    print('    {:d} independant loops'.format(nloop))
    print('    {0:d} binary interactions'.format(nbi_tot))
    print('    ==> reduction of {0:f} % wrt the {1:d} naive binary iteractions'.format(100*(1-nbi_tot/nbi_naive),nbi_naive))
    print('')

    # ~ for i in range(n_reconverge_it_max+1):
    for i in [0]:
        
        args = callfun[0]
        print('Convergence attempt number : ',i+1)
        print('    Number of scalar parameters before constraints : ',args['coeff_to_param_list'][i].shape[1])
        print('    Number of scalar parameters after  constraints : ',args['coeff_to_param_list'][i].shape[0])
        print('    Reduction of ',100*(1-args['coeff_to_param_list'][i].shape[0]/args['coeff_to_param_list'][i].shape[1]),' %')
        print('')
        

    x0 = np.random.random(callfun[0]['param_to_coeff_list'][i].shape[1])
    xmin = Compute_MinDist(x0,callfun)
    if (xmin < 1e-5):
        print(xmin)
        raise ValueError("Init inter body distance too low. There is something wrong with constraints")

    # ~ filehandler = open(store_folder+'/callfun_list.pkl',"wb")
    # ~ pickle.dump(callfun_list,filehandler)

    if (Penalize_Escape):

        Action_grad_mod = Compute_action_onlygrad_escape

    elif (Use_deflation):
        print('Loading previously saved sols as deflation vectors')
        
        Init_deflation(callfun)
        Load_all_defl(store_folder,callfun)
        
        Action_grad_mod = Compute_action_defl
        
    else:
        
        Action_grad_mod = Compute_action_onlygrad

    callfun[0]["current_cvg_lvl"] = 0
    ncoeff = callfun[0]["ncoeff_list"][callfun[0]["current_cvg_lvl"]]
    nint = callfun[0]["nint_list"][callfun[0]["current_cvg_lvl"]]

    all_coeffs_avg = Gen_init_avg(nTf,nbs,nbf,1,ncoeff,all_coeffs_slow_load,all_coeffs_fast_load,callfun,Rotate_fast_with_slow,Optimize_Init,Randomize_Fast_Init)
    all_coeffs_avg_precond = Gen_init_avg(nTf,nbs,1,nbf,ncoeff_precond,all_coeffs_slow_load,all_coeffs_lone_wolf,callfun_precond,Rotate_fast_with_slow=False,Optimize_Init=False,Randomize_Fast_Init=False)
    
    callfun_precond[0]["current_cvg_lvl"] = 0
    ncoeff_precond = callfun_precond[0]["ncoeff_list"][callfun_precond[0]["current_cvg_lvl"]]
    x0_precond = Package_all_coeffs(all_coeffs_avg_precond,callfun_precond)
    nparam_precond = x0_precond.shape[0]
    

    # ~ ActHessPrecond_LinOpt = Compute_action_hess_LinOpt(x0_precond,callfun_precond)
    ActHessPrecond_LinOpt = Compute_action_hess_LinOpt_precond(x0_precond,callfun,callfun_precond)

    print(nparam_precond)
    
    nparam = args['coeff_to_param_list'][i].shape[0]

    ActHessPrecond_dense = ActHessPrecond_LinOpt * np.eye(nparam)
    # ~ print(dir(ActHessPrecond_dense))
    
    atol_nnz = 1e-11
    # ~ atol_nnz = 0
    for i in range(nparam):
        for j in range(nparam):
            if (abs(ActHessPrecond_dense[i,j]) < atol_nnz):
                ActHessPrecond_dense[i,j] = 0.
    
    ActHessPrecond_csr = scipy.sparse.csr_matrix(ActHessPrecond_dense)
    
    print(ActHessPrecond_csr.nnz)
    print(ActHessPrecond_csr.nnz/(nparam*nparam))
    
    tstart = time.perf_counter()
    precond = scipy.sparse.linalg.spilu(ActHessPrecond_csr)
    tstop = time.perf_counter()
    print(tstop-tstart)
    
    tstart = time.perf_counter()
    precond = scipy.sparse.linalg.spilu(ActHessPrecond_LinOpt)
    tstop = time.perf_counter()
    print(tstop-tstart)
    
    # ~ print(ActHessPrecond_csr)
    
    sys.exit(0)
    # ~ scipy.sparse.linalg.spilu(

    # ~ ActHessPrecond_csr = ActHessPrecond.tocsr()

    # ~ print(ActHessPrecond_csr)


    coeff_ampl_o=1e-16
    k_infl=1
    k_max=200
    coeff_ampl_min=1e-16

    all_coeffs_min,all_coeffs_max = Make_Init_bounds_coeffs(nloop,ncoeff,coeff_ampl_o,k_infl,k_max,coeff_ampl_min)

    x_min = Package_all_coeffs(all_coeffs_min,callfun)
    x_max = Package_all_coeffs(all_coeffs_max,callfun)
    x_avg = Package_all_coeffs(all_coeffs_avg,callfun)

    rand_eps = coeff_ampl_min
    rand_dim = 0
    for i in range(callfun[0]['coeff_to_param_list'][0].shape[0]):
        if ((x_max[i] - x_min[i]) > rand_eps):
            rand_dim +=1

    print('Number of initialization dimensions : ',rand_dim)

    sampler = UniformRandom(d=rand_dim)

    freq_erase_dict = 1000
    hash_dict = {}

    n_opt = 0
    n_opt_max = 1
    # ~ n_opt_max = 1e10
    while (n_opt < n_opt_max):
        
        if ((n_opt % freq_erase_dict) == 0):
            
            hash_dict = {}
            _ = SelectFiles_Action(store_folder,hash_dict)

        n_opt += 1
        
        print('Optimization attempt number : ',n_opt)

        callfun[0]["current_cvg_lvl"] = 0
        ncoeff = callfun[0]["ncoeff_list"][callfun[0]["current_cvg_lvl"]]
        nint = callfun[0]["nint_list"][callfun[0]["current_cvg_lvl"]]
        
        x0 = np.zeros((callfun[0]['coeff_to_param_list'][callfun[0]["current_cvg_lvl"]].shape[0]),dtype=np.float64)
        
        if (n_opt == 1) :
                
            for i in range(callfun[0]['coeff_to_param_list'][callfun[0]["current_cvg_lvl"]].shape[0]):
                x0[i] = x_avg[i]
            
        else :
            
            xrand = sampler.random()
            
            rand_dim = 0
            for i in range(callfun[0]['coeff_to_param_list'][callfun[0]["current_cvg_lvl"]].shape[0]):
                if ((x_max[i] - x_min[i]) > rand_eps):
                    x0[i] = x_avg[i] + x_min[i] + (x_max[i] - x_min[i])*xrand[rand_dim]
                    rand_dim +=1
                else:
                    x0[i] = x_avg[i]

        if save_init:

            if Save_img :
                plot_all_2D(x0,nint_plot_img,callfun,'init.png',fig_size=img_size,color=color)        
                
            if Save_anim :
                plot_all_2D_anim(x0,nint_plot_anim,callfun,'init.mp4',nperiod_anim,Plot_trace=Plot_trace_anim,fig_size=vid_size)
                
            print(1/0)
            
        f0 = Action_grad_mod(x0,callfun)
        best_sol = current_best(x0,f0)

        print('Initialization Action Grad Norm : ',best_sol.f_norm)

        # ~ gradtol = 1e-1
        # ~ gradtol = 1e-2
        gradtol = 1e-9
        # ~ gradtol = 1e-11
        # ~ maxiter = 500
        maxiter = 25000

        try : 
            
            # ~ rdiff = 1e-7
            # ~ rdiff = 0   
            rdiff = None
            # ~ rdiff = 1e-1
            
            outer_k = 5
            # ~ outer_k = 0
            
            # Classical root
            # ~ opt_result = opt.root(fun=Action_grad_mod,x0=x0,args=callfun,method='krylov', options={'line_search':line_search,'disp':disp_scipy_opt,'maxiter':maxiter,'fatol':gradtol,'jac_options':{'method':krylov_method,'rdiff':rdiff }},callback=best_sol.update)
            
            if (Use_exact_Jacobian):

                # Non-classical nonlin_solve with exact Krylov Jacobian
                F = lambda x : Action_grad_mod(x,callfun)
                FGrad = lambda x,dx : Compute_action_hess_mul(x,dx,callfun)
                # ~ def FGrad(x,dx): 
                    # ~ callfun[0]["Do_Pos_FFT"] = False
                    # ~ res = Compute_action_hess_mul(x,dx,callfun)
                    # ~ callfun[0]["Do_Pos_FFT"] = True
                    # ~ return res
                    
                ActHessPrecond_LinOpt = Compute_action_hess_LinOpt_precond(x0_precond,callfun,callfun_precond)
                
                ActHessPrecond = inv_op(ActHessPrecond_LinOpt)
                
                
                # ~ inner_M = None
                jac_options = {'method':krylov_method,'rdiff':rdiff,'outer_k':outer_k}
                inner_M = ActHessPrecond

                # ~ jac_options = {'method':krylov_method,'rdiff':rdiff,'outer_k':outer_k ,'inner_M':inner_M}
                jac_options = {'method':krylov_method,'rdiff':rdiff,'outer_k':outer_k}
                jacobian = ExactKrylovJacobian(exactgrad=FGrad,**jac_options)

                opt_result = opt.nonlin.nonlin_solve(F=F,x0=x0,jacobian=jacobian,verbose=disp_scipy_opt,maxiter=maxiter,f_tol=gradtol,line_search=line_search,callback=best_sol.update,raise_exception=False)
            
            else: 
                
                # Classical nonlin_solve with standard Krylov Jacobian
                F = lambda x : Action_grad_mod(x,callfun)

                jac_options = {'method':krylov_method,'rdiff':rdiff,'outer_k':outer_k }
                jacobian = opt.nonlin.KrylovJacobian(**jac_options)

                opt_result = opt.nonlin.nonlin_solve(F=F,x0=x0,jacobian=jacobian,verbose=disp_scipy_opt,maxiter=maxiter,f_tol=gradtol,line_search=line_search,callback=best_sol.update,raise_exception=False)

            Go_On = True

        except Exception as exc:
            
            print(exc)
            print("Value Error occured, skipping.")
            Go_On = False
            raise(exc)

        if (Check_loop_dist and Go_On):
            
            Escaped,_ = Detect_Escape(best_sol.x,callfun)
            Go_On = not(Escaped)

            if not(Go_On):
                print('One loop escaped. Starting over')    
        
        if (Go_On):
            
            print('Approximate solution found ! Action Grad Norm : ',best_sol.f_norm)
            
            PreciseEnough = (best_sol.f_norm < (gradtol*100))
            ErrorOccured = False

            Found_duplicate = False

            if (Look_for_duplicates and PreciseEnough):
                
                print('Checking Duplicates.')

                Action,GradAction = Compute_action(best_sol.x,callfun)
                
                Found_duplicate,file_path = Check_Duplicates(best_sol.x,callfun,hash_dict,store_folder,duplicate_eps)
                
            else:
                
                Found_duplicate = False
                
            if (ErrorOccured):
                
                print("Value Error occured, skipping.")
                
            elif (not(PreciseEnough)):
            
                print('Initial convergence not good enough')   
                print('Restarting')   
                
                
            elif (Found_duplicate):
            
                print('Found Duplicate !')   
                print('Path : ',file_path)
                
            else:

                print('Reconverging solution')
                
                Newt_err_norm = 1.
                
                while ((Newt_err_norm > Newt_err_norm_max) and (callfun[0]["current_cvg_lvl"] < n_reconverge_it_max) and Go_On):
                            
                    all_coeffs_old = Unpackage_all_coeffs(best_sol.x,callfun)
                    
                    ncoeff = callfun[0]["ncoeff_list"][callfun[0]["current_cvg_lvl"]]
                    ncoeff_new = ncoeff * 2

                    all_coeffs = np.zeros((nloop,ndim,ncoeff_new,2),dtype=np.float64)
                    for k in range(ncoeff):
                        all_coeffs[:,:,k,:] = all_coeffs_old[:,:,k,:]   
                    
                    callfun[0]["current_cvg_lvl"] += 1
                    x0 = Package_all_coeffs(all_coeffs,callfun)
                    
                    ncoeff = callfun[0]["ncoeff_list"][callfun[0]["current_cvg_lvl"]]
                    nint = callfun[0]["nint_list"][callfun[0]["current_cvg_lvl"]]
                    
                    f0 = Action_grad_mod(x0,callfun)
                    best_sol = current_best(x0,f0)
                    
                    print('')
                    print('After Resize lvl '+str(callfun[0]["current_cvg_lvl"])+' : Action Grad Norm : ',best_sol.f_norm)
                                    

                    # ~ maxiter = 50
                    # ~ maxiter = 10000
                    maxiter = 2000
                    gradtol = 1e-13
                    
                    # ~ outer_k = 0
                    outer_k = 10
                    # ~ outer_k = 100

                    try : 

                        # Classical root
                        # ~ opt_result = opt.root(fun=Action_grad_mod,x0=x0,args=callfun,method='krylov', options={'line_search':line_search,'disp':disp_scipy_opt,'maxiter':maxiter,'fatol':gradtol,'jac_options':{'method':krylov_method,'rdiff':rdiff }},callback=best_sol.update)
                        
                        if (Use_exact_Jacobian):

                            # Non-classical nonlin_solve with exact Krylov Jacobian
                            F = lambda x : Action_grad_mod(x,callfun)
                            FGrad = lambda x,dx : Compute_action_hess_mul(x,dx,callfun)
                            # ~ def FGrad(x,dx): 
                                # ~ callfun[0]["Do_Pos_FFT"] = False
                                # ~ res = Compute_action_hess_mul(x,dx,callfun)
                                # ~ callfun[0]["Do_Pos_FFT"] = True
                                # ~ return res

                            jac_options = {'method':krylov_method,'rdiff':rdiff,'outer_k':outer_k }
                            jacobian = ExactKrylovJacobian(exactgrad=FGrad,**jac_options)

                            opt_result = opt.nonlin.nonlin_solve(F=F,x0=x0,jacobian=jacobian,verbose=disp_scipy_opt,maxiter=maxiter,f_tol=gradtol,line_search=line_search,callback=best_sol.update,raise_exception=False)
                
                        else:
                            
                            # Classical nonlin_solve with standard Krylov Jacobian
                            F = lambda x : Action_grad_mod(x,callfun)

                            jac_options = {'method':krylov_method,'rdiff':rdiff,'outer_k':outer_k }
                            jacobian = opt.nonlin.KrylovJacobian(**jac_options)

                            opt_result = opt.nonlin.nonlin_solve(F=F,x0=x0,jacobian=jacobian,verbose=disp_scipy_opt,maxiter=maxiter,f_tol=gradtol,line_search=line_search,callback=best_sol.update,raise_exception=False)
            
                        all_coeffs = Unpackage_all_coeffs(best_sol.x,callfun)
                        
                        print('Opt Action Grad Norm : ',best_sol.f_norm)
                    
                        Newt_err = Compute_Newton_err(best_sol.x,callfun)
                        Newt_err_norm = np.linalg.norm(Newt_err)/nint
                        
                        print('Newton Error : ',Newt_err_norm)
                        
                        if (save_approx):
                            
                            max_num_file = 0
                            
                            for filename in os.listdir(store_folder):
                                file_path = os.path.join(store_folder, filename)
                                file_root, file_ext = os.path.splitext(os.path.basename(file_path))
                                
                                if (file_ext == '.txt' ):
                                    try:
                                        max_num_file = max(max_num_file,int(file_root))
                                    except:
                                        pass
                                
                            max_num_file = max_num_file + 1

                            filename_output = store_folder+'/'+str(max_num_file)+'_'+str(callfun[0]["current_cvg_lvl"])
                            
                            plot_all_2D(best_sol.x,nint_plot_img,callfun,filename_output+'.png',fig_size=img_size,color=color)
                            
                        SaveSol = (Newt_err_norm < Newt_err_norm_max_save)
                                        
                        if (Check_loop_dist):
                            
                            Escaped,_ = Detect_Escape(best_sol.x,callfun)
                            Go_On = Go_On and not(Escaped)
                            
                            if not(Go_On):
                                print('One loop escaped. Starting over')    
                                        
                        if (Look_for_duplicates):
                            
                            print('Checking Duplicates.')
                            
                            Action,GradAction = Compute_action(best_sol.x,callfun)
                    
                            Found_duplicate,file_path = Check_Duplicates(best_sol.x,callfun,hash_dict,store_folder,duplicate_eps)
                            
                            Go_On = Go_On and not(Found_duplicate)
                            
                            if (Found_duplicate):
                            
                                print('Found Duplicate !')  
                                print('Path : ',file_path) 

                    except Exception as exc:

                        print(exc)
                        print("Value Error occured, skipping.")
                        Go_On = False
                        SaveSol = False


                if (not(SaveSol) and Go_On):
                    print('Newton Error too high, discarding solution')
            
                if (((SaveSol) or (Save_Bad_Sols)) and Go_On):
                            
                    if (Look_for_duplicates):
                        
                        print('Checking Duplicates.')
                        
                        Action,GradAction = Compute_action(best_sol.x,callfun)
                
                        Found_duplicate,file_path = Check_Duplicates(best_sol.x,callfun,hash_dict,store_folder,duplicate_eps)
                        
                    else:
                        Found_duplicate = False
                    
                    
                    if (Found_duplicate):
                    
                        print('Found Duplicate !')  
                        print('Path : ',file_path) 
                    
                    else:
                        
                        max_num_file = 0
                        
                        for filename in os.listdir(store_folder):
                            file_path = os.path.join(store_folder, filename)
                            file_root, file_ext = os.path.splitext(os.path.basename(file_path))
                            
                            if (file_ext == '.txt' ):
                                try:
                                    max_num_file = max(max_num_file,int(file_root))
                                except:
                                    pass
                            
                        max_num_file = max_num_file + 1
                        
                        filename_output = store_folder+'/'+str(max_num_file)
                        
                        if not(SaveSol):
                            # ~ filename_output = filename_output + '_bad'
                            filename_output = 'bad'
                        
                        print('Saving solution as '+filename_output+'.*')
                 
                        Write_Descriptor(best_sol.x,callfun,filename_output+'.txt')
                        
                        if Save_img :
                            plot_all_2D(best_sol.x,nint_plot_img,callfun,filename_output+'.png',fig_size=img_size,color=color)
                            
                        if Save_anim :
                            plot_all_2D_anim(best_sol.x,nint_plot_anim,callfun,filename_output+'.mp4',nperiod_anim,Plot_trace=Plot_trace_anim,fig_size=vid_size)
                        
                        all_coeffs = Unpackage_all_coeffs(best_sol.x,callfun)
                        np.save(filename_output+'.npy',all_coeffs)

        if (Use_deflation):
            
            # ~ HessMat = Compute_action_hess_LinOpt(best_sol.x,callfun)
            # ~ w ,v = sp.linalg.eigsh(HessMat,k=10,which='SA')
            
            # ~ print(w)
            
            Newt_err = Compute_Newton_err(best_sol.x,callfun)
            Newt_err_norm = np.linalg.norm(Newt_err)/nint
            
            if(Newt_err_norm < Newt_err_norm_max_save):
                
                print("Deflation coeff at sol : ",Compute_defl_fac(best_sol.x,callfun))
                print("Modified Action Grad at sol : ",np.linalg.norm(Compute_action_defl(best_sol.x,callfun)))
                
                all_coeffs = Unpackage_all_coeffs(best_sol.x,callfun)
                Add_deflation_coeffs(all_coeffs,callfun)            
                
                print("Added to deflation vects list")
                print("Length of deflation vects list : ",len(callfun[0]['defl_vec_list']))
        
        # ~ print('Press Enter to continue ...')
        # ~ pause = input()

        print('')
        print('')
        print('')


    print('Done !')


if __name__ == "__main__":
    
    tstart = time.perf_counter()
    
    parser = argparse.ArgumentParser(description='Welcome to the targeted choreography finder')
    parser.add_argument('-pp','--preprint_msg',nargs=1,type=None,required=False,default=None,help='Adds a systematic message before every print')
    
    args = parser.parse_args(sys.argv[1:])
    
    if args.preprint_msg is None:
        
        main()
        
    else:    
        
        preprint_msg = args.preprint_msg[0].strip() + ' : '
        main(preprint_msg = preprint_msg)
        
    tstop = time.perf_counter()
    
    print('Total time in seconds : ',tstop-tstart)