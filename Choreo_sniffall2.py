
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
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import cnames
from matplotlib import animation
import copy
import shutil
import time

from Choreo_funs import *



def main(preprint_msg=''):
    
    def print(*args, **kwargs):
        """My custom print() function."""
        # Adding new arguments to the print function signature 
        # is probably a bad idea.
        # Instead consider testing if custom argument keywords
        # are present in kwargs
        __builtins__.print(preprint_msg,end='')
        return __builtins__.print(*args, **kwargs)
    

    # nbody =     3

    # Sym_list = []
    # the_lcm = 3


    # SymType = {
        # 'name'  : 'C',
        # 'n'     : nbody,
        # 'k'     : 1,
        # 'l'     : 0 ,
        # 'p'     : 0 ,
        # 'q'     : 1 ,
    # }
    # istart = 0
    # Sym_list.extend(Make2DChoreoSym(SymType,[i+istart for i in range(nbody)]))



    # SymType = {
        # 'name'  : 'C',
        # 'n'     : -15,
        # 'k'     : 1,
        # 'l'     : 1 ,
        # 'p'     : 0 ,
        # 'q'     : 1 ,
    # }

    # istart = 5
    # Sym_list.extend(Make2DChoreoSym(SymType,[i+istart for i in range(3)]))
    # Sym_list.append(ChoreoSym(
                    # LoopTarget=istart,
                    # LoopSource=istart,
                    # SpaceRot = np.identity(ndim,dtype=np.float64),
                    # TimeRev=1,
                    # TimeShift=fractions.Fraction(numerator=1,denominator=5)
                    # ))

    # nbpl = [3,2,5]
    # nbpl = [1,2,3,4,5]
    nbpl = [1,1]
    # nbpl = [3,2]
    # nbpl = [2,3]
    # nbpl = [i+1 for i in range(4)]
    # nbpl = [1 for i in range(10)]
    # nbpl = [5]
    the_lcm = m.lcm(*nbpl)

    # SymName = ['D' ]
    SymName = None
    Sym_list,nbody = Make2DChoreoSymManyLoops(nbpl=nbpl,SymName=SymName)

    # nbpl = [1,1,1,1,1]
    # nbpl = [4,3,2]
    # nbpl = [1,2]

    # the_lcm = m.lcm(*nbpl)
    # SymName = None
    # Sym_list,nbody = Make2DChoreoSymManyLoops(nbpl=nbpl,SymName=SymName)



    # rot_angle =  twopi * 1 /  2
    # s = 1

    # Sym_list.append(ChoreoSym(
        # LoopTarget=2,
        # LoopSource=3,
        # SpaceRot = np.array([[s*np.cos(rot_angle),-s*np.sin(rot_angle)],[np.sin(rot_angle),np.cos(rot_angle)]],dtype=np.float64),
        # TimeRev=1,
        # TimeShift=fractions.Fraction(numerator=0,denominator=2)
        # ))

    # Sym_list.append(ChoreoSym(
        # LoopTarget=2,
        # LoopSource=3,
        # SpaceRot = np.array([[s*np.cos(rot_angle),-s*np.sin(rot_angle)],[np.sin(rot_angle),np.cos(rot_angle)]],dtype=np.float64),
        # TimeRev=1,
        # TimeShift=fractions.Fraction(numerator=-1,denominator=2)
        # ))



    # rot_angle = 0
    # s = -1

    # Sym_list.append(ChoreoSym(
        # LoopTarget=4,
        # LoopSource=4,
        # SpaceRot = np.array([[s*np.cos(rot_angle),-s*np.sin(rot_angle)],[np.sin(rot_angle),np.cos(rot_angle)]],dtype=np.float64),
        # TimeRev=-1,
        # TimeShift=fractions.Fraction(numerator=0,denominator=1)
        # ))







    mass = np.array([2,1],dtype=np.float64)
    # mass = np.ones((nbody))

    # mass[0]=2

    # mass = np.array([1.,1.5])

    # mass[0:3]  = 2*5
    # mass[3:5]  = 3*5
    # mass[5:10] = 2*3



    MomConsImposed = True
    # MomConsImposed = False

    store_folder = './Sniff_all_sym/'
    store_folder = store_folder+str(nbody)
    if not(os.path.isdir(store_folder)):
        os.makedirs(store_folder)

    Use_exact_Jacobian = True
    # Use_exact_Jacobian = False

    # Use_deflation = True
    Use_deflation = False

    Look_for_duplicates = True
    # Look_for_duplicates = False

    Check_Escape = True
    # Check_Escape = False

    # Penalize_Escape = True
    Penalize_Escape = False

    save_init = False
    # save_init = True

    save_approx = False
    # save_approx = True

    Save_img = True
    # Save_img = False

    # img_size = (12,12) # Image size in inches
    img_size = (8,8) # Image size in inches

    nint_plot_img = 10000

    Save_anim = True
    # Save_anim = False

    vid_size = (8,8) # Image size in inches
    nint_plot_anim = 2*2*2*3*3*5 * 6
    # nperiod_anim = 1./nbody

    color = "body"
    # color = "loop"
    # color = "velocity"
    # color = "all"

    try:
        the_lcm
    except NameError:
        period_div = 1.
    else:
        period_div = the_lcm


    nperiod_anim = 1./period_div

    # Plot_trace_anim = True
    Plot_trace_anim = False

    n_reconverge_it_max = 4
    # n_reconverge_it_max = 1

    # ncoeff_init = 20
    # ncoeff_init = 800
    # ncoeff_init = 201   
    # ncoeff_init = 300   
    ncoeff_init = 600
    # ncoeff_init = 990
    # ncoeff_init = 1200
    # ncoeff_init = 90

    disp_scipy_opt = False
    # disp_scipy_opt = True

    Newt_err_norm_max = 1e-9
    Newt_err_norm_max_save = Newt_err_norm_max * 100

    # Save_Bad_Sols = True
    Save_Bad_Sols = False

    duplicate_eps = 1e-9

    # krylov_method = 'lgmres'
    # krylov_method = 'gmres'
    # krylov_method = 'bicgstab'
    krylov_method = 'cgs'
    # krylov_method = 'minres'

    # line_search = 'armijo'
    line_search = 'wolfe'
    
    gradtol_list =          [1e-3   ,1e-5   ,1e-7   ,1e-9   ,1e-11  ,1e-13]
    inner_maxiter_list =    [30     ,50     ,60     ,70     ,80     ,100]
    maxiter_list =          [1000   ,1000   ,1000   ,500   ,500    ,500]
    outer_k_list =          [5      ,5      ,5      ,5      ,7      ,7]
    store_outer_Av_list =   [False  ,False  ,False  ,False  ,True   ,True]
    
    n_optim_param = len(gradtol_list)
    
    gradtol_max = 100*gradtol_list[n_optim_param-1]
    foundsol_tol = 1000*gradtol_list[0]

    escape_fac = 1e0
    # escape_fac = 1e-1
    # escape_fac = 1e-2
    # escape_fac = 1e-3
    # escape_fac = 1e-4
    # escape_fac = 1e-5
    # escape_fac = 0
    escape_min_dist = 1
    escape_pow = 2.0
    # escape_pow = 2.5
    # escape_pow = 1.5
    # escape_pow = 0.5

    n_grad_change = 1.
    # n_grad_change = 1.5

    print('Searching periodic solutions of {:d} bodies'.format(nbody))
    # print('Processing symmetries for {:d} convergence levels ...'.format(n_reconverge_it_max+1))

    print('Processing symmetries for {0:d} convergence levels'.format(n_reconverge_it_max+1))
    callfun = setup_changevar(nbody,ncoeff_init,mass,n_reconverge_it_max,Sym_list=Sym_list,MomCons=MomConsImposed,n_grad_change=n_grad_change)

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

    # for i in range(n_reconverge_it_max+1):
    for i in [0]:
        
        args = callfun[0]
        print('Convergence attempt number : ',i+1)
        print('    Number of scalar parameters before constraints : ',args['coeff_to_param_list'][i].shape[1])
        print('    Number of scalar parameters after  constraints : ',args['coeff_to_param_list'][i].shape[0])
        print('    Reduction of ',100*(1-args['coeff_to_param_list'][i].shape[0]/args['coeff_to_param_list'][i].shape[1]),' %')
        print('')
        

    x0 = np.random.random(callfun[0]['param_to_coeff_list'][0].shape[1])
    xmin = Compute_MinDist(x0,callfun)
    if (xmin < 1e-5):
        print(xmin)
        raise ValueError("Init inter body distance too low. There is something wrong with constraints")

    # filehandler = open(store_folder+'/callfun_list.pkl',"wb")
    # pickle.dump(callfun_list,filehandler)

    if (Penalize_Escape):

        Action_grad_mod = Compute_action_onlygrad_escape

    else:
        
        Action_grad_mod = Compute_action_onlygrad

    callfun[0]["current_cvg_lvl"] = 0
    ncoeff = callfun[0]["ncoeff_list"][callfun[0]["current_cvg_lvl"]]
    nint = callfun[0]["nint_list"][callfun[0]["current_cvg_lvl"]]

    coeff_ampl_o=1e-1
    k_infl=1
    k_max=600
    coeff_ampl_min=1e-16

    all_coeffs_min,all_coeffs_max = Make_Init_bounds_coeffs(nloop,ncoeff,coeff_ampl_o,k_infl,k_max,coeff_ampl_min)

    x_min = Package_all_coeffs(all_coeffs_min,callfun)
    x_max = Package_all_coeffs(all_coeffs_max,callfun)

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
    # n_opt_max = 1
    # n_opt_max = 5
    n_opt_max = 1e10
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
        
        xrand = sampler.random()
        
        rand_dim = 0
        for i in range(callfun[0]['coeff_to_param_list'][callfun[0]["current_cvg_lvl"]].shape[0]):
            if ((x_max[i] - x_min[i]) > rand_eps):
                x0[i] = x_min[i] + (x_max[i] - x_min[i])*xrand[rand_dim]
                rand_dim +=1
            else:
                x0[i] = 0.

        if save_init:
            
            print('Saving init state')

            if Save_img :
                plot_all_2D(x0,nint_plot_img,callfun,'init.png',fig_size=img_size,color=color)        
                
            # if Save_anim :
                # plot_all_2D_anim(x0,nint_plot_anim,callfun,'init.mp4',nperiod_anim,Plot_trace=Plot_trace_anim,fig_size=vid_size)
                

                all_coeffs = Unpackage_all_coeffs(x0,callfun)
                np.save('init.npy',all_coeffs)
                
            # print(1/0)
            
        f0 = Action_grad_mod(x0,callfun)
        best_sol = current_best(x0,f0)

        GoOn = True
        
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

            inner_M = None
            
            print('Action Grad Norm on entry : ',best_sol.f_norm)
            print('Optim level : ',i_optim_param+1,' / ',n_optim_param , 'Resize level : ',callfun[0]["current_cvg_lvl"]+1,' / ',n_reconverge_it_max+1)

            
            F = lambda x : Action_grad_mod(x,callfun)
            # jac_options = {'method':krylov_method,'rdiff':rdiff,'outer_k':outer_k,'inner_inner_m':inner_maxiter,'inner_store_outer_Av':store_outer_Av,'inner_tol':inner_tol }
            
            if (krylov_method == 'lgmres'):
                jac_options = {'method':krylov_method,'rdiff':rdiff,'outer_k':outer_k,'inner_inner_m':inner_maxiter,'inner_store_outer_Av':store_outer_Av,'inner_tol':inner_tol,'inner_M':inner_M }
            elif (krylov_method == 'gmres'):
                jac_options = {'method':krylov_method,'rdiff':rdiff,'outer_k':outer_k,'inner_tol':inner_tol,'inner_M':inner_M }
            else:
                jac_options = {'method':krylov_method,'rdiff':rdiff,'outer_k':outer_k,'inner_tol':inner_tol,'inner_M':inner_M }


            if (Use_exact_Jacobian):

                FGrad = lambda x,dx : Compute_action_hess_mul(x,dx,callfun)
                jacobian = ExactKrylovJacobian(exactgrad=FGrad,**jac_options)

            else: 
                jacobian = opt.nonlin.KrylovJacobian(**jac_options)

            try : 
                
                x0 = np.copy(best_sol.x)
                opt_result = opt.nonlin.nonlin_solve(F=F,x0=x0,jacobian=jacobian,verbose=disp_scipy_opt,maxiter=maxiter,f_tol=gradtol,line_search=line_search,callback=best_sol.update,raise_exception=False)
                
            except Exception as exc:
                
                print(exc)
                print("Value Error occured, skipping.")
                GoOn = False
                # raise(exc)
                
            SaveSol = False
            
            if (GoOn and Check_Escape):
                
                Escaped,_ = Detect_Escape(best_sol.x,callfun)

                if Escaped:
                    print('One loop escaped. Starting over')    
                    
                GoOn = GoOn and not(Escaped)
                
            if (GoOn and Look_for_duplicates):

                Action,GradAction = Compute_action(best_sol.x,callfun)
                
                Found_duplicate,file_path = Check_Duplicates(best_sol.x,callfun,hash_dict,store_folder,duplicate_eps)
                
                if (Found_duplicate):
                
                    print('Found Duplicate !')   
                    print('Path : ',file_path)
                    
                GoOn = GoOn and not(Found_duplicate)
                
            if (GoOn):
                
                ParamFoundSol = (best_sol.f_norm < foundsol_tol)
                ParamPreciseEnough = (best_sol.f_norm < gradtol_max)
                print('Opt Action Grad Norm : ',best_sol.f_norm)
            
                Newt_err = Compute_Newton_err(best_sol.x,callfun)
                Newt_err_norm = np.linalg.norm(Newt_err)/(nint*nbody)
                NewtonPreciseGood = (Newt_err_norm < Newt_err_norm_max)
                NewtonPreciseEnough = (Newt_err_norm < Newt_err_norm_max_save)
                print('Newton Error : ',Newt_err_norm)
                
                CanChangeOptimParams = i_optim_param < (n_optim_param-1)
                
                CanRefine = (callfun[0]["current_cvg_lvl"] < n_reconverge_it_max)
                
                if CanRefine :
                    
                    all_coeffs_coarse = Unpackage_all_coeffs(best_sol.x,callfun)
                    ncoeff_coarse = callfun[0]["ncoeff_list"][callfun[0]["current_cvg_lvl"]]
                    
                    callfun[0]["current_cvg_lvl"] += 1
                    ncoeff_fine = callfun[0]["ncoeff_list"][callfun[0]["current_cvg_lvl"]]

                    all_coeffs_fine = np.zeros((nloop,ndim,ncoeff_fine,2),dtype=np.float64)
                    # all_coeffs_fine[:,:,0:ncoeff_coarse,:] = np.copy(all_coeffs_coarse)
                    for k in range(ncoeff_coarse):
                        all_coeffs_fine[:,:,k,:] = all_coeffs_coarse[:,:,k,:]
                        
                    x_fine = Package_all_coeffs(all_coeffs_fine,callfun)
                    f_fine = Action_grad_mod(x_fine,callfun)
                    f_fine_norm = np.linalg.norm(f_fine)
                    
                    NeedsRefinement = (f_fine_norm > 3*best_sol.f_norm)
                    
                    callfun[0]["current_cvg_lvl"] += -1
                
                else:
                    
                    NeedsRefinement = False
                    
                NeedsChangeOptimParams = GoOn and CanChangeOptimParams and not(ParamPreciseEnough) and not(NeedsRefinement)
                
                if GoOn and not(ParamFoundSol):
                
                    GoOn = False
                    print('Optimizer could not zero in on a solution')

                if GoOn and not(ParamPreciseEnough) and not(NewtonPreciseEnough) and not(CanChangeOptimParams):
                
                    GoOn = False
                    print('Newton Error too high, discarding solution')
                
                if GoOn and ParamPreciseEnough and not(NewtonPreciseGood) and not(NeedsRefinement):

                    GoOn=False
                    print("Stopping Search : there might be something wrong with the constraints")
                
                if GoOn and NewtonPreciseGood :

                    GoOn = False
                    print("Stopping Search : Found solution")
                    SaveSol = True
                    
                if GoOn and NewtonPreciseEnough and not(CanChangeOptimParams) :

                    GoOn = False
                    print("Stopping Search : Found approximate solution")
                    SaveSol = True

                if SaveSol :
                    
                    GoOn  = False
                    
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

                    print('Saving solution as '+filename_output+'.*')
             
                    Write_Descriptor(best_sol.x,callfun,filename_output+'.txt')
                    
                    if Save_img :
                        plot_all_2D(best_sol.x,nint_plot_img,callfun,filename_output+'.png',fig_size=img_size,color=color)
                        
                    if Save_anim :
                        plot_all_2D_anim(best_sol.x,nint_plot_anim,callfun,filename_output+'.mp4',nperiod_anim,Plot_trace=Plot_trace_anim,fig_size=vid_size)
                    
                    all_coeffs = Unpackage_all_coeffs(best_sol.x,callfun)
                    np.save(filename_output+'.npy',all_coeffs)

                
                if GoOn and NeedsRefinement:
                    
                    best_sol = current_best(x_fine,f_fine)
                    callfun[0]["current_cvg_lvl"] += 1
                    
                    ncoeff = callfun[0]["ncoeff_list"][callfun[0]["current_cvg_lvl"]]
                    nint = callfun[0]["nint_list"][callfun[0]["current_cvg_lvl"]]
                    
                if GoOn and NeedsChangeOptimParams:
                    
                    i_optim_param += 1
                    
                
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
