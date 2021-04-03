import numpy as np
import math as m
import scipy.optimize as opt
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import cnames
from matplotlib import animation
import copy
import os, shutil
import time

from Choreo_funs import *

nbody =     3
mass = np.ones((nbody))

Sym_list = []

# ~ SymType = {
    # ~ 'name'  : 'C',
    # ~ 'n'     : 2,
    # ~ 'k'     : 1,
    # ~ 'l'     : 1 ,
    # ~ 'p'     : 0 ,
    # ~ 'q'     : 2 ,
# ~ }

# ~ Sym_list.extend(Make2DChoreoSym(SymType,[0,1]))

# ~ SymType = {
    # ~ 'name'  : 'C',
    # ~ 'n'     : 2,
    # ~ 'k'     : 1,
    # ~ 'l'     : 1 ,
    # ~ 'p'     : 0 ,
    # ~ 'q'     : 2 ,
# ~ }

# ~ Sym_list.extend(Make2DChoreoSym(SymType,[2,3]))

# ~ SymType = {
    # ~ 'name'  : 'C',
    # ~ 'n'     : 2,
    # ~ 'k'     : 1,
    # ~ 'l'     : 1 ,
    # ~ 'p'     : 0 ,
    # ~ 'q'     : 2 ,
# ~ }

# ~ Sym_list.extend(Make2DChoreoSym(SymType,[4,5]))




Search_Min_Only = False
# ~ Search_Min_Only = True

MomConsImposed = True
# ~ MomConsImposed = False

store_folder = './Sniff_all_sym/'
store_folder = store_folder+str(nbody)
if not(os.path.isdir(store_folder)):
    os.mkdir(store_folder)


# ~ Use_deflation = True
Use_deflation = False

Look_for_duplicates = True
# ~ Look_for_duplicates = False

Check_loop_dist = True
# ~ Check_loop_dist = False

save_init = False
# ~ save_init = True

save_approx = False
# ~ save_approx = True

Save_img = True
# ~ Save_img = False

img_size = (12,12) # Image size in inches

Save_anim = True
# ~ Save_anim = False

vid_size = (5,5) # Image size in inches

Plot_trace_anim = True
# ~ Plot_trace_anim = False

n_reconverge_it_max = 7
# ~ n_reconverge_it_max = 1

# ~ ncoeff_init = 100
# ~ ncoeff_init = 800
# ~ ncoeff_init = 350   
ncoeff_init = 600
# ~ ncoeff_init = 990
# ~ ncoeff_init = 1200
# ~ ncoeff_init = 90

disp_scipy_opt = False
# ~ disp_scipy_opt = True

Newt_err_norm_max = 1e-9
Newt_err_norm_max_save = Newt_err_norm_max * 100

# ~ Save_Bad_Sols = True
Save_Bad_Sols = False

duplicate_eps = 1e-9

# ~ krylov_method = 'lgmres'
# ~ krylov_method = 'gmres'
# ~ krylov_method = 'bicgstab'
krylov_method = 'cgs'
# ~ krylov_method = 'minres'

# ~ line_search = 'armijo'
line_search = 'wolfe'

print('Searching periodic solutions of {:d} bodies'.format(nbody))
# ~ print('Processing symmetries for {:d} convergence levels ...'.format(n_reconverge_it_max+1))


print('Processing symmetries for {0:d} convergence levels'.format(n_reconverge_it_max+1))
callfun = setup_changevar(nbody,ncoeff_init,mass,n_reconverge_it_max,Sym_list=Sym_list,MomCons=MomConsImposed)


print('')

args = callfun[0]
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

if (Use_deflation):
    print('Loading previously saved sols as deflation vectors')
    
    Init_deflation(callfun)
    Load_all_defl(store_folder,callfun)
    
    Action_grad_mod = Compute_action_defl
    
else:
    
    Action_grad_mod = Compute_action_onlygrad


n_opt = 0
# ~ n_opt_max = 1
n_opt_max = 1e10
while (n_opt < n_opt_max):

    n_opt += 1
    
    print('Optimization attempt number : ',n_opt)

    callfun[0]["current_cvg_lvl"] = 0
    ncoeff = callfun[0]["ncoeff_list"][callfun[0]["current_cvg_lvl"]]
    nint = callfun[0]["nint_list"][callfun[0]["current_cvg_lvl"]]
    
    all_coeffs = np.zeros((nloop,ndim,ncoeff,2),dtype=np.float64)

    # ~ amplitude_o = 0.3
    # ~ amplitude_o = 0.1
    amplitude_o = 0.001
    # ~ amplitude_o = 0.08

    decrease_pow = 2.
    decrease_fac = 1 - 0.3

    for il in range(nloop):
        # ~ mass[il] = np.random.rand()+2.
        for idim in range(ndim):
            # ~ kfac = 1.
            kfac = 1.
            for k in range(1,ncoeff):
                
                randphase = np.random.rand() * twopi * 3.
                randampl = np.random.rand()* amplitude_o
            
                ko = 1
                k1 =20
                k2= 40
                if (k <= ko):
                    # ~ randampl = 0.12
                    randampl = 0.00 * np.random.rand()
                    
                
                elif (k <= k1):
                    randampl = 0.03*np.random.rand()
                
                elif (k <= k2):
                    randampl = 0.005*np.random.rand()
                
            
                k_thresh_damp = k2
                # ~ k_thresh_damp = 1
                
                if (k >= k_thresh_damp):
                    kfac = kfac* decrease_fac
                    randampl = randampl*kfac
                
      
                all_coeffs[il,idim,k,0] = randampl*np.cos(randphase)
                all_coeffs[il,idim,k,1] = randampl*np.sin(randphase)
    
    x0 = Package_all_coeffs(all_coeffs,callfun)
    
    

    if Search_Min_Only:
                    
        gradtol = 1e-5
        maxiter = 1000
        
        opt_result = opt.minimize(fun=Compute_action,x0=x0,args=callfun,method='trust-krylov',jac=True,hessp=Compute_action_hess_mul,options={'disp':disp_scipy_opt,'maxiter':maxiter,'gtol' : gradtol,'inexact': True})
        
        x_opt = opt_result['x']
        
        Go_On = True
        
    else:
            
        f0 = Action_grad_mod(x0,callfun)
        best_sol = current_best(x0,f0)

        gradtol = 1e-5
        maxiter = 500

        try : 
            
            opt_result = opt.root(fun=Action_grad_mod,x0=x0,args=callfun,method='krylov', options={'line_search':line_search,'disp':disp_scipy_opt,'maxiter':maxiter,'fatol':gradtol,'jac_options':{'method':krylov_method}},callback=best_sol.update)
            
            print("After Krylov : ",best_sol.f_norm)
            
            Go_On = True
            
            x_opt = best_sol.x
            
        except Exception as exc:
            
            print(exc)
            print("Value Error occured, skipping.")
            Go_On = False

    if (Check_loop_dist and Go_On):
        
        Go_On = not(Detect_Escape(x_opt,callfun))

        if not(Go_On):
            print('One loop escaped. Starting over')    
    
    if (Go_On):

        f0 = Action_grad_mod(x_opt,callfun)
        best_sol = current_best(x_opt,f0)

        maxiter = 10
        gradtol = 1e-11
        opt_result = opt.root(fun=Action_grad_mod,x0=x_opt,args=callfun,method='krylov', options={'disp':disp_scipy_opt,'maxiter':maxiter,'fatol':gradtol,'jac_options':{'method':krylov_method}},callback=best_sol.update)

        print('Approximate solution found ! Action Grad Norm : ',best_sol.f_norm)

        Found_duplicate = False

        if (Look_for_duplicates):
            
            print('Checking Duplicates.')

            Action,GradAction = Compute_action(best_sol.x,callfun)
            
            Found_duplicate,file_path = Check_Duplicates(best_sol.x,callfun,store_folder,duplicate_eps)
            
        else:
            
            Found_duplicate = False
            
        if (Found_duplicate):
        
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
                
                print('After Resize : Action Grad Norm : ',best_sol.f_norm)
                                
                if Search_Min_Only:
                         
                    gradtol = 1e-7
                    maxiter = 1000
                    opt_result = opt.minimize(fun=Compute_action,x0=x0,args=callfun,method='trust-krylov',jac=True,hessp=Compute_action_hess_mul,options={'disp':disp_scipy_opt,'maxiter':maxiter,'gtol' : gradtol,'inexact': True})

                    x0 = opt_result['x']

                    maxiter = 20
                    gradtol = 1e-15
                    opt_result = opt.root(fun=Action_grad_mod,x0=x0,args=callfun,method='krylov', options={'disp':disp_scipy_opt,'maxiter':maxiter,'fatol':gradtol,'jac_options':{'method':krylov_method}},callback=best_sol.update)
                    
                    all_coeffs = Unpackage_all_coeffs(best_sol.x,callfun)
                    
                    print('Opt Action Grad Norm : ',best_sol.f_norm)
                
                    Newt_err = Compute_Newton_err(best_sol.x,callfun)
                    Newt_err_norm = np.linalg.norm(Newt_err)/nint
                    
                    print('Newton Error : ',Newt_err_norm)
                
                    SaveSol = (Newt_err_norm < Newt_err_norm_max_save)
                                    
                    if (Check_loop_dist):
                        
                        Go_On = not(Detect_Escape(best_sol.x,callfun))

                        if not(Go_On):
                            print('One loop escaped. Starting over')   
                else:

                    maxiter = 500
                    gradtol = 1e-13
                    
                    try : 
                                    
                        opt_result = opt.root(fun=Action_grad_mod,x0=x0,args=callfun,method='krylov', options={'line_search':line_search,'disp':disp_scipy_opt,'maxiter':maxiter,'fatol':gradtol,'jac_options':{'method':krylov_method}},callback=best_sol.update)
                                
                        all_coeffs = Unpackage_all_coeffs(best_sol.x,callfun)
                        
                        print('Opt Action Grad Norm : ',best_sol.f_norm)
                    
                        Newt_err = Compute_Newton_err(best_sol.x,callfun)
                        Newt_err_norm = np.linalg.norm(Newt_err)/nint
                        
                        print('Newton Error : ',Newt_err_norm)
                    
                        SaveSol = (Newt_err_norm < Newt_err_norm_max_save)
                                        
                        if (Check_loop_dist):
                            
                            Go_On = not(Detect_Escape(best_sol.x,callfun))

                            if not(Go_On):
                                print('One loop escaped. Starting over')    

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
            
                    Found_duplicate,file_path = Check_Duplicates(best_sol.x,callfun,store_folder,duplicate_eps)
                    
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
                        filename_output = filename_output + '_bad'
                    
                    print('Saving solution as '+filename_output+'.*')
                    
                    if Save_img :
                        nint_plot = 10000
                        plot_all_2D(best_sol.x,nint_plot,callfun,filename_output+'.png',fig_size=img_size)
                        
                    if Save_anim :
                        nint_plot = 1000
                        nperiod = 1
                        plot_all_2D_anim(best_sol.x,nint_plot,callfun,filename_output+'.mp4',nperiod,Plot_trace=Plot_trace_anim,fig_size=vid_size)
                        
                    Write_Descriptor(best_sol.x,callfun,filename_output+'.txt')
                    
                    all_coeffs = Unpackage_all_coeffs(best_sol.x,callfun)
                    np.save(filename_output+'.npy',all_coeffs)
                    
                    # ~ pickle.dump(best_sol.x,filename_output+'_params.txt'
                    # ~ pickle.dump(best_sol.x,filename_output+'_params.txt'
                    
                    
                    # ~ HessMat = Compute_action_hess_LinOpt(best_sol.x,callfun)
                    # ~ w ,v = sp.linalg.eigsh(HessMat,k=10,which='SA')
                    
                    # ~ print(w)
                    
            
                    # ~ print('Press Enter to continue ...')
                    # ~ pause = input()
                    
                    
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

