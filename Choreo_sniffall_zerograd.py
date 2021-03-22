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


nbody = np.array([1,1,1])
# ~ nbody = np.array([2,2])
# ~ nbody = np.array([3])
# ~ nbody = np.array([4])
# ~ nbody = np.array([5])
# ~ nbody = np.array([6])
# ~ nbody = np.array([7])
# ~ nbody = np.array([9])

nloop = nbody.size
mass = np.ones((nloop))
# ~ mass = np.array([100,10,1],dtype=np.float64)


store_folder = './Sniff_all/'
store_folder = store_folder+str(nbody[0])
for i in range(len(nbody)-1):
    store_folder = store_folder+'x'+str(nbody[i+1])


Look_for_duplicates = True
# ~ Look_for_duplicates = False

Check_loop_dist = True
# ~ Check_loop_dist = False

save_init = False

# ~ save_init = True

# ~ save_approx = False
save_approx = True

# ~ Reconverge_sols = False
Reconverge_sols = True


n_reconverge_it_max = 4

# ~ theta_rot_dupl = [0.,.2,.4,.6,.8]
theta_rot_dupl = np.linspace(start=0.,stop=twopi,endpoint=False,num=nbody[0])
dt_shift_dupl = np.linspace(start=0.,stop=1.,endpoint=False,num=nbody[0]*5)

# ~ print(1/0)

# ~ ncoeff_init = 100
ncoeff_init = 600
# ~ ncoeff_init = 700
# ~ ncoeff_init = 900
# ~ ncoeff_init = 1200
# ~ ncoeff_init = 90

# ~ ncoeff_cutoff = ncoeff_init
ncoeff_cutoff = 100

change_var_coeff =1.
# ~ change_var_coeff =0.5

MomCons_As_Sym = True
# ~ MomCons_As_Sym = False

SymGens = []

for i in range(nloop-1):
    LoopSelect = np.array([i],dtype=int)
    LoopPerm = np.array([i+1],dtype=int)

    rot_angle = twopi * 1/3
    SpaceSym = False 
    if (SpaceSym):
        s = -1
    else:
        s = 1

    TimeRev = False
    # ~ TimeRev = True

    TimeShift = 0

    SymGens.extend([{
    'LoopSelect' : LoopSelect,
    'LoopPerm' : LoopPerm,
    'SpaceRot' : np.array([[s*np.cos(rot_angle),-s*np.sin(rot_angle)],[np.sin(rot_angle),np.cos(rot_angle)]],dtype=np.float64),
    'TimeRev' : TimeRev,
    'TimeShift' : TimeShift,
    }])



il = 0
SymType = {
    'name'  : 'D',
    'n'     : nbody[il],
    'k'     : 1,
    'l'     : 1,
}
SymGens.extend(Make2DSymOneLoop(SymType,il))




n_opt = 0

disp_scipy_opt = False
# ~ disp_scipy_opt = True

Newt_err_norm_max = 1e-9
Newt_err_norm_max_save = Newt_err_norm_max * 1000

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

coeff_to_param_list = []
param_to_coeff_list = []

for i in range(n_reconverge_it_max+1):
    
    ncoeff = ncoeff_init * (2**i)
    
    coeff_to_param , param_to_coeff = setup_changevar(nloop,nbody,ncoeff,mass,MomCons=MomCons_As_Sym,n_grad_change=change_var_coeff,Sym_list=SymGens)
    
    print('Number of scalar parameters before symmetries : ',coeff_to_param.shape[1])
    print('Number of scalar parameters after  symmetries : ',coeff_to_param.shape[0])
    print('Reduction of ',100*(1-coeff_to_param.shape[0]/coeff_to_param.shape[1]),' %')
    print('')
    
    coeff_to_param_list.append(coeff_to_param)
    param_to_coeff_list.append(param_to_coeff)

while (True):
    
    n_opt += 1
    
    print('Optimization attempt number : ',n_opt)

    ncoeff = ncoeff_init
    nint = 2*ncoeff
    
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
            
                ko = 10
                k1 = 20
                k2= 30
                if (k <= ko):
                    # ~ randampl = 0.12
                    randampl = 0.00 * np.random.rand()
                    
                
                elif (k <= k1):
                    randampl = 0.05*np.random.rand()
                
                elif (k <= k2):
                    randampl = 0.00*np.random.rand()
                
            
                k_thresh_damp = k2
                # ~ k_thresh_damp = 1
                
                if (k >= k_thresh_damp):
                    kfac = kfac* decrease_fac
                    randampl = randampl*kfac
                
                # ~ if (k >= k_thresh_damp):
                    # ~ kfac = 0.
                
                # ~ if (k >= k_thresh_damp):
                    # ~ kfac = 1./ ((k-k_thresh_damp+1)**decrease_pow)
                    
                # ~ if (k % nbody[il]  == 0):
                    # ~ randampl = 0.
                    
                
                
                # ~ if ((il==0) and (k % nbody[1] != 0)):
                    # ~ print(k)
                    # ~ randampl = 0
                    
                all_coeffs[il,idim,k,0] = randampl*np.cos(randphase)
                all_coeffs[il,idim,k,1] = randampl*np.sin(randphase)
                
    if (save_init):
        nint_plot = 200
        plot_all_2D(nloop,nbody,nint_plot,all_coeffs,'init.png')
        np.save('init.npy',all_coeffs)

    x0,callfun = Package_args(nloop,nbody,ncoeff,mass,nint,all_coeffs,coeff_to_param_list[0],param_to_coeff_list[0])
    f0 = Compute_action_onlygrad_package(x0,callfun)
    best_sol = current_best(x0,f0)
    
    # ~ Action,GradAction = Compute_action_package(x0,callfun)    
    # ~ print(Action)
    # ~ print(1/0)
    
    gradtol = 1e-5
    maxiter = 5000

    # ~ gradtol = 1e-7
    # ~ maxiter = 1000
    
    try : 
        
        opt_result = opt.root(fun=Compute_action_onlygrad_package,x0=x0,args=callfun,method='krylov', options={'line_search':line_search,'disp':disp_scipy_opt,'maxiter':maxiter,'fatol':gradtol,'jac_options':{'method':krylov_method}},callback=best_sol.update)
        
        print("After Krylov : ",best_sol.f_norm)
        
        Go_On = True
        
    except ValueError:
        
        print("Value Error occured, skipping.")
        Go_On = False
    
    
    if (Go_On and not(opt_result['success'])):
        
                
        gradtol = 1e-5
        maxiter = 5000
        x0 = best_sol.x
        opt_result = opt.root(fun=Compute_action_onlygrad_package,x0=x0,args=callfun,method='df-sane', options={'disp':disp_scipy_opt,'maxfev':maxiter,'fatol':gradtol},callback=best_sol.update)

        print("After DF-SANE : ",best_sol.f_norm)
        
        Go_On = opt_result['success']
    
    elif (Go_On):
        
        opt_grad = opt_result['fun']
        opt_grad_norm = np.linalg.norm(opt_grad)

    
    if ((Go_On) and (Check_loop_dist)):
        
        x_opt = opt_result['x']
        all_coeffs = Unpackage_all_coeffs(x_opt,callfun)
        
        max_loop_size = 0.
        for il in range(nloop):
            loop_size = np.linalg.norm(all_coeffs[il,:,1:ncoeff,:])
            max_loop_size = max(loop_size,max_loop_size)
        
        max_loop_dist = 0.
        for il in range(nloop):
            for ilp in range(nloop):
                loop_dist = np.linalg.norm(all_coeffs[il,:,0,0] - all_coeffs[ilp,:,0,0] )
                max_loop_dist = max(loop_dist,max_loop_dist)
        
        Go_On = (max_loop_dist < (4.5 * nloop * max_loop_size))
        
        if not(Go_On):
            print('One loop escaped. Starting over')    
    
    if (Go_On):

        print('Approximate solution found ! Action Grad Norm : ',opt_grad_norm)

        x_opt = opt_result['x']
        all_coeffs = Unpackage_all_coeffs(x_opt,callfun)

        if (save_approx):
            nint_plot = 200
            plot_all_2D(nloop,nbody,nint_plot,all_coeffs,'approx.png')

        Found_duplicate = False

        if (Look_for_duplicates):
            
            print('Checking Duplicates.')

            Action,GradAction = Compute_action_package(best_sol.x,callfun)

            Found_duplicate,dist_sols,file_path = Check_Duplicates(store_folder,all_coeffs,nbody,duplicate_eps,Action_val=Action,ncoeff_cutoff=ncoeff_cutoff,theta_rot_dupl=theta_rot_dupl,dt_shift_dupl=dt_shift_dupl,TimeReversal=True,SpaceSym=True)
            
        else:
            Found_duplicate = False
            
        if (Found_duplicate):
        
            print('Found Duplicate !')  
            print('Distance :',dist_sols)  
            print('Path : ',file_path)
            
        else:

            if (Reconverge_sols):
                
                print('Reconverging solution')
                
                Newt_err_norm = 1.
                
                n_reconverge_it = 0

                while ((Newt_err_norm > Newt_err_norm_max) and (n_reconverge_it < n_reconverge_it_max) and Go_On):
                            
                    # ~ nint_plot = 200
                    # ~ imgfilename = store_folder+'/'+str(n_opt)+'_'+str(n_reconverge_it)
                    # ~ plot_all_2D(nloop,nbody,nint_plot,all_coeffs,imgfilename+'.png')
                    
                    
                    n_reconverge_it = n_reconverge_it + 1
                    
                    all_coeffs_old = np.copy(all_coeffs)
                    
                    ncoeff_new = ncoeff * 2

                    all_coeffs = np.zeros((nloop,ndim,ncoeff_new,2),dtype=np.float64)
                    for k in range(ncoeff):
                        all_coeffs[:,:,k,:] = all_coeffs_old[:,:,k,:]
                        
                    ncoeff = ncoeff_new
                    nint = 2*ncoeff
                    
                    x0,callfun = Package_args(nloop,nbody,ncoeff,mass,nint,all_coeffs,coeff_to_param_list[n_reconverge_it],param_to_coeff_list[n_reconverge_it])
                    
                    f0 = Compute_action_onlygrad_package(x0,callfun)
                    best_sol = current_best(x0,f0)
                    
                    print('After Resize : Action Grad Norm : ',best_sol.f_norm)
                    
                    gradtol = 1e-11
                    maxiter = 1000
                    # ~ opt_result = opt.root(fun=Compute_action_onlygrad_package,x0=x0,args=callfun,method='df-sane', options={'disp':disp_scipy_opt,'maxfev':maxiter,'fatol':gradtol})

                    # ~ x0 = opt_result['x']

                    maxiter = 100
                    gradtol = 1e-15
                    
                    try : 
                                    
                        opt_result = opt.root(fun=Compute_action_onlygrad_package,x0=x0,args=callfun,method='krylov', options={'line_search':line_search,'disp':disp_scipy_opt,'maxiter':maxiter,'fatol':gradtol,'jac_options':{'method':krylov_method}},callback=best_sol.update)

                        Go_On = True
                        
                    except ValueError:
                        
                        print("Value Error occured, skipping.")
                        Go_On = False
                        SaveSol = False


                    if ((Go_On) and (Check_loop_dist)):
                        
                        x_opt = opt_result['x']
                        all_coeffs = Unpackage_all_coeffs(x_opt,callfun)
                        
                        max_loop_size = 0.
                        for il in range(nloop):
                            loop_size = np.linalg.norm(all_coeffs[il,:,1:ncoeff,:])
                            max_loop_size = max(loop_size,max_loop_size)
                        
                        max_loop_dist = 0.
                        for il in range(nloop):
                            for ilp in range(nloop):
                                loop_dist = np.linalg.norm(all_coeffs[il,:,0,0] - all_coeffs[ilp,:,0,0] )
                                max_loop_dist = max(loop_dist,max_loop_dist)
                        
                        Go_On = (max_loop_dist < (4.5 * nloop * max_loop_size))
                        
                        if not(Go_On):
                            print('One loop escaped. Starting over')    
                            SaveSol = False
                    
                    if (Go_On):

                        all_coeffs = Unpackage_all_coeffs(best_sol.x,callfun)
                        
                        print('Opt Action Grad Norm : ',best_sol.f_norm)
                    
                        Newt_err = Compute_Newton_err(nloop,nbody,ncoeff,mass,nint,all_coeffs)
                        Newt_err_norm = np.linalg.norm(Newt_err)
                        
                        print('Newton Error : ',Newt_err_norm)
                    
                        SaveSol = (Newt_err_norm < Newt_err_norm_max_save)
                
                if (Go_On and not(SaveSol)):
                    print('Newton Error too high, discarding solution')
            
            else:
                
                SaveSol = True
            
            if ((SaveSol) or (Save_Bad_Sols)):
                        
                if (Look_for_duplicates):
                    
                    print('Checking Duplicates.')
                
                    Action,GradAction = Compute_action_package(best_sol.x,callfun)
                    
                    Found_duplicate,dist_sols,file_path = Check_Duplicates(store_folder,all_coeffs,nbody,duplicate_eps,Action_val=Action,ncoeff_cutoff=ncoeff_cutoff,theta_rot_dupl=theta_rot_dupl,dt_shift_dupl=dt_shift_dupl,TimeReversal=True,SpaceSym=True)
                
                else:
                    Found_duplicate = False
                
                
                if (Found_duplicate):
                
                    print('Found Duplicate !')  
                    print('Distance :',dist_sols)  
                    print('Path : ',file_path) 
                
                else:
                    
                    max_num_file = 0
                    
                    for filename in os.listdir(store_folder):
                        file_path = os.path.join(store_folder, filename)
                        file_root, file_ext = os.path.splitext(os.path.basename(file_path))
                        
                        if (file_ext == '.npy' ):
                            try:
                                max_num_file = max(max_num_file,int(file_root))
                            except:
                                pass
                        
                    max_num_file = max_num_file + 1
                    
                    filename_output = store_folder+'/'+str(max_num_file)
                    
                    if not(SaveSol):
                        filename_output = filename_output + '_bad'
                    
                    print('Saving solution as '+filename_output+'.*')
                    
                    nint_plot = 1000
                    nperiod = 2
                    np.save(filename_output+'.npy',all_coeffs)
                    plot_all_2D(nloop,nbody,nint_plot,all_coeffs,filename_output+'.png')
                    plot_all_2D_anim(nloop,nbody,nint_plot,nperiod,all_coeffs,filename_output+'.mp4')
                    Write_Descriptor(nloop,nbody,ncoeff,mass,nint,all_coeffs,filename_output+'.txt')
    
    print('')
    print('')
    print('')

