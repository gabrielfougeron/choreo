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


# ~ slow_base_filename = './data/2_cercle.npy'
# ~ slow_base_filename = './data/3_cercle.npy'
slow_base_filename = './data/3_huit.npy'

# ~ fast_base_filename = './data/2_cercle.npy'
# ~ fast_base_filename = './data/3_cercle.npy'
fast_base_filename = './data/3_huit.npy'
# ~ fast_base_filename = './data/3_heart.npy'

nTf = 23
nbs = 3
nbf = 3

Rotate_fast_with_slow = True
# ~ Rotate_fast_with_slow = False

Randomize_Fast_Init = True
# ~ Randomize_Fast_Init = False


all_coeffs_slow_load = np.load(slow_base_filename)
all_coeffs_fast_load = np.load(fast_base_filename)

if (all_coeffs_slow_load.shape[0] != 1):
    raise ValueError("Several loops in slow base")

if (all_coeffs_fast_load.shape[0] != 1):
    raise ValueError("Several loops in fast base")

nbody =  nbs * nbf

mass = np.ones((nbody))

Sym_list = []

SymType = {
    'name'  : 'C',
    'n'     : nbody,
    'k'     : 1,
    'l'     : 1 ,
    'p'     : 0 ,
    'q'     : 1 ,
}


Sym_list.extend(Make2DChoreoSym(SymType,range(nbody)))


Search_Min_Only = False
# ~ Search_Min_Only = True

MomConsImposed = True
# ~ MomConsImposed = False

store_folder = './Target_res/'
store_folder = store_folder+str(nbody)
if not(os.path.isdir(store_folder)):
    os.mkdir(store_folder)


# ~ Use_deflation = True
Use_deflation = False

Look_for_duplicates = True
# ~ Look_for_duplicates = False

Check_loop_dist = True
# ~ Check_loop_dist = False

# ~ save_first_init = False
save_first_init = True

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
nint_plot_anim = nbody*250
nperiod_anim = 1./nbody
# ~ nperiod_anim = 1.


Plot_trace_anim = True
# ~ Plot_trace_anim = False

n_reconverge_it_max = 5
# ~ n_reconverge_it_max = 1

# ~ ncoeff_init = 100
# ~ ncoeff_init = 800
# ~ ncoeff_init = 350   
# ~ ncoeff_init = 600
ncoeff_init = 900
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

callfun[0]["current_cvg_lvl"] = 0
ncoeff = callfun[0]["ncoeff_list"][callfun[0]["current_cvg_lvl"]]
nint = callfun[0]["nint_list"][callfun[0]["current_cvg_lvl"]]

ncoeff_slow = all_coeffs_slow_load.shape[2]
ncoeff_fast = all_coeffs_fast_load.shape[2]



all_coeffs_slow_mod = np.zeros((nloop,ndim,ncoeff,2),dtype=np.float64)
all_coeffs_fast_mod = np.zeros((nloop,ndim,ncoeff,2),dtype=np.float64)


phys_exp = 1./(-n+1)


rfac_slow = (nbf)**(-phys_exp/2)
rfac_fast = (1. / nTf)**phys_exp



print('nbs = ',nbs)
print('nbf = ',nbf)
print('nTf = ',nTf)

print('rfac_slow = ',rfac_slow)
print('rfac_fast = ',rfac_fast)

k_fac_slow = nbf
# ~ k_fac_fast = nbs*nbf*nTf
k_fac_fast = nTf

for il in range(nloop):
    for idim in range(ndim):
        for k in range(1,min(ncoeff//k_fac_slow,ncoeff_slow)):
            
            all_coeffs_slow_mod[il,idim,k*k_fac_slow,0]  = rfac_slow * all_coeffs_slow_load[il,idim,k,0]
            all_coeffs_slow_mod[il,idim,k*k_fac_slow,1]  = rfac_slow * all_coeffs_slow_load[il,idim,k,1]

if Randomize_Fast_Init :

    theta = 2 * np.pi * np.random.random()
    TimeRevscal = 1. if (np.random.random() > 1./2.) else -1.

else :
        
    theta = 0.
    TimeRevscal = 1.


RanRotMat = np.array( [[np.cos(theta) , np.sin(theta)] , [-np.sin(theta),np.cos(theta)]])


for il in range(nloop):
    for k in range(1,min(ncoeff // (k_fac_fast),ncoeff_fast)):
            
        v = RanRotMat.dot(all_coeffs_fast_load[il,:,k,0])
        w = TimeRevscal * RanRotMat.dot(all_coeffs_fast_load[il,:,k,1])
            
        for idim in range(ndim):
            
            all_coeffs_fast_mod[il,idim,k*k_fac_fast,0]  = rfac_fast * v[idim]
            all_coeffs_fast_mod[il,idim,k*k_fac_fast,1]  = rfac_fast * w[idim]
 

if Rotate_fast_with_slow :

    c_coeffs_slow = all_coeffs_slow_mod.view(dtype=np.complex128)[...,0]
    all_pos_slow = np.fft.irfft(c_coeffs_slow,n=nint,axis=2)

    c_coeffs_fast = all_coeffs_fast_mod.view(dtype=np.complex128)[...,0]
    all_pos_fast = np.fft.irfft(c_coeffs_fast,n=nint,axis=2)


    all_coeffs_slow_mod_speed = np.zeros((nloop,ndim,ncoeff,2),dtype=np.float64)


    for il in range(nloop):
        for idim in range(ndim):
            for k in range(ncoeff):

                all_coeffs_slow_mod_speed[il,idim,k,0] = k * all_coeffs_slow_mod[il,idim,k,1] 
                all_coeffs_slow_mod_speed[il,idim,k,1] = -k * all_coeffs_slow_mod[il,idim,k,0] 
            

    c_coeffs_slow_mod_speed = all_coeffs_slow_mod_speed.view(dtype=np.complex128)[...,0]
    all_pos_slow_mod_speed = np.fft.irfft(c_coeffs_slow_mod_speed,n=nint,axis=2)

    all_pos_avg = np.zeros((nloop,ndim,nint),dtype=np.float64)

    for il in range(nloop):
        for iint in range(nint):
            
            v = all_pos_slow_mod_speed[il,:,iint]
            v = v / np.linalg.norm(v)

            SpRotMat = np.array( [[v[0] , -v[1]] , [v[1],v[0]]])
            # ~ SpRotMat = np.array( [[v[0] , v[1]] , [-v[1],v[0]]])
            
            all_pos_avg[il,:,iint] = all_pos_slow[il,:,iint] + SpRotMat.dot(all_pos_fast[il,:,iint])


    c_coeffs_avg = np.fft.rfft(all_pos_avg,n=nint,axis=2)
    all_coeffs_avg = np.zeros((nloop,ndim,ncoeff,2),dtype=np.float64)

    for il in range(nloop):
        for idim in range(ndim):
            for k in range(min(ncoeff,ncoeff_slow)):
                all_coeffs_avg[il,idim,k,0] = c_coeffs_avg[il,idim,k].real
                all_coeffs_avg[il,idim,k,1] = c_coeffs_avg[il,idim,k].imag
                
                
else :
    
    all_coeffs_avg = all_coeffs_fast_mod + all_coeffs_slow_mod















# ~ for il in range(nloop):
    # ~ for k in range(ncoeff):
        # ~ if (k < 50):
            # ~ print(k,np.linalg.norm(all_coeffs_avg[il,:,k,:]) )












all_coeffs_min = np.zeros((nloop,ndim,ncoeff,2),dtype=np.float64)
all_coeffs_max = np.zeros((nloop,ndim,ncoeff,2),dtype=np.float64)

for il in range(nloop):
    for idim in range(ndim):
        for k in range(1,ncoeff):

            ko = 0
            k1 = 50
            k2= 0

            # ~ ko = 0  
            # ~ k1 = 0
            # ~ k2=  0
            if (k <= ko):
                randampl = 0.00005
            elif (k <= k1):
                # ~ randampl = 1.5
                randampl = 0.00001
            elif (k <= k2):
                randampl = 0.005
            else:
                randampl = 0.

            all_coeffs_min[il,idim,k,0] = -randampl
            all_coeffs_min[il,idim,k,1] = -randampl
            all_coeffs_max[il,idim,k,0] =  randampl
            all_coeffs_max[il,idim,k,1] =  randampl

x_avg = Package_all_coeffs(all_coeffs_avg,callfun)
x_min = Package_all_coeffs(all_coeffs_min,callfun)
x_max = Package_all_coeffs(all_coeffs_max,callfun)

rand_eps = 1e-6
rand_dim = 0
for i in range(callfun[0]['coeff_to_param_list'][0].shape[0]):
    if ((x_max[i] - x_min[i]) > rand_eps):
        rand_dim +=1

print('Number of initialization dimensions : ',rand_dim)


sampler = UniformRandom(d=rand_dim)
# ~ sampler = Halton(d=rand_dim)


hash_dict = {}

n_opt = 0
# ~ n_opt_max = 1
# ~ n_opt_max = 1e10
n_opt_max = 100
while (n_opt < n_opt_max):

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

    if save_init or (save_first_init and n_opt == 1):
    
        plot_all_2D(x0,nint_plot_img,callfun,'init.png',fig_size=img_size)        
        
        if Save_anim :
            plot_all_2D_anim(x0,nint_plot_anim,callfun,'init.mp4',nperiod_anim,Plot_trace=Plot_trace_anim,fig_size=vid_size)      

        all_coeffs = Unpackage_all_coeffs(x0,callfun)
        np.save('init.npy',all_coeffs)
    
    
    if Search_Min_Only:
                    
        gradtol = 1e-1
        maxiter = 1000
        
        opt_result = opt.minimize(fun=Compute_action,x0=x0,args=callfun,method='trust-krylov',jac=True,hessp=Compute_action_hess_mul,options={'disp':disp_scipy_opt,'maxiter':maxiter,'gtol' : gradtol,'inexact': True})
        
        x_opt = opt_result['x']
        
        Go_On = True
        
    else:
            
        f0 = Action_grad_mod(x0,callfun)
        best_sol = current_best(x0,f0)

        gradtol = 1e-1
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

        maxiter = 20
        gradtol = 1e-11
        opt_result = opt.root(fun=Action_grad_mod,x0=x_opt,args=callfun,method='krylov', options={'disp':disp_scipy_opt,'maxiter':maxiter,'fatol':gradtol,'jac_options':{'method':krylov_method}},callback=best_sol.update)

        print('Approximate solution found ! Action Grad Norm : ',best_sol.f_norm)

        Found_duplicate = False

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
                        plot_all_2D(best_sol.x,nint_plot_img,callfun,filename_output+'.png',fig_size=img_size)
                        
                    if Save_anim :
                        plot_all_2D_anim(best_sol.x,nint_plot_anim,callfun,filename_output+'.mp4',nperiod_anim,Plot_trace=Plot_trace_anim,fig_size=vid_size)
                    
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

