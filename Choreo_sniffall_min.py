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

nbody = 3
mass = np.ones((nbody))

Sym_list = []

SymType = {
    'name'  : 'C',
    'n'     : 2,
    'k'     : 1,
    'l'     : 1 ,
    'p'     : 0 ,
    'q'     : 2 ,
}

Sym_list.extend(Make2DChoreoSym(SymType,[0,1]))

Sym = ChoreoSym(
    LoopTarget=2,
    LoopSource=2,
    SpaceRot = np.identity(ndim),
    TimeRev=1,
    TimeShift=fractions.Fraction(numerator=1,denominator=2)
    )

Sym_list.append(Sym)

# ~ Sym_list.extend(Make2DChoreoSym(SymType,[1]))

# ~ Sym_list.extend(Make2DChoreoSym(SymType,[2]))



store_folder = './Sniff_all_sym/'
store_folder = store_folder+str(nbody)
if not(os.path.isdir(store_folder)):
    os.mkdir(store_folder)



Look_for_duplicates = True
# ~ Look_for_duplicates = False

Check_loop_dist = True
# ~ Check_loop_dist = False

save_init = False
# ~ save_init = True

save_approx = False
# ~ save_approx = True

# ~ Reconverge_sols = False
Reconverge_sols = True

Save_anim = True
# ~ Save_anim = False

n_reconverge_it_max = 3
# ~ n_reconverge_it_max = 0

# ~ ncoeff_init = 100
# ~ ncoeff_init = 800
# ~ ncoeff_init = 700
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

callfun_list = []

print('Searching periodic solutions of {:d} bodies'.format(nbody))
# ~ print('Processing symmetries for {:d} convergence levels ...'.format(n_reconverge_it_max+1))

for i in range(n_reconverge_it_max+1):
    
    print('Processing symmetries for convergence level {0:d} of {1:d}'.format(i+1,n_reconverge_it_max+1))
    
    ncoeff = ncoeff_init * (2**i)
    
    callfun = setup_changevar(nbody,ncoeff,mass,Sym_list=Sym_list)

    callfun_list.append(callfun)

print('')

args = callfun_list[0][0]
nloop = args['nloop']
loopnb = args['loopnb']
loopnbi = args['loopnbi']
nbi_tot = 0
for il in range(nloop):
    for ilp in range(il+1,nloop):
        nbi_tot += loopnb[il]*loopnb[ilp]
    nbi_tot += loopnbi[il]
nbi_naive = (nbody*(nbody-1))//2


not_disp_list = []
not_disp_list = ['coeff_to_param','param_to_coeff']


# ~ for key,value in args.items():
    # ~ if key not in not_disp_list:
        # ~ print(key)
        # ~ print(value)
        # ~ print('')
    # ~ else:
        # ~ print(key)
        # ~ print(value.shape)
        # ~ print('')


print('Imposed constraints lead to the detection of :')
print('    {:d} independant loops'.format(nloop))
print('    {0:d} binary interactions'.format(nbi_tot))
print('    ==> reduction of {0:f} % wrt the {1:d} naive binary iteractions'.format(100*(1-nbi_tot/nbi_naive),nbi_naive))
print('')



# ~ for i in range(n_reconverge_it_max+1):
for i in [0]:
    
    args = callfun_list[i][0]
    print('Convergence attempt number : ',i+1)
    print('    Number of scalar parameters before symmetries : ',args['coeff_to_param'].shape[1])
    print('    Number of scalar parameters after  symmetries : ',args['coeff_to_param'].shape[0])
    print('    Reduction of ',100*(1-args['coeff_to_param'].shape[0]/args['coeff_to_param'].shape[1]),' %')
    print('')
    

callfun = callfun_list[0]
x0 = np.random.random(callfun[0]['param_to_coeff'].shape[1])
xmin = Compute_MinDist(x0,callfun)
if (xmin < 1e-5):
    print(xmin)
    raise ValueError("Init inter body distance too low. There is something wrong with constraints")



n_opt = 0
# ~ n_opt_max = 1
n_opt_max = 1e10
while (n_opt < n_opt_max):

    n_opt += 1
    
    print('Optimization attempt number : ',n_opt)
    
    callfun = callfun_list[0]
    
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
            
                ko = 0
                k1 =20
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
                
      
                all_coeffs[il,idim,k,0] = randampl*np.cos(randphase)
                all_coeffs[il,idim,k,1] = randampl*np.sin(randphase)
                
    x0 = Package_all_coeffs(all_coeffs,callfun)
    
    gradtol = 1e-5
    maxiter = 1000

    opt_result = opt.minimize(fun=Compute_action,x0=x0,args=callfun,method='trust-krylov',jac=True,hessp=Compute_action_hess_mul,options={'disp':disp_scipy_opt,'maxiter':maxiter,'gtol' : gradtol,'inexact': True})
    
    Go_On = True
    
    if (Check_loop_dist):
        
        x_opt = opt_result['x']
        Go_On = not(Detect_Escape(x_opt,callfun))

        if not(Go_On):
            print('One loop escaped. Starting over')    
    
    if (Go_On):

        x0 = opt_result['x']
        f0 = Compute_action_onlygrad(x0,callfun)
        best_sol = current_best(x0,f0)

        maxiter = 10
        gradtol = 1e-11
        opt_result = opt.root(fun=Compute_action_onlygrad,x0=x0,args=callfun,method='krylov', options={'disp':disp_scipy_opt,'maxiter':maxiter,'fatol':gradtol,'jac_options':{'method':krylov_method}},callback=best_sol.update)

        all_coeffs = Unpackage_all_coeffs(best_sol.x,callfun)
        
        print('Approximate solution found ! Action Grad Norm : ',best_sol.f_norm)

        if (save_approx):
            nint_plot = 200
            plot_all_2D(nloop,nbody,nint_plot,all_coeffs,'approx.png')

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
                    
                    callfun = callfun_list[n_reconverge_it]
                    x0 = Package_all_coeffs(all_coeffs,callfun)
                    
                    f0 = Compute_action_onlygrad(x0,callfun)
                    best_sol = current_best(x0,f0)
                    
                    print('After Resize : Action Grad Norm : ',best_sol.f_norm)
                    
                    gradtol = 1e-7
                    maxiter = 1000
                    opt_result = opt.minimize(fun=Compute_action,x0=x0,args=callfun,method='trust-krylov',jac=True,hessp=Compute_action_hess_mul,options={'disp':disp_scipy_opt,'maxiter':maxiter,'gtol' : gradtol,'inexact': True})

                    x0 = opt_result['x']

                    maxiter = 20
                    gradtol = 1e-15
                    opt_result = opt.root(fun=Compute_action_onlygrad,x0=x0,args=callfun,method='krylov', options={'disp':disp_scipy_opt,'maxiter':maxiter,'fatol':gradtol,'jac_options':{'method':krylov_method}},callback=best_sol.update)

                    all_coeffs = Unpackage_all_coeffs(best_sol.x,callfun)
                    
                    print('Opt Action Grad Norm : ',best_sol.f_norm)
                
                    Newt_err = Compute_Newton_err(best_sol.x,callfun)
                    Newt_err_norm = np.linalg.norm(Newt_err)/args['nint']
                    
                    print('Newton Error : ',Newt_err_norm)
                
                    SaveSol = (Newt_err_norm < Newt_err_norm_max_save)
                                    
                    if (Check_loop_dist):
                        
                        Go_On = not(Detect_Escape(best_sol.x,callfun))

                        if not(Go_On):
                            print('One loop escaped. Starting over')    
                
                if not(SaveSol):
                    print('Newton Error too high, discarding solution')
            
            else:
                
                SaveSol = True
            
            if ((SaveSol) or (Save_Bad_Sols)):
                        
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
                    
                    nint_plot = 10000
                    # ~ np.save(filename_output+'.npy',all_coeffs)
                    plot_all_2D(best_sol.x,nint_plot,callfun,filename_output+'.png')
                    
                    
                    # ~ print(all_coeffs)
                    # ~ print(1/0)
                    
                    if Save_anim :
                        nint_plot = 1000
                        nperiod = 1
                        plot_all_2D_anim(best_sol.x,nint_plot,callfun,filename_output+'.mp4',nperiod,Plot_trace=True)
                    Write_Descriptor(best_sol.x,callfun,filename_output+'.txt')
                    # ~ pickle.dump(best_sol.x,filename_output+'_params.txt'
            
    
    
    print('')
    print('')
    print('')

