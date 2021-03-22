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

store_folder = './Sniff_all/2x2x2bis/'
# ~ ncoeff_cutoff = 100
ncoeff_cutoff = 300

all_coeffs = np.load('./Sniff_all/2x2x2/46.npy')



nloop = all_coeffs.shape[0]
ncoeff = all_coeffs.shape[2]
nbody = np.array([2,1,1])
mass = np.ones((nloop))


# ~ print(nloop,nbody,ncoeff)
# ~ nint_plot = 1000
# ~ nperiod = 2
# ~ filename_output='./film'
# ~ plot_all_2D_anim(nloop,nbody,nint_plot,nperiod,all_coeffs,filename_output+'.mp4',Plot_trace=False)




# ~ # Duplicate test
theta_rot_dupl = np.linspace(start=0.,stop=twopi,endpoint=False,num=nbody[0]*4)
dt_shift_dupl = np.linspace(start=0.,stop=1.,endpoint=False,num=nbody[0]*4)
duplicate_eps = 1e-9

Found_duplicate,dist_sols,file_path = Check_Duplicates(store_folder,all_coeffs,nbody,duplicate_eps,Action_val=37.2925249298,ncoeff_cutoff=ncoeff_cutoff,theta_rot_dupl=theta_rot_dupl,dt_shift_dupl=dt_shift_dupl,TimeReversal=True,SpaceSym=True)         
    
print('Found_Duplicate : ',Found_duplicate)
print('Dist : ',dist_sols)
print('File ',file_path)
    
    
    
    
    
    
    
    
    
    
    
    
# ~ change_var_coeff = 1.
# ~ VelChangeVar = {
# ~ 'direct' : lambda nloop,nbody,ncoeff,all_coeffs :  VelChangeDirect(nloop,nbody,ncoeff,all_coeffs,change_var_coeff),
# ~ 'inverse' : lambda nloop,nbody,ncoeff,all_coeffs :  VelChangeInverse(nloop,nbody,ncoeff,all_coeffs,change_var_coeff),
# ~ 'Grad' : lambda nloop,nbody,ncoeff,all_idx,Action_grad :  VelChangeGrad(nloop,nbody,ncoeff,all_idx,Action_grad,change_var_coeff),
# ~ }
    
    
# ~ print('Reconverging solution')

# ~ Newt_err_norm = 1.
# ~ Newt_err_norm_max = 1e-9

# ~ n_reconverge_it = 0
# ~ n_reconverge_it_max = 3

# ~ disp_scipy_opt = False

# ~ while ((Newt_err_norm > Newt_err_norm_max) and (n_reconverge_it < n_reconverge_it_max)):

    
    # ~ n_reconverge_it = n_reconverge_it + 1
    
    # ~ all_coeffs_old = np.copy(all_coeffs)
    
    # ~ ncoeff_new = ncoeff * 2

    # ~ all_coeffs = np.zeros((nloop,ndim,ncoeff_new,2),dtype=np.float64)
    # ~ for k in range(ncoeff):
        # ~ all_coeffs[:,:,k,:] = all_coeffs_old[:,:,k,:]
        
    # ~ ncoeff = ncoeff_new
    # ~ nint = 2*ncoeff
    
    # ~ n_idx,all_idx =  setup_idx(nloop,nbody,ncoeff)
    # ~ x0,callfun = Package_args(nloop,nbody,ncoeff,mass,nint,all_coeffs,n_idx,all_idx,VelChangeVar)
    
    # ~ Action,GradAction = Compute_action(nloop,nbody,ncoeff,mass,nint,all_coeffs,n_idx,all_idx)
    # ~ opt_grad_norm = np.linalg.norm(GradAction)
    # ~ print('After Resize : Action Grad Norm : ',opt_grad_norm)
    
    # ~ maxiter = 10
    # ~ gradtol = 1e-15
    # ~ krylov_method = 'lgmres'

    # ~ opt_result = opt.root(fun=Compute_action_onlygrad_package,x0=x0,args=callfun,method='krylov', options={'disp':disp_scipy_opt,'maxiter':maxiter,'fatol':gradtol,'jac_options':{'method':krylov_method}})

    # ~ x_opt = opt_result['x']
    # ~ all_coeffs = Unpackage_all_coeffs(x_opt,callfun)
    
    # ~ opt_grad = opt_result['fun']
    # ~ opt_grad_norm = np.linalg.norm(opt_grad)
    # ~ print('Opt Action Grad Norm : ',opt_grad_norm)

    # ~ Newt_err = Compute_Newton_err(nloop,nbody,ncoeff,mass,nint,all_coeffs,n_idx,all_idx)
    # ~ Newt_err_norm = 0.
    # ~ for ib in range(len(Newt_err)):
        # ~ Newt_err_norm += np.linalg.norm(Newt_err[ib])
    
    # ~ print('Newton Error : ',Newt_err_norm)




# ~ nint_plot = 1000
# ~ imgfilename = './test'
# ~ plot_all_2D(nloop,nbody,nint_plot,all_coeffs,imgfilename+'.png')
