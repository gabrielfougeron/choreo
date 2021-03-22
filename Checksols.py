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


store_folder = './Sniff_all'
reconverge_folder = './Sniff_all_reconverge'

Newt_err_plot_eps = 1e-9
disp_scipy_opt = False


for filename in os.listdir(store_folder):
    file_path = os.path.join(store_folder, filename)
    file_root, file_ext = os.path.splitext(os.path.basename(file_path))
    
    if (file_ext == '.npy' ):
        
        print('Testing found solution '+file_path)
        
        all_coeffs_old = np.load(file_path)
            
        nloop = all_coeffs_old.shape[0]
        ndim = all_coeffs_old.shape[1]
        ncoeff_old = all_coeffs_old.shape[2]
        
        mass = np.ones((nloop))
        nbody = np.array([5])
        
        # ~ ncoeff_new = ncoeff_old 
        # ~ ncoeff_new = ncoeff_old * 5
        ncoeff_new = ncoeff_old * 2

        all_coeffs = np.zeros((nloop,ndim,ncoeff_new,2),dtype=np.float64)
        for k in range(min(ncoeff_old,ncoeff_new)):
            all_coeffs[:,:,k,:] = all_coeffs_old[:,:,k,:]
            
        ncoeff = ncoeff_new
        nint = 2*ncoeff
        change_var_coeff = 1.
        
        n_idx,all_idx =  setup_idx(nloop,nbody,ncoeff)

        VelChangeVar = {
        'direct' : lambda nloop,nbody,ncoeff,all_coeffs :  VelChangeDirect(nloop,nbody,ncoeff,all_coeffs,change_var_coeff),
        'inverse' : lambda nloop,nbody,ncoeff,all_coeffs :  VelChangeInverse(nloop,nbody,ncoeff,all_coeffs,change_var_coeff),
        'Grad' : lambda nloop,nbody,ncoeff,all_idx,Action_grad :  VelChangeGrad(nloop,nbody,ncoeff,all_idx,Action_grad,change_var_coeff),
        }


        x0,callfun = Package_args(nloop,nbody,ncoeff,mass,nint,all_coeffs,n_idx,all_idx,VelChangeVar)
        
        Newt_err = Compute_Newton_err(nloop,nbody,ncoeff,mass,nint,all_coeffs,n_idx,all_idx)
        Newt_err_norm = np.linalg.norm(Newt_err)
        
        print('Newton Error : ',Newt_err_norm)

        if (Newt_err_norm > Newt_err_plot_eps):
            
            
            print('Newton Error is above tolerance. Trying to re-converge solution')
            
            gradtol = 1e-5
            maxiter = 1000
            opt_result = opt.minimize(fun=Compute_action_package,x0=x0,args=callfun,method='trust-krylov',jac=True,hessp=Compute_action_hess_mul_package,options={'disp':disp_scipy_opt,'maxiter':maxiter,'gtol' : gradtol,'inexact': True})

            x0 = opt_result['x']

            maxiter = 10
            gradtol = 1e-13
            krylov_method = 'lgmres'
            # ~ krylov_method = 'gmres'
            # ~ krylov_method = 'bicgstab'
            # ~ krylov_method = 'cgs'
            # ~ krylov_method = 'minres'
            opt_result = opt.root(fun=Compute_action_onlygrad_package,x0=x0,args=callfun,method='krylov', options={'disp':disp_scipy_opt,'maxiter':maxiter,'fatol':gradtol,'jac_options':{'method':krylov_method}})

            x_opt = opt_result['x']
            all_coeffs = Unpackage_all_coeffs(x_opt,callfun)

            Newt_err = Compute_Newton_err(nloop,nbody,ncoeff,mass,nint,all_coeffs,n_idx,all_idx)
            Newt_err_norm = np.linalg.norm(Newt_err)
            
            print('Reconverged Newton Error : ',Newt_err_norm)

            print('')
            print('CREATING MOVIE')
            
            filename_output = reconverge_folder+'/'+file_root+'_bis'
            
            nint_plot = 1000
            nperiod = 2
            np.save(filename_output+'.npy',all_coeffs)
            plot_all_2D(nloop,nbody,nint_plot,all_coeffs,filename_output+'.png')
            plot_all_2D_anim(nloop,nbody,nint_plot,nperiod,all_coeffs,filename_output+'.mp4')
            
            
            
        else:
            
            print('Newton Error is below tolerance')
        

        
        
        print('')
        print('')
