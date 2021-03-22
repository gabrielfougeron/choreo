import numpy as np
import math as m
import scipy.optimize as opt
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import cnames
from matplotlib import animation
import copy

import time


from Choreo_funs import *


# ~ A =  np.random.rand(100, 2)
# ~ # Cast the array as a complex array
# ~ A_comp = A.view(dtype=np.complex128)
# ~ # To get the original array A back from the complex version
# ~ A = A.view(dtype=np.float64)




# ~ nloop = 1
nloop = 3
ndim = 2
ncoeff = 2*3*5

nint = 4*(ncoeff)
# ~ nint =   4*(ncoeff-1)

nbody = np.array([2,3,5])
# ~ nbody = np.array([2])
mass = np.ones((nloop))

all_coeffs = np.zeros((nloop,ndim,ncoeff,2),dtype=np.float64)

np.random.seed(10)

amplitude_o = 0.1

for il in range(nloop):
    for idim in range(ndim):
        for k in range(ncoeff):
            
            randphase = np.random.rand() * twopi
            randampl = np.random.rand()* amplitude_o
        
            ko = 2
            if (k <= ko):
                randampl = 0.
            
        
            k_thresh_damp = 5
            
            if (k >= k_thresh_damp):
                randampl = randampl / ((k-k_thresh_damp+1)**2)
                
            all_coeffs[il,idim,k,0] = randampl*np.cos(randphase)
            all_coeffs[il,idim,k,1] = randampl*np.sin(randphase)

all_coeffs2 = np.zeros((nloop,ndim,ncoeff,2),dtype=np.float64)

for il in range(nloop):
    for idim in range(ndim):
        for k in range(ncoeff):
            
            randphase = np.random.rand() * twopi
            randampl = np.random.rand()* amplitude_o
        
            ko = 2
            if (k <= ko):
                randampl = 0.
            
        
            k_thresh_damp = 5
            
            if (k >= k_thresh_damp):
                randampl = randampl / ((k-k_thresh_damp+1)**2)
                
            all_coeffs2[il,idim,k,0] = randampl*np.cos(randphase)
            all_coeffs2[il,idim,k,1] = randampl*np.sin(randphase)
            



# ~ input_filename = 'opt_result'
# ~ input_filename = 'opt_result_2'
# ~ input_filename = 'opt_result_3'

# ~ all_coeffs = np.load(input_filename+'.npy')
# ~ nloop = all_coeffs.shape[0]
# ~ ndim = all_coeffs.shape[1]
# ~ ncoeff = all_coeffs.shape[2]

# ~ nint = 2*ncoeff

# ~ nbody = np.array([3,4])
# ~ mass = np.ones((nloop))

n_idx,all_idx =  setup_idx(nloop,nbody,ncoeff)
# ~ change_var_coeff = 0.
change_var_coeff = 1.


VelChangeVar = {
'direct' : lambda nloop,nbody,ncoeff,all_coeffs :  VelChangeDirect(nloop,nbody,ncoeff,all_coeffs,change_var_coeff),
'inverse' : lambda nloop,nbody,ncoeff,all_coeffs :  VelChangeInverse(nloop,nbody,ncoeff,all_coeffs,change_var_coeff),
'Grad' : lambda nloop,nbody,ncoeff,all_idx,Action_grad :  VelChangeGrad(nloop,nbody,ncoeff,all_idx,Action_grad,change_var_coeff),
}

x0,callfun = Package_args(nloop,nbody,ncoeff,mass,nint,all_coeffs,n_idx,all_idx,VelChangeVar)


Actiono, Actiongrado = Compute_action_package(x0,callfun)
# ~ Actiono, Actiongrado = Compute_action_gradnormsq_package(x0,callfun)
# ~ sq_disto, sq_distgrado = sq_dist_transform_2d(nloop,ncoeff,all_coeffs,all_coeffs2,x0)
sq_disto, sq_distgrado = sq_dist_transform_2d_noscal(nloop,ncoeff,all_coeffs,all_coeffs2,x0)

# ~ print('Action 0 : ',Actiono)

ncoeffs_args = x0.shape[0]



dx = np.random.random((ncoeffs_args))



# ~ dx[0] = 0
# ~ dx[1] = 0
# ~ dx[2] = 0
# ~ dx[3] = 0


df_ex = np.dot(Actiongrado,dx)

print('\n\n\n')

for exponent_eps in range(0):
# ~ for exponent_eps in range(8):
    
    eps = 10**(-exponent_eps)


    xp = np.copy(x0) + eps*dx
    fp ,gfp = Compute_action_package(xp,callfun)
    # ~ fp ,gfp = Compute_action_gradnormsq_package(xp,callfun)
    # ~ fp ,gfp = sq_dist_transform_2d_noscal(nloop,ncoeff,all_coeffs,all_coeffs2,xp)
    xm = np.copy(x0) - eps*dx
    fm ,gfm = Compute_action_package(xm,callfun)
    # ~ fm ,gfm = Compute_action_gradnormsq_package(xm,callfun)
    # ~ fm ,gfm = sq_dist_transform_2d_noscal(nloop,ncoeff,all_coeffs,all_coeffs2,xm)

    df_difffin = (fp-fm)/(2*eps)

    print('')
    print('eps : ',eps)
    print('df : ',df_difffin,df_ex)
    print('Abs_diff : ',abs(df_difffin-df_ex))
    print('Rel_diff : ',abs(df_difffin-df_ex)/((abs(df_ex)+abs(df_difffin))/2))



dxa = np.random.random((ncoeffs_args))
dxb =  np.random.random((ncoeffs_args))

# ~ dxa = np.zeros((ncoeffs_args))
# ~ dxb =  np.zeros((ncoeffs_args))

# ~ i_nz =  all_idx[0,0,2,0]
# ~ j_nz =  all_idx[1,1,1,0]
# ~ dxa[i_nz] = 1.
# ~ dxb[j_nz] = 1.

# ~ dxa = np.random.random((ncoeffs_args))
# ~ dxb =  np.zeros((ncoeffs_args))

# ~ j_nz = all_idx[0,0,1,0]
# ~ dxb[j_nz] = 1.

Hdxb = Compute_action_hess_mul_package(x0,dxb,callfun)

    



# ~ for exponent_eps in [8]:
for exponent_eps in range(16):
    
    eps = 10**(-exponent_eps)
    
    xp = np.copy(x0) + eps*dxb
    fp, gfp = Compute_action_package(xp,callfun)
    dfp = np.dot(gfp,dxa)
    
    xm = np.copy(x0) - eps*dxb
    fm, gfm = Compute_action_package(xm,callfun)
    dfm = np.dot(gfm,dxa)
    
    dgf_difffin = (gfp-gfm)/(2*eps)
    
    print('')
    print('eps : ',eps)
    print('Abs_diff : ')
    err_vect = dgf_difffin-Hdxb
    print('DF : ',np.linalg.norm(dgf_difffin))
    print('EX : ',np.linalg.norm(Hdxb))
    print('Abs_diff : ',np.linalg.norm(err_vect))
    print('Rel_diff : ',np.linalg.norm(err_vect)/(np.linalg.norm(dgf_difffin)+np.linalg.norm(Hdxb)))
    
    # ~ ddf_difffin = (dfp-dfm)/(2*eps)
    
    # ~ print('')
    # ~ print('eps : ',eps)
    # ~ print('df vals : ',ddf_difffin,ddf_fft_d)
    # ~ print('Abs_diff : ',abs(ddf_difffin-ddf_fft_d))
    # ~ print('Rel_diff : ',abs(ddf_difffin-ddf_fft_d)/((abs(ddf_fft_d)+abs(ddf_difffin))/2))
    


