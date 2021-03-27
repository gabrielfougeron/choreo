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




nbody = 3
mass = np.ones((nbody))
ncoeff = 12
rotangle = 2*np.pi * 0/3
mirror = 1

rotmat = np.array([[mirror*np.cos(rotangle),-mirror*np.sin(rotangle)],[np.sin(rotangle),np.cos(rotangle)]],dtype=np.float64)

Sym_list = []

Sym_list.append(ChoreoSym(
    LoopTarget=1,
    LoopSource=0,
    SpaceRot = rotmat,
    TimeRev=1,
    TimeShift=fractions.Fraction(numerator=1,denominator=3)
    ))

Sym_list.append(ChoreoSym(
    LoopTarget=2,
    LoopSource=1,
    SpaceRot = rotmat,
    TimeRev=1,
    TimeShift=fractions.Fraction(numerator=1,denominator=3)
    ))

# ~ Sym_list.append(ChoreoSym(
    # ~ LoopTarget=0,
    # ~ LoopSource=2,
    # ~ SpaceRot = rotmat,
    # ~ TimeRev=1,
    # ~ TimeShift=fractions.Fraction(numerator=1,denominator=3)
    # ~ ))


callfun = setup_changevar(nbody,ncoeff,mass,Sym_list=Sym_list)

ncoeffs_args = callfun[0]['coeff_to_param'].shape[0]

x0 = np.random.random((ncoeffs_args))

# ~ not_disp_list = []
# ~ not_disp_list = ['coeff_to_param','param_to_coeff']


# ~ for key,value in callfun[0].items():
    # ~ if key not in not_disp_list:
        # ~ print(key)
        # ~ print(value)
        # ~ print('')
    # ~ else:
        # ~ print(key)
        # ~ print(value.shape)
        # ~ print('')


# ~ print(callfun)

Actiono, Actiongrado = Compute_action_package(x0,callfun)
# ~ Actiono, Actiongrado = Compute_action_gradnormsq_package(x0,callfun)
# ~ sq_disto, sq_distgrado = sq_dist_transform_2d(nloop,ncoeff,all_coeffs,all_coeffs2,x0)
# ~ sq_disto, sq_distgrado = sq_dist_transform_2d_noscal(nloop,ncoeff,all_coeffs,all_coeffs2,x0)

# ~ print('Action 0 : ',Actiono)

# ~ print(Actiongrado)




print('\n\n\n')

# ~ for i in range(ncoeffs_args):
for i in range(1):
    dx = np.zeros((ncoeffs_args))
    dx[i] = 1
    
    dx = np.random.random((ncoeffs_args))
    
    
    df_ex = np.dot(Actiongrado,dx)


    for exponent_eps in [8]:
    # ~ for exponent_eps in range(16):
        
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
        
        print(i,df_difffin,df_ex)



# ~ dxa = np.random.random((ncoeffs_args))
# ~ dxb =  np.random.random((ncoeffs_args))

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

# ~ Hdxb = Compute_action_hess_mul_package(x0,dxb,callfun)

    



# ~ for exponent_eps in [8]:
# ~ for exponent_eps in range(16):
    
    # ~ eps = 10**(-exponent_eps)
    
    # ~ xp = np.copy(x0) + eps*dxb
    # ~ fp, gfp = Compute_action_package(xp,callfun)
    # ~ dfp = np.dot(gfp,dxa)
    
    # ~ xm = np.copy(x0) - eps*dxb
    # ~ fm, gfm = Compute_action_package(xm,callfun)
    # ~ dfm = np.dot(gfm,dxa)
    
    # ~ dgf_difffin = (gfp-gfm)/(2*eps)
    
    # ~ print('')
    # ~ print('eps : ',eps)
    # ~ print('Abs_diff : ')
    # ~ err_vect = dgf_difffin-Hdxb
    # ~ print('DF : ',np.linalg.norm(dgf_difffin))
    # ~ print('EX : ',np.linalg.norm(Hdxb))
    # ~ print('Abs_diff : ',np.linalg.norm(err_vect))
    # ~ print('Rel_diff : ',np.linalg.norm(err_vect)/(np.linalg.norm(dgf_difffin)+np.linalg.norm(Hdxb)))
    
    # ~ ddf_difffin = (dfp-dfm)/(2*eps)
    
    # ~ print('')
    # ~ print('eps : ',eps)
    # ~ print('df vals : ',ddf_difffin,ddf_fft_d)
    # ~ print('Abs_diff : ',abs(ddf_difffin-ddf_fft_d))
    # ~ print('Rel_diff : ',abs(ddf_difffin-ddf_fft_d)/((abs(ddf_fft_d)+abs(ddf_difffin))/2))
    


