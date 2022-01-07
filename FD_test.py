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


# ~ ncoeff = 3**10

load_file = './save_tests/9/8.npy'
all_coeffs = np.load(load_file)
ncoeff = all_coeffs.shape[2]




ncoeff_init = ncoeff

nTf = 101
nbs = 3
nbf = 3
nbody =  nbs * nbf

mass = np.ones((nbody))

Sym_list = []

nbpl = [nbody]
the_lcm = m.lcm(*nbpl)
SymName = None
Sym_list,nbody = Make2DChoreoSymManyLoops(nbpl=nbpl,SymName=SymName)

# ~ MomConsImposed = True
MomConsImposed = False



n_reconverge_it_max = 1
n_grad_change = 1.
callfun = setup_changevar(nbody,ncoeff_init,mass,n_reconverge_it_max,Sym_list=Sym_list,MomCons=MomConsImposed,n_grad_change=n_grad_change)
ncoeffs_args = callfun[0]['coeff_to_param_list'][0].shape[0]

x0 = np.random.random((ncoeffs_args))
# ~ x0 = Package_all_coeffs(all_coeffs,callfun)




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

Actiono, Actiongrado = Compute_action(x0,callfun)

# ~ print('Action 0 : ',Actiono)
print(np.linalg.norm(Actiongrado))




print('\n\n\n')

epslist = []
Abs_difflist = []
Rel_difflist = []


# ~ for i in range(ncoeffs_args):
# ~ for i in range(1):
for i in range(0):
    dx = np.zeros((ncoeffs_args))
    dx[i] = 1
    
    dx = np.random.random((ncoeffs_args))
    
    
    df_ex = np.dot(Actiongrado,dx)


    for exponent_eps in [8]:
    # ~ for exponent_eps in range(16):
        
        eps = 10**(-exponent_eps)


        xp = np.copy(x0) + eps*dx
        fp ,gfp = Compute_action(xp,callfun)
        # ~ fp ,gfp = Compute_action_gradnormsq(xp,callfun)
        # ~ fp ,gfp = sq_dist_transform_2d_noscal(nloop,ncoeff,all_coeffs,all_coeffs2,xp)
        xm = np.copy(x0) - eps*dx
        fm ,gfm = Compute_action(xm,callfun)
        # ~ fm ,gfm = Compute_action_gradnormsq(xm,callfun)
        # ~ fm ,gfm = sq_dist_transform_2d_noscal(nloop,ncoeff,all_coeffs,all_coeffs2,xm)

        df_difffin = (fp-fm)/(2*eps)

        print('')
        epslist.appdn(eps)
        print('eps : ',eps)
        print('df : ',df_difffin,df_ex)
        Abs_diff = abs(df_difffin-df_ex)
        Abs_difflist.append(Abs_diff)
        print('Abs_diff : ',Abs_diff)
        Rel_diff = abs(df_difffin-df_ex)/((abs(df_ex)+abs(df_difffin))/2)
        Rel_difflist.append(Rel_diff)
        print('Rel_diff : ',Rel_diff)
        
        print(i,df_difffin,df_ex)



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

Hdxb = Compute_action_hess_mul(x0,dxb,callfun)

    

epslist = []
Abs_difflist = []
Rel_difflist = []


# ~ for exponent_eps in [8]:
for exponent_eps in range(16):
    
    eps = 10**(-exponent_eps)
    
    # Second order approx
    # ~ xp = np.copy(x0) + eps*dxb
    # ~ fp, gfp = Compute_action(xp,callfun)
    # ~ dfp = np.dot(gfp,dxa)
    
    # ~ xm = np.copy(x0) - eps*dxb
    # ~ fm, gfm = Compute_action(xm,callfun)
    # ~ dfm = np.dot(gfm,dxa)
    
    # ~ dgf_difffin = (gfp-gfm)/(2*eps)
    
    # First order scipy_like approx
    xp = np.copy(x0) + eps*dxb
    fp, gfp = Compute_action(xp,callfun)
    dfp = np.dot(gfp,dxa)
    
    xm = np.copy(x0)
    fm, gfm = Compute_action(xm,callfun)
    dfm = np.dot(gfm,dxa)
    
    dgf_difffin = (gfp-gfm)/(eps)
    
    
    
    print('')
    epslist.append(eps)
    print('eps : ',eps)
    err_vect = dgf_difffin-Hdxb
    print('DF : ',np.linalg.norm(dgf_difffin))
    print('EX : ',np.linalg.norm(Hdxb))

    Abs_diff = np.linalg.norm(err_vect)
    Abs_difflist.append(Abs_diff)
    print('Abs_diff : ',Abs_diff)
    Rel_diff = np.linalg.norm(err_vect)/(np.linalg.norm(dgf_difffin)+np.linalg.norm(Hdxb))
    Rel_difflist.append(Rel_diff)
    print('Rel_diff : ',Rel_diff)

    ddf_difffin = (dfp-dfm)/(2*eps)
    
    # ~ print('')
    # ~ print('eps : ',eps)
    # ~ print('df vals : ',ddf_difffin,ddf_fft_d)
    # ~ print('Abs_diff : ',abs(ddf_difffin-ddf_fft_d))
    # ~ print('Rel_diff : ',abs(ddf_difffin-ddf_fft_d)/((abs(ddf_fft_d)+abs(ddf_difffin))/2))
    

fig = plt.figure()
fig.set_size_inches(10, 8)
ax = fig.add_subplot(111)

plt.plot(epslist,Rel_difflist)

ax.invert_xaxis()
plt.yscale('log')
plt.xscale('log')

plt.tight_layout()

filename = './FD_cvgence.png'

plt.savefig(filename)

plt.close()
