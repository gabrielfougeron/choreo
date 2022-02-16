import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import sys
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


ncoeff = 12
# ncoeff = 900

# load_file = './save_tests/9/9.npy'
# all_coeffs = np.load(load_file)
# ncoeff = all_coeffs.shape[2]




ncoeff_init = ncoeff

print("ncoeffs : ",ncoeff_init)

# nTf = 101
# nbs = 3
# nbf = 3
# nbody =  nbs * nbf

nbody = 3

# mass = np.ones((nbody))
# mass = np.array([1.+x for x in range(nbody)])
mass = np.array([1.,1,2])

Sym_list = []

nbpl = [2,1]
the_lcm = m.lcm(*nbpl)
SymName = None
Sym_list,nbody = Make2DChoreoSymManyLoops(nbpl=nbpl,SymName=SymName)

MomConsImposed = True
# MomConsImposed = False



n_reconverge_it_max = 1
n_grad_change = 1.
callfun = setup_changevar(nbody,ncoeff_init,mass,n_reconverge_it_max,Sym_list=Sym_list,MomCons=MomConsImposed,n_grad_change=n_grad_change)
ncoeffs_args = callfun[0]['coeff_to_param_list'][0].shape[0]

print('n params ',ncoeffs_args)

x0 = np.random.random((ncoeffs_args))
# x0 = Package_all_coeffs(all_coeffs,callfun)




# not_disp_list = []
# not_disp_list = ['coeff_to_param','param_to_coeff']


# for key,value in callfun[0].items():
    # if key not in not_disp_list:
        # print(key)
        # print(value)
        # print('')
    # else:
        # print(key)
        # print(value.shape)
        # print('')


# print(callfun)

Actiono, Actiongrado = Compute_action(x0,callfun)

# print('Action 0 : ',Actiono)
print(np.linalg.norm(Actiongrado))



epslist = []
Abs_difflist = []
Rel_difflist = []


dxa = np.random.random((ncoeffs_args))
dxb =  np.random.random((ncoeffs_args))


Hdxb = Compute_action_hess_mul(x0,dxb,callfun)



# dxa = np.zeros((ncoeffs_args))
# dxb =  np.zeros((ncoeffs_args))

# i_nz =  all_idx[0,0,2,0]
# j_nz =  all_idx[1,1,1,0]
# dxa[i_nz] = 1.
# dxb[j_nz] = 1.

# dxa = np.random.random((ncoeffs_args))
# dxb =  np.zeros((ncoeffs_args))

# j_nz = all_idx[0,0,1,0]
# dxb[j_nz] = 1.


# nperf = 100

# tstart = time.perf_counter()
# for iperf in range(nperf):
    # Actiono, Actiongrado = Compute_action(x0,callfun)
# tstop = time.perf_counter()
# print("GRAD fft YES recompute time ",tstop-tstart)

# tstart = time.perf_counter()
# for iperf in range(nperf):
    # Hdxb = Compute_action_hess_mul(x0,dxb,callfun)
# tstop = time.perf_counter()
# print("HESS fft YES recompute time ",tstop-tstart)

# callfun[0]["Do_Pos_FFT"] = False
# tstart = time.perf_counter()
# for iperf in range(nperf):
    # Hdxb = Compute_action_hess_mul(x0,dxb,callfun)
# tstop = time.perf_counter()
# print("HESS fft NO recompute time ",tstop-tstart)

callfun[0]["Do_Pos_FFT"] = True



tstart = time.perf_counter()
HessMat = Compute_action_hess_LinOpt(x0,callfun)
w ,v = sp.linalg.eigsh(HessMat,k=45,which='SA')
tstop = time.perf_counter()
print("EIG fft YES recompute time ",tstop-tstart)

# callfun[0]["Do_Pos_FFT"] = False

# tstart = time.perf_counter()
# HessMat = Compute_action_hess_LinOpt(x0,callfun)
# w ,v = sp.linalg.eigsh(HessMat,k=10,which='SA')
# tstop = time.perf_counter()
# print("EIG fft NO recompute time ",tstop-tstart)


# callfun[0]["Do_Pos_FFT"] = True




print(w)

# print(v.shape)
# for i in range(v.shape[1]):
    #for j in range(v.shape[0]):
    # for j in range(4):
        # if (abs(v[j,i]) > 1e-9):
            # print(i,j)
        

# print(v)




# sys.exit(0)


epslist = []
Abs_difflist = []
Rel_difflist = []


# for exponent_eps in [8]:
for exponent_eps in range(16):
    
    eps = 10**(-exponent_eps)
    
    # Second order approx
    # xp = np.copy(x0) + eps*dxb
    # fp, gfp = Compute_action(xp,callfun)
    # dfp = np.dot(gfp,dxa)
    
    # xm = np.copy(x0) - eps*dxb
    # fm, gfm = Compute_action(xm,callfun)
    # dfm = np.dot(gfm,dxa)
    
    # dgf_difffin = (gfp-gfm)/(2*eps)
    
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
    
    # print('')
    # print('eps : ',eps)
    # print('df vals : ',ddf_difffin,ddf_fft_d)
    # print('Abs_diff : ',abs(ddf_difffin-ddf_fft_d))
    # print('Rel_diff : ',abs(ddf_difffin-ddf_fft_d)/((abs(ddf_fft_d)+abs(ddf_difffin))/2))
    

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
