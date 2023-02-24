import os

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


__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(__PROJECT_ROOT__)


os.environ['OMP_NUM_THREADS'] = '8'

from choreo import *


test_cython_prange()

# exit()

nint = 24

ncoeff = nint//2 + 1


do_perf = False
# do_perf = True

# Compare_FD_grad = False
Compare_FD_grad = True

# Compare_FD_hess = False
Compare_FD_hess = True

# exponent_eps_list = range(16)
exponent_eps_list = [8]


ncoeff_init = ncoeff

print("ncoeffs : ",ncoeff_init)


nbody = 3

# mass = np.ones((nbody))
# mass = np.array([1.+x for x in range(nbody)])
mass = np.array([1.,1.,2.])

Sym_list = []

nbpl = [2,1]
the_lcm = m.lcm(*nbpl)
SymName = None
Sym_list,nbody = Make2DChoreoSymManyLoops(nbpl=nbpl,SymName=SymName)

# MomConsImposed = True
MomConsImposed = False



n_reconverge_it_max = 1
n_grad_change = 1.
ActionSyst = setup_changevar(2,nbody,nint,mass,n_reconverge_it_max,Sym_list=Sym_list,MomCons=MomConsImposed,n_grad_change=n_grad_change)
ncoeffs_args = ActionSyst.coeff_to_param.shape[0]


ActionSyst.ComputeGradBackend = Compute_action_Cython
ActionSyst.ComputeHessBackend = Compute_action_hess_mul_Cython


print('n params ',ncoeffs_args)

x0 = np.random.random((ncoeffs_args))
# x0 = ActionSyst.Package_all_coeffs(all_coeffs)


dxa = np.random.random((ncoeffs_args))
dxb =  np.random.random((ncoeffs_args))


Actiono, Actiongrado = ActionSyst.Compute_action(x0)
Hesso = ActionSyst.Compute_action_hess_mul(x0,dxb)

# print('Action 0 : ',Actiono)
# print(np.linalg.norm(Actiongrado))


ActionSyst.ComputeGradBackend = Compute_action_Cython
ActionSyst.ComputeHessBackend = Compute_action_hess_mul_Cython


ActionSyst.ComputeGradBackend = Compute_action_Cython_2D
ActionSyst.ComputeHessBackend = Compute_action_hess_mul_Cython_2D



Action1, Actiongrad1 = ActionSyst.Compute_action(x0)
Hess1 = ActionSyst.Compute_action_hess_mul(x0,dxb)

err = abs(Actiono - Action1) / (abs(Actiono) + abs(Action1))
print("Backend change action error :",err)
err = np.linalg.norm(Actiongrado - Actiongrad1) / (np.linalg.norm(Actiongrado) + np.linalg.norm(Actiongrad1))
print("Backend change grad error :",err)
err = np.linalg.norm(Hesso - Hess1) / (np.linalg.norm(Hesso) + np.linalg.norm(Hess1))
print("Backend change grad error :",err)




ActionSyst.ComputeGradBackend = Compute_action_Cython
ActionSyst.ComputeHessBackend = Compute_action_hess_mul_Cython



dfdxa = np.dot(Actiongrado,dxa)
Hdxb = ActionSyst.Compute_action_hess_mul(x0,dxb)





if do_perf:
    nperf = 100

    tstart = time.perf_counter()
    for iperf in range(nperf):
        Actiono, Actiongrado = ActionSyst.Compute_action(x0)
    tstop = time.perf_counter()
    print("GRAD fft YES recompute time ",tstop-tstart)

    tstart = time.perf_counter()
    for iperf in range(nperf):
        Hdxb = ActionSyst.Compute_action_hess_mul(x0,dxb)
    tstop = time.perf_counter()
    print("HESS fft YES recompute time ",tstop-tstart)

    ActionSyst.Do_Pos_FFT = False
    tstart = time.perf_counter()
    for iperf in range(nperf):
        Hdxb = ActionSyst.Compute_action_hess_mul(x0,dxb)
    tstop = time.perf_counter()
    print("HESS fft NO recompute time ",tstop-tstart)

    ActionSyst.Do_Pos_FFT = True

    tstart = time.perf_counter()
    HessMat = ActionSyst.Compute_action_hess_LinOpt(x0)
    w ,v = sp.linalg.eigsh(HessMat,k=45,which='SA')
    tstop = time.perf_counter()
    print("EIG fft YES recompute time ",tstop-tstart)

    ActionSyst.Do_Pos_FFT = False

    tstart = time.perf_counter()
    HessMat = ActionSyst.Compute_action_hess_LinOpt(x0)
    w ,v = sp.linalg.eigsh(HessMat,k=10,which='SA')
    tstop = time.perf_counter()
    print("EIG fft NO recompute time ",tstop-tstart)

    ActionSyst.Do_Pos_FFT = True


    
if Compare_FD_grad:

    epslist = []
    Abs_difflist = []
    Rel_difflist = []

    for exponent_eps in exponent_eps_list:
        
        eps = 10**(-exponent_eps)
        
        # Second order approx
        # xp = np.copy(x0) + eps*dxa
        # fp, gfp = ActionSyst.Compute_action(xp)
        
        # xm = np.copy(x0) - eps*dxa
        # fm, gfm = ActionSyst.Compute_action(xm)
        
        # df_difffin = (fp-fm)/(2*eps)
        
        # First order scipy_like approx
        xp = np.copy(x0) + eps*dxa
        fp, gfp = ActionSyst.Compute_action(xp)
        
        xm = np.copy(x0)
        fm, gfm = ActionSyst.Compute_action(xm)
        
        df_difffin = (fp-fm)/(eps)
        
        print('')
        epslist.append(eps)
        print('eps : ',eps)
        err_vect = df_difffin-dfdxa
        print('DF : ',np.linalg.norm(df_difffin))
        print('EX : ',np.linalg.norm(dfdxa))

        Abs_diff = np.linalg.norm(err_vect)
        Abs_difflist.append(Abs_diff)
        print('Abs_diff : ',Abs_diff)
        Rel_diff = abs(err_vect)/(abs(df_difffin)+abs(dfdxa))
        Rel_difflist.append(Rel_diff)
        print('Rel_diff : ',Rel_diff)

        
    fig = plt.figure()
    fig.set_size_inches(10, 8)
    ax = fig.add_subplot(111)
    
    plt.plot(epslist,Rel_difflist)
    
    ax.invert_xaxis()
    plt.yscale('log')
    plt.xscale('log')
    
    plt.tight_layout()
    
    filename = './FD_cvgence_grad.png'
    
    plt.savefig(filename)
    
    plt.close()

    
if Compare_FD_hess:

    epslist = []
    Abs_difflist = []
    Rel_difflist = []

    for exponent_eps in exponent_eps_list:
        
        eps = 10**(-exponent_eps)
        
        # Second order approx
        # xp = np.copy(x0) + eps*dxb
        # fp, gfp = ActionSyst.Compute_action(xp)
        # dfp = np.dot(gfp,dxa)
        
        # xm = np.copy(x0) - eps*dxb
        # fm, gfm = ActionSyst.Compute_action(xm)
        # dfm = np.dot(gfm,dxa)
        
        # dgf_difffin = (gfp-gfm)/(2*eps)
        
        # First order scipy_like approx
        xp = np.copy(x0) + eps*dxb
        fp, gfp = ActionSyst.Compute_action(xp)
        dfp = np.dot(gfp,dxa)
        
        xm = np.copy(x0)
        fm, gfm = ActionSyst.Compute_action(xm)
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
    
    filename = './FD_cvgence_hess.png'
    
    plt.savefig(filename)
    
    plt.close()
