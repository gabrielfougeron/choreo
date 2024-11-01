import os
import multiprocessing

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

n_proc = multiprocessing.cpu_count()
# n_proc = multiprocessing.cpu_count() // 2

val = str(n_proc)

print(f'Parallel backends are launched on {val} threads')

os.environ['OMP_NUM_THREADS'] = val
os.environ['NUMBA_NUM_THREADS'] = val

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

from choreo import *

nint = 2400

ncoeff = nint//2 + 1

print('ncoeff ',ncoeff)


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

nbpl = [1,1,1]
the_lcm = m.lcm(*nbpl)
SymName = None
Sym_list,nbody = Make2DChoreoSymManyLoops(nbpl=nbpl,SymName=SymName)

# MomConsImposed = True
MomConsImposed = False


n_reconverge_it_max = 1
n_grad_change = 1.
ActionSyst = setup_changevar(2,nbody,nint,mass,n_reconverge_it_max,Sym_list=Sym_list,MomCons=MomConsImposed,n_grad_change=n_grad_change)
ActionSystForced = setup_changevar(2,nbody,nint,mass,n_reconverge_it_max,Sym_list=Sym_list,MomCons=MomConsImposed,n_grad_change=n_grad_change,ForceMatrixChangevar=True)

grad_backend_list = [
    Compute_action_Python_2D_serial,
    Compute_action_Cython_2D_serial,
    Compute_action_Numba_2D_serial,
    Compute_action_Python_nD_serial,
    Compute_action_Cython_nD_serial,
    Compute_action_Numba_nD_serial,
    Compute_action_Cython_2D_parallel,
    Compute_action_Numba_2D_parallel,
    Compute_action_Cython_nD_parallel,
    Compute_action_Numba_nD_parallel,
]

hess_backend_list = [
    Compute_action_hess_mul_Python_2D_serial,
    Compute_action_hess_mul_Cython_2D_serial,
    Compute_action_hess_mul_Numba_2D_serial,
    Compute_action_hess_mul_Python_nD_serial,
    Compute_action_hess_mul_Cython_nD_serial,
    Compute_action_hess_mul_Numba_nD_serial,
    Compute_action_hess_mul_Cython_2D_parallel,
    Compute_action_hess_mul_Numba_2D_parallel,
    Compute_action_hess_mul_Cython_nD_parallel,
    Compute_action_hess_mul_Numba_nD_parallel,
]


ActionSyst.ComputeGradBackend = grad_backend_list[0]
ActionSyst.ComputeHessBackend = hess_backend_list[0]


print('n params ',ActionSyst.nparams)

x0 = np.random.random((ActionSyst.nparams))
# x0 = ActionSyst.Package_all_coeffs(all_coeffs)


dxa = np.zeros((ActionSyst.nparams))
dxb =  np.random.random((ActionSyst.nparams))

Actiono, Actiongrado = ActionSyst.Compute_action(x0)
Hesso = ActionSyst.Compute_action_hess_mul(x0,dxb)


print('')
print(f'MatrixFreeChangevar : {ActionSyst.MatrixFreeChangevar}')

for i in range(len(grad_backend_list)):

    print(grad_backend_list[i].__name__)
    print(hess_backend_list[i].__name__)
        
    ActionSyst.ComputeGradBackend = grad_backend_list[i]
    ActionSyst.ComputeHessBackend = hess_backend_list[i]

    Action1, Actiongrad1 = ActionSyst.Compute_action(x0)
    Hess1 = ActionSyst.Compute_action_hess_mul(x0,dxb)

    err = abs(Actiono - Action1) / (abs(Actiono) + abs(Action1))
    print("Backend change action error :",err)
    err = np.linalg.norm(Actiongrado - Actiongrad1) / (np.linalg.norm(Actiongrado) + np.linalg.norm(Actiongrad1))
    print("Backend change grad error :",err)
    err = np.linalg.norm(Hesso - Hess1) / (np.linalg.norm(Hesso) + np.linalg.norm(Hess1))
    print("Backend change hess error :",err)

print('')
print(f'MatrixFreeChangevar : {ActionSystForced.MatrixFreeChangevar}')

for i in range(len(grad_backend_list)):

    print(grad_backend_list[i].__name__)
    print(hess_backend_list[i].__name__)
        
    ActionSystForced.ComputeGradBackend = grad_backend_list[i]
    ActionSystForced.ComputeHessBackend = hess_backend_list[i]

    Action1, Actiongrad1 = ActionSystForced.Compute_action(x0)
    Hess1 = ActionSystForced.Compute_action_hess_mul(x0,dxb)

    err = abs(Actiono - Action1) / (abs(Actiono) + abs(Action1))
    print("Backend change action error :",err)
    err = np.linalg.norm(Actiongrado - Actiongrad1) / (np.linalg.norm(Actiongrado) + np.linalg.norm(Actiongrad1))
    print("Backend change grad error :",err)
    err = np.linalg.norm(Hesso - Hess1) / (np.linalg.norm(Hesso) + np.linalg.norm(Hess1))
    print("Backend change hess error :",err)

