import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'

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



nint = 24

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


dxa = np.zeros((ncoeffs_args))
dxb =  np.random.random((ncoeffs_args))



ActionSyst.ComputeGradBackend = Compute_action_Cython
ActionSyst.ComputeHessBackend = Compute_action_hess_mul_Cython


Actiono, Actiongrado = ActionSyst.Compute_action(x0)
Hesso = ActionSyst.Compute_action_hess_mul(x0,dxb)

# print('Action 0 : ',Actiono)
# print(np.linalg.norm(Actiongrado))



ActionSyst.ComputeGradBackend = Compute_action_Cython_2D
ActionSyst.ComputeHessBackend = Compute_action_hess_mul_Cython_2D



Action1, Actiongrad1 = ActionSyst.Compute_action(x0)
Hess1 = ActionSyst.Compute_action_hess_mul(x0,dxb)

err = abs(Actiono - Action1) / (abs(Actiono) + abs(Action1))
print("Backend change action error :",err)
err = np.linalg.norm(Actiongrado - Actiongrad1) / (np.linalg.norm(Actiongrado) + np.linalg.norm(Actiongrad1))
print("Backend change grad error :",err)
err = np.linalg.norm(Hesso - Hess1) / (np.linalg.norm(Hesso) + np.linalg.norm(Hess1))
print("Backend change hess error :",err)

