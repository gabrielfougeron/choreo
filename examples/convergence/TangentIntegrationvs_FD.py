"""
Valiadation of tangent integration alorithms using Finite Differences
=====================================================================
"""

# %%
# Evaluation of relative quadrature error with the following parameters:

# sphinx_gallery_start_ignore

import os
import sys
import itertools
import functools

try:
    __PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir,os.pardir))

    if ':' in __PROJECT_ROOT__:
        __PROJECT_ROOT__ = os.getcwd()

except (NameError, ValueError): 

    __PROJECT_ROOT__ = os.path.abspath(os.path.join(os.getcwd(),os.pardir,os.pardir))

sys.path.append(__PROJECT_ROOT__)

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import matplotlib.pyplot as plt
import numpy as np
import math as m
import random
import scipy

import choreo
import choreo.scipy_plus.precomputed_tables as precomputed_tables

bench_folder = os.path.join(__PROJECT_ROOT__,'examples','generated_files')

if not(os.path.isdir(bench_folder)):
    os.makedirs(bench_folder)
    
basename_bench_filename = 'Tan_IVP_vs_FD'

nsteps = 8
method = "Gauss"
rk = choreo.scipy_plus.multiprec_tables.ComputeImplicitRKTable_Gauss(nsteps, method=method)

ndim = 2

A = np.random.random((ndim,ndim))

def fun(t,x):
    
    return np.dot(A,x)*np.dot(x,x)

def gradfun(t,x,dx):

    dxr = np.asarray(dx).reshape(-1)
    resr = 2*np.dot(x,dxr)*np.dot(A,x) + np.dot(x,x)*np.dot(A,dxr)
    return resr.reshape(-1,1)



to = random.random()
fun_inst = functools.partial(fun,to)
gradfun_inst = functools.partial(gradfun,to)

xo = np.random.rand(ndim)
dxo = np.random.rand(ndim)

orderlist = [1,2]
epslist = [10**(-i) for i in range(16)]

norder = len(orderlist)
neps = len(epslist)

dpi = 150
figsize = (1600/dpi, 800 / dpi)

fig, ax = plt.subplots(
    figsize = figsize,
    dpi = dpi   ,
)

for i, order in enumerate(orderlist):

    err = choreo.scipy_plus.test.compare_FD_and_exact_grad(fun_inst,gradfun_inst,xo,dx=dxo,epslist=epslist,order=order)

    plt.plot(epslist,err)

ax.set_xscale('log')
ax.set_yscale('log')

plt.tight_layout()

# plt.show()

# %%

# plt.close()

nint = 100

t_span = (0.,0.1)

def int_fun(x):
    
    xx = x[:ndim].copy()
    vx = x[ndim:].copy()
    
    resx, resv = choreo.scipy_plus.ODE.ImplicitSymplecticIVP(
        fun         ,
        fun         ,
        t_span      ,
        xx          ,
        vx          ,
        rk          ,
        rk          ,
        nint = nint ,
    )

    return np.ascontiguousarray(np.concatenate((resx,resv)).reshape(-1))

def int_grad_fun(x,dx):
    
    xx = x[:ndim].copy()
    vx = x[ndim:].copy()
    
    dxx = dx[:ndim].reshape(ndim,-1).copy()
    dxv = dx[ndim:].reshape(ndim,-1).copy()
    
    resx, resv, grad_resx, grad_resv = choreo.scipy_plus.ODE.ImplicitSymplecticIVP(
        fun         ,
        fun         ,
        t_span      ,
        xx          ,
        vx          ,
        rk          ,
        rk          ,
        grad_fun = gradfun  ,
        grad_gun = gradfun  ,
        grad_x0 = dxx       ,
        grad_v0 = dxv       ,
        nint = nint         ,
    )
    
    return np.ascontiguousarray(np.concatenate((grad_resx,grad_resv)).reshape(-1))

xo = np.random.rand(2*ndim)
dxo = np.random.rand(2*ndim)

fig, ax = plt.subplots(
    figsize = figsize,
    dpi = dpi   ,
)

for i, order in enumerate(orderlist):

    err = choreo.scipy_plus.test.compare_FD_and_exact_grad(int_fun,int_grad_fun,xo,dx=dxo,epslist=epslist,order=order)

    plt.plot(epslist,err)

ax.set_xscale('log')
ax.set_yscale('log')

plt.tight_layout()

plt.show()
