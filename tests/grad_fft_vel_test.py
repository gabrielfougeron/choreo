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

twopi = 2*np.pi


def f_base(xj,vj):

    # return np.cos(xj)
    return np.cos(xj) + np.cos(vj) 

def f_base_grad(xj,vj):

    # return np.array([- np.sin( xj ),0.])
    return np.array([- np.sin( xj ),- np.sin( vj )])


def int_fun(f_base,x,v):

    nint = x.shape[0]
    f_int = 0.

    for i in range(nint):

        f_int += f_base(x[i],v[i])

    return f_int / nint

def int_fun_grad(f_base_grad,x,v):

    nint = x.shape[0]

    f_all = np.zeros((nint,2))

    for i in range(nint):

        f_all[i,:] = f_base_grad(x[i],v[i])

    return f_all / nint

def int_fun_grad_c(f_base_grad,x,v):

    f_all = int_fun_grad(f_base_grad,x,v)
    f_all_fft = np.fft.rfft(f_all,axis=0,norm="forward")

    ncoeffs = f_all_fft.shape[0]

    f_all_fft_real = np.zeros((ncoeffs-1,2))

    f_all_fft_real[0,0] = f_all_fft[0,0].real
    for k in range(1,ncoeffs-1):
        f_all_fft_real[k,0] = 2*(f_all_fft[k,0].real + twopi*k*f_all_fft[k,1].imag)
        f_all_fft_real[k,1] = 2*(f_all_fft[k,0].imag - twopi*k*f_all_fft[k,1].real)

    return f_all_fft_real





ncoeffs = 12
nint=ncoeffs*2
# nint = 900



eps_list = [ 10**(-i) for i in range(10) ]
# eps_list = [ 10**(-i) for i in [5] ]

c_init = np.zeros((ncoeffs,2))
c_init[:,0] = np.random.random((ncoeffs))
c_init[:,1] = np.random.random((ncoeffs))
c_init[ncoeffs-1,:] = 0
c_init_c = c_init.view(dtype=np.complex128)[...,0]
cv_init = c_init.copy()
cv_init_c = cv_init.view(dtype=np.complex128)[...,0]
for k in range(ncoeffs):
    cv_init_c[k] *= twopi * 1J * k


x = np.fft.irfft(c_init_c,n=nint)
v = np.fft.irfft(cv_init_c,n=nint)





print(ncoeffs,nint)



# x = np.random.random((nint))
# x = np.zeros((nint))
# x[0] = 1.

# ncoeffs = nint //2 +1


dc = np.zeros((ncoeffs,2))
dc[:,0] = np.random.random((ncoeffs))
dc[:,1] = np.random.random((ncoeffs))
dc[ncoeffs-1,:] = 0
dc_c = dc.view(dtype=np.complex128)[...,0]
dcv_c = dc_c.copy()
for k in range(ncoeffs):
    dcv_c[k] *= twopi * 1J * k


dx = np.fft.irfft(dc_c,n=nint)
dv = np.fft.irfft(dcv_c,n=nint)

dc_real = np.zeros((ncoeffs,2))
dc_real[:,0] = dc_c.real
dc_real[:,1] = dc_c.imag

dx_inv = np.fft.irfft(dc_c,n=nint)
print(np.linalg.norm(dx-dx_inv))


fx = int_fun(f_base,x,v)
f_grad = int_fun_grad(f_base_grad,x,v)
# df_grad = np.tensordot(dx,f_grad)

df_grad = np.dot(dx,f_grad[:,0]) + np.dot(dv,f_grad[:,1])


print("Comparison between finite differences and real gradient")
print('')

for eps in eps_list:

    xp = np.copy(x)+eps*dx
    xm = np.copy(x)-eps*dx
    vp = np.copy(v)+eps*dv
    vm = np.copy(v)-eps*dv
    
    fp = int_fun(f_base,xp,vp)
    fm = int_fun(f_base,xm,vm)

    df_difffin = (fp-fm)/(2*eps)

    abs_err = np.linalg.norm(df_grad-df_difffin)
    # rel_err = 2*abs_err/(np.linalg.norm(df_grad)+np.linalg.norm(df_difffin))

    print(f'eps : {eps}')
    print(f'Abs error : {abs_err}')
    # print(f'Rel error : {rel_err}')
    print('')


print("Comparison between real and complex gradients")
print('')


f_grad_c = int_fun_grad_c(f_base_grad,x,v)
df_grad_c = np.dot(dc_real.reshape(-1),f_grad_c.reshape(-1))


# print(f_grad_c)
# print(dc_real)


# print("f_grad : ",f_grad.sum(axis=1))
# print("f_grad_c :",f_grad_c*nint)
# print("mean val : ",f_grad.sum() - f_grad_c[0,0]*nint)
# print("f_grad_c :",f_grad_c*nint)

# print(df_grad)
# print(df_grad)
# print(df_grad_c)


abs_err = np.linalg.norm(df_grad-df_grad_c)
# rel_err = abs_err/(np.linalg.norm(df_grad)+np.linalg.norm(df_grad_c))

print('')
print(f'Abs error : {abs_err}')
# print(f'Rel error : {rel_err}')

