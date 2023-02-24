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


def f_base(xj):

    # return xj*xj/2
    return np.cos(xj)

def f_base_grad(xj):

    # return xj
    return -np.sin( xj )


def int_fun(f_base,x):

    nint = x.shape[0]
    f_int = 0.

    for i in range(nint):

        f_int += f_base(x[i])

    return f_int / nint

def int_fun_grad(f_base_grad,x):

    nint = x.shape[0]

    f_all = np.zeros((nint))

    for i in range(nint):

        f_all[i] = f_base_grad(x[i])

    return f_all / nint

def int_fun_grad_c(f_base_grad,x):

    f_all = int_fun_grad(f_base_grad,x)
    # f_all_fft = np.fft.ihfft(f_all)
    f_all_fft = np.fft.rfft(f_all,norm="forward")
    # f_all_fft = np.fft.rfft(f_all) / nint

    ncoeffs = f_all_fft.shape[0]

    f_all_fft_real = np.zeros((ncoeffs-1,2))

    f_all_fft_real[0,0] = f_all_fft[0].real
    for k in range(1,ncoeffs-1):
        f_all_fft_real[k,0] = 2*f_all_fft[k].real
        f_all_fft_real[k,1] = 2*f_all_fft[k].imag

    return f_all_fft_real





# ncoeffs = 12
nint = 900

eps_list = [ 10**(-i) for i in range(10) ]
# eps_list = [ 10**(-i) for i in [5] ]


x = np.random.random((nint))
# x = np.zeros((nint))
# x[0] = 1.

ncoeffs = nint //2 +1


dc_init = np.zeros((ncoeffs,2))
dc_init[:,0] = np.random.random((ncoeffs))
dc_init[:,1] = np.random.random((ncoeffs))
dc_init[ncoeffs-1,:] = 0
dc_init_c = dc_init.view(dtype=np.complex128)[...,0]



dx = np.fft.irfft(dc_init_c)
# dx = np.random.random((nint))
# dx = np.zeros((nint))
# dx[1] = 1.


dc = np.fft.rfft(dx)
ncoeffs = dc.shape[0]

assert ncoeffs == nint//2 + 1

dc_real = np.zeros((ncoeffs-1,2))
# dc_real[:,0] = dc.real
# dc_real[:,1] = dc.imag

dc_real[0,0] = dc[0].real
for k in range(1,ncoeffs-1):
    dc_real[k,0] = dc[k].real
    dc_real[k,1] = dc[k].imag

dx_inv = np.fft.irfft(dc)
print(np.linalg.norm(dx-dx_inv))


fx = int_fun(f_base,x )
f_grad = int_fun_grad(f_base_grad,x)
# df_grad = np.tensordot(dx,f_grad)
df_grad = np.dot(dx.reshape(-1),f_grad.reshape(-1))

print("Comparison between finite differences and real gradient")
print('')

for eps in eps_list:

    xp = np.copy(x)+eps*dx
    xm = np.copy(x)-eps*dx
    
    fp = int_fun(f_base,xp)
    fm = int_fun(f_base,xm)

    df_difffin = (fp-fm)/(2*eps)

    abs_err = np.linalg.norm(df_grad-df_difffin)
    # rel_err = 2*abs_err/(np.linalg.norm(df_grad)+np.linalg.norm(df_difffin))

    print(f'eps : {eps}')
    print(f'Abs error : {abs_err}')
    # print(f'Rel error : {rel_err}')
    print('')


print("Comparison between real and complex gradients")
print('')



f_grad_c = int_fun_grad_c(f_base_grad,x)
df_grad_c = np.dot(dc_real.reshape(-1),f_grad_c.reshape(-1))


# print(f_grad_c)
# print(dc_real)


# print("f_grad : ",f_grad.sum(axis=1))
# print("f_grad_c :",f_grad_c*nint)
# print("mean val : ",f_grad.sum() - f_grad_c[0,0]*nint)
# print("f_grad_c :",f_grad_c*nint)

# print(df_grad)
print(df_grad)
print(df_grad_c)


abs_err = np.linalg.norm(df_grad-df_grad_c)
# rel_err = abs_err/(np.linalg.norm(df_grad)+np.linalg.norm(df_grad_c))

print('')
print(f'Abs error : {abs_err}')
# print(f'Rel error : {rel_err}')