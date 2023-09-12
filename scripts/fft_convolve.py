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
import random
import scipy
import time

        
print()        
print("="*80)
print()

# Convolution of complex arrays

nint = 2* 10
# ncoeffs = nint // 2 + 1
ncoeffs = nint


f = np.zeros(ncoeffs,dtype = np.complex128)

for i in range(ncoeffs):
    f[i] = random.random() + 1j*random.random()

g = np.zeros(ncoeffs,dtype = np.complex128)
for i in range(ncoeffs):
    g[i] = random.random() + 1j*random.random()
    
fg = f*g

ifft_f  = scipy.fft.ifft(f)
ifft_g  = scipy.fft.ifft(g)
ifft_fg = scipy.fft.ifft(fg)

convolve_ifft_fg = np.zeros(nint,dtype = np.complex128)

for iint in range(nint):
    for jint in range(nint):
        
        
        ijint = (iint - jint + nint) % nint
        
        convolve_ifft_fg[iint] += ifft_f[jint] * ifft_g[ijint]
        
print(np.linalg.norm(ifft_fg - convolve_ifft_fg))
        
print()        
print("="*80)
print()

# Convolution of real arrays
 
nint = 2* 10
ncoeffs = nint // 2 + 1


f = np.zeros(ncoeffs,dtype = np.complex128)
 
for i in range(ncoeffs):
    f[i] = random.random() + 1j*random.random()
    
f[0] = f[0].real
f[ncoeffs-1] = f[ncoeffs-1].real

g = np.zeros(ncoeffs,dtype = np.complex128)
for i in range(ncoeffs):
    g[i] = random.random() + 1j*random.random()
    
g[0] = g[0].real
g[ncoeffs-1] = g[ncoeffs-1].real  
    
fg = f*g

ifft_f  = scipy.fft.irfft(f)
ifft_g  = scipy.fft.irfft(g)
ifft_fg = scipy.fft.irfft(fg)

convolve_ifft_fg = np.zeros(nint,dtype = np.float64)

for iint in range(nint):
    for jint in range(nint):
        
        ijint = (iint - jint + nint) % nint
        
        convolve_ifft_fg[iint] += ifft_f[jint] * ifft_g[ijint]
                
print(np.linalg.norm(ifft_fg - convolve_ifft_fg))
        
        
print()        
print("="*80)
print()

# Convolution of real matrix arrays
 
nint = 2* 10
ncoeffs = nint // 2 + 1
pdim = 5
qdim = 3


f = np.random.random((pdim, qdim, ncoeffs)) + 1j * np.random.random((pdim, qdim, ncoeffs))
f[:,:,0] = f[:,:,0].real
f[:,:,ncoeffs-1] = f[:,:,ncoeffs-1].real

g = np.random.random((qdim, ncoeffs)) + 1j * np.random.random((qdim, ncoeffs))
g[:,0] = g[:,0].real
g[:,ncoeffs-1] = g[:,ncoeffs-1].real

fg = np.zeros((pdim, ncoeffs), dtype=np.complex128)
for k in range(ncoeffs):
    fg[:,k] = np.matmul(f[:,:,k],g[:,k])
    

ifft_f  = scipy.fft.irfft(f,axis=2)
ifft_g  = scipy.fft.irfft(g,axis=1)
ifft_fg = scipy.fft.irfft(fg,axis=1)

convolve_ifft_fg = np.zeros((pdim,nint),dtype = np.float64)

for iint in range(nint):
    for jint in range(nint):
        
        ijint = (iint - jint + nint) % nint
        
        convolve_ifft_fg[:,iint] += np.matmul(ifft_f[:,:,jint],ifft_g[:,ijint])
                
print(np.linalg.norm(ifft_fg - convolve_ifft_fg))
        
print()        
print("="*80)
print()

# Convolution of real matrix arrays with subperiod
#  
# nint = 2* 16
# # ncoeffs = nint // 2 + 1
# ncoeffs = nint
# pdim = 5
# 
# 
# 
# nsub = 2*4
# assert nint % nsub == 0
# # ncoeffssub = nsub // 2 + 1
# ncoeffssub = nsub
# 
# ndiv = nint // nsub
# # ncoeffsdiv = ndiv // 2 + 1
# ncoeffsdiv = ndiv
# 
# 
# f = np.random.random((pdim,  ncoeffssub)) + 1j * np.random.random((pdim,  ncoeffssub))
# # f[:,0] = f[:,0].real
# # f[:,ncoeffssub-1] = f[:,ncoeffssub-1].real
# 
# # g = np.random.random((ncoeffssub, ncoeffsdiv)) + 1j * np.random.random((ncoeffssub, ncoeffsdiv))
# g_ = np.random.random((ncoeffs)) + 1j * np.random.random((ncoeffs))
# # g_[0] = g_[0].real
# # g_[ncoeffs-1] = g_[ncoeffs-1].real
# 
# fg = f * g_
# 
# g = np.zeros((ncoeffssub,ncoeffsdiv), dtype=np.complex128)
# 
# for k in range(ncoeffs):
#     l = k % ncoeffssub
#     q = k % ncoeffsdiv
#     g[l,q] = g_[k]
# 
# # # 
# # fg = np.zeros((pdim, ncoeffs), dtype=np.complex128)
# # for k in range(ncoeffs):
# #     # l = k % ncoeffssub
# #     # q = k % ncoeffsdiv
# #     # fg[:,k] = f[:,l]*g[l,q]
# #     
# #     l = k % ncoeffssub
# #     q = (k-l) // ncoeffssub
# #     fg[:,k] = f[:,l]*g[l,q]
# # #     
# 
# # ifft_f  = scipy.fft.irfft(f,axis=1)
# # ifft_g  = scipy.fft.irfft(g,axis=1)
# # ifft_fg = scipy.fft.irfft(fg,axis=1)
# 
# ifft_f  = scipy.fft.ifft(f,axis=1)
# ifft_g  = scipy.fft.ifft(g,axis=1)
# ifft_fg = scipy.fft.ifft(fg,axis=1)
# 
# convolve_ifft_fg = np.zeros((pdim,nint),dtype = np.complex128)
# 
# for iint in range(nint):
#     for jint in range(nint):
#         
#         jjint = (iint - jint + nint) % nint
#         
#         lint = jint % ifft_f.shape[1]
#         
#         llint = jjint % ifft_g.shape[0]
#         qqint = (jjint-llint) // ifft_g.shape[0]
# 
#         convolve_ifft_fg[:,iint] += ifft_f[:,lint] * ifft_g[llint,qqint]
#                 
# # print(np.linalg.norm(ifft_fg - convolve_ifft_fg))
# # print(ifft_fg /convolve_ifft_fg)
# print(ifft_fg - convolve_ifft_fg)


