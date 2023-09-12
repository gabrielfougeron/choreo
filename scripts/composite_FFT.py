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

np.set_printoptions(edgeitems=10,linewidth=200)

        
print()        
print("="*80)
print()

# FFT of complex arrays

pint = 2
qint = 3
nint = pint * qint

f = np.random.random((nint)) + 1j * np.random.random((nint))

fft_f = scipy.fft.fft(f)

rf = f.copy().reshape(qint, pint) 

rf_1 = scipy.fft.fft(rf, axis=0)


for iq in range(qint):
    for ip in range(pint):        
        
        w = np.exp((-2j*m.pi*ip*iq)/nint)
        
        rf_1[iq, ip] *= w
        
# CAREFUL ! TRANSPOSE HERE !        
fft_rf = scipy.fft.fft(rf_1, axis=1).T.reshape(-1)

print(np.linalg.norm(fft_f - fft_rf))

print()        
print("="*80)
print()

# IDEM IFFT

ifft_f = scipy.fft.ifft(f)

rf = f.copy().reshape(qint, pint) 

rf_1 = scipy.fft.ifft(rf, axis=0)


for iq in range(qint):
    for ip in range(pint):        
        
        w = np.exp((2j*m.pi*ip*iq)/nint)
        
        rf_1[iq, ip] *= w
        
# CAREFUL ! TRANSPOSE HERE !        
ifft_rf = scipy.fft.ifft(rf_1, axis=1).T.reshape(-1)

print(np.linalg.norm(ifft_f - ifft_rf))



print()        
print("="*80)
print()

# FFT of real arrays


qint = 2 * 3
pint = 3
nint = pint * qint

ncoeff = nint // 2 + 1
qcoeff = qint // 2 + 1

f = np.random.random((nint))

rfft_f = scipy.fft.rfft(f)

rf = f.copy().reshape(qint, pint) 

rf_1 = scipy.fft.rfft(rf, axis=0)

for iq in range(qcoeff):
    for ip in range(pint):        
        
        w = np.exp((-2j*m.pi*ip*iq)/nint)
        
        rf_1[iq, ip] *= w
        
 
rfft_rf_quad = scipy.fft.fft(rf_1, axis=1)

reorder = np.zeros((ncoeff),dtype=np.complex128)

# Weird reordering algorithm

j = 0
jq = 0
jp = 0
reorder[j] = rfft_rf_quad[jq,jp]
jqdir =  1
jpdir = -1
for ip in range(pint):   
    for iq in range(qcoeff-1):
        j = j + 1
        jq = jq + jqdir
        
        reorder[j] = rfft_rf_quad[jq,jp].real + 1j*jqdir*rfft_rf_quad[jq,jp].imag
        

    jqdir = - jqdir
    
    jp = (jp + jpdir + pint) % pint
    jpdir = - jpdir - jqdir
  
        
        
# print(rfft_f)
# print(rfft_rf_quad)
        
print(np.linalg.norm(rfft_f - reorder))


print()        
print("="*80)
print()


# print(rfft_rf_quad.shape)
# print(pint)
# print(qcoeff)
# 
# exit()
# IRFFT


# qint = 2 * 101
# pint = 37
# nint = pint * qint

ncoeff = nint // 2 + 1
qcoeff = qint // 2 + 1

f_sol = np.random.random((nint))
# f_sol = f.copy()

cf = scipy.fft.rfft(f_sol)

reorder = np.zeros((qcoeff,pint),dtype=np.complex128)

# Weird reordering algorithm

j = 0
jq = 0
jp = 0
reorder[jq,jp] = cf[j]
jqdir =  1
jpdir = -1
for ip in range(pint):   
    for iq in range(qcoeff-1):
        j = j + 1
        jq = jq + jqdir
        
        reorder[jq,jp] = cf[j].real + 1j*jqdir*cf[j].imag
        
    jqdir = - jqdir
    
    jp = (jp + jpdir + pint) % pint
    jpdir = - jpdir - jqdir
    
    reorder[jq,jp] = cf[j].real + 1j*jqdir*cf[j].imag


rfft_rf_quad = scipy.fft.ifft(reorder, axis=1)

for iq in range(qcoeff):
    for ip in range(pint):        
        
        w = np.exp((2j*m.pi*ip*iq)/nint)
        
        rfft_rf_quad[iq, ip] *= w
        
rfft_total = scipy.fft.irfft(rfft_rf_quad, axis=0).reshape(-1)
        

print(np.linalg.norm(f_sol - rfft_total))
