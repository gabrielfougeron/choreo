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

np.set_printoptions(
    precision = 3,
    edgeitems = 10,
    linewidth = 150,
    floatmode = "fixed",
    )

        
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
fft_rf = scipy.fft.fft(rf_1.T, axis=0).reshape(-1)

print(np.linalg.norm(fft_f - fft_rf))

print()        
print("="*80)
print()

# IDEM tranposed

pint = 2
qint = 3
nint = pint * qint

f = np.random.random((nint)) + 1j * np.random.random((nint))

fft_f = scipy.fft.fft(f)

rf = f.copy().reshape(qint, pint) 

# CAREFUL ! TRANSPOSE HERE !    
rf_1 = scipy.fft.fft(rf.T, axis=1)


for iq in range(qint):
    for ip in range(pint):        
        
        w = np.exp((-2j*m.pi*ip*iq)/nint)
        
        rf_1[ip, iq] *= w
    
fft_rf = scipy.fft.fft(rf_1, axis=0).reshape(-1)


print(np.linalg.norm(fft_f - fft_rf))


print()        
print("="*80)
print()


# IDEM with array of transposed shape (since pint and qint have similar roles)

pint = 2
qint = 3
nint = pint * qint

f = np.random.random((nint)) + 1j * np.random.random((nint))

fft_f = scipy.fft.fft(f)

rf = f.copy().reshape(pint,qint) 

# CAREFUL ! TRANSPOSE HERE !    
rf_1 = scipy.fft.fft(rf.T, axis=1)


for iq in range(qint):
    for ip in range(pint):        
        
        w = np.exp((-2j*m.pi*ip*iq)/nint)
        
        rf_1[iq, ip] *= w
    
fft_rf = scipy.fft.fft(rf_1, axis=0).reshape(-1)


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


print()        
print("="*80)
print()


# Now for the general setup, with complex numbers:


ncoeffs_min = 2
nparam_per_period = 3
nperiods = 3
nint = ncoeffs_min * nperiods

All_params_basis = np.random.random((ncoeffs_min, nparam_per_period)) + 1j * np.random.random((ncoeffs_min, nparam_per_period))
all_params = np.random.random((nparam_per_period, nperiods)) + 1j * np.random.random((nparam_per_period, nperiods))

all_coeffs = np.dot(All_params_basis, all_params).reshape(-1)

all_pos_direct = scipy.fft.ifft(all_coeffs)



# Making the tables artificially longer before FFT + Naive convolution

All_params_basis_long = np.zeros((nint,nparam_per_period),dtype=np.complex128)
all_params_long = np.zeros((nparam_per_period,nint),dtype=np.complex128)

for iint in range(nint):

    All_params_basis_long[iint,: ] = All_params_basis[iint // nperiods,:]
        
for iint in range(nint):
        
    all_params_long[:, iint] = all_params[:, iint % nperiods]  
    
ifft_f_long = scipy.fft.ifft(All_params_basis_long, axis=0)
ifft_g_long = scipy.fft.ifft(all_params_long, axis=1) # This is sparse with sparsity pattern one in ncoeffs_min


convolve_ifft_fg = np.zeros((nint),dtype = np.complex128)

for iint in range(nint):
    for jint in range(nint):
        
        ijint = (iint - jint + nint) % nint
        
        convolve_ifft_fg[iint] += np.dot(ifft_f_long[jint,:], ifft_g_long[:,ijint])
        
print(np.linalg.norm(all_pos_direct - convolve_ifft_fg))
print()         



# Finding the right tables


ifft_f  = scipy.fft.ifft(All_params_basis, axis=0)
ifft_g  = scipy.fft.ifft(all_params, axis=1)


ifft_f_long_bis = np.zeros((nint,nparam_per_period),dtype=np.complex128)
ifft_g_long_bis = np.zeros((nparam_per_period,nint),dtype=np.complex128) 


for ip in range(nperiods):
    for iq in range(ncoeffs_min):
    
        jint = iq + ncoeffs_min*ip

        if jint == 0:
            mul = 1.
        else:

            mul = 1./nperiods * (1-np.exp(2j*m.pi*iq/ncoeffs_min)) / (1-np.exp(2j*m.pi*jint/nint))
            
        ifft_f_long_bis[jint,:] = ifft_f[iq,:] * mul


print(np.linalg.norm(ifft_f_long - ifft_f_long_bis))

print()

for ip in range(nperiods):
    ifft_g_long_bis[:,ip * ncoeffs_min] = ifft_g[:,ip]

print(np.linalg.norm(ifft_g_long - ifft_g_long_bis))

# Reduced convolution version I

convolve_ifft_fg = np.zeros((nint),dtype = np.complex128)

ifft_f_mod = ifft_f.copy()

for iq in range(ncoeffs_min):
    
    if iq == 0:
        mul = 1.
    else:        
        mul = (1-np.exp(2j*m.pi*iq / ncoeffs_min))

    ifft_f_mod[iq,:] *= mul


for ip in range(nperiods):
    
    for jpp in range(ip,ip+nperiods):
            
        for iq in range(ncoeffs_min):
                    
            jp = jpp-ip
            
            iint = ((jpp)*ncoeffs_min + iq) % nint
            
            jint = iq + ncoeffs_min * (jpp-ip)
            
            if iq == 0:
                if jp == 0:
                    mul = 1.
                else:
                    mul = 0.
            else:
                mul = 1. / (nperiods * (1-np.exp(2j*m.pi*jint / nint)))


            convolve_ifft_fg[iint] += np.dot(ifft_f_mod[iq,:], ifft_g[:,ip]) * mul
        

        
print()        
print(np.linalg.norm(all_pos_direct - convolve_ifft_fg))
#         

# Reduced convolution version II

convolve_ifft_fg = np.zeros((nint),dtype = np.complex128)

for ip in range(nperiods):
    for jint in range(nint):
        
        ijint = (ip * ncoeffs_min + jint) % nint

        convolve_ifft_fg[ijint] += np.dot(ifft_f_long[jint,:], ifft_g[:,ip])
        
        
print()        
print(np.linalg.norm(all_pos_direct - convolve_ifft_fg))



print()        
print("="*80)
print()


# Without convolution



ncoeffs_min = 2
nparam_per_period = 3
nperiods = 3
nint = ncoeffs_min * nperiods

All_params_basis = np.random.random((ncoeffs_min, nparam_per_period)) + 1j * np.random.random((ncoeffs_min, nparam_per_period))
all_params = np.random.random((nparam_per_period, nperiods)) + 1j * np.random.random((nparam_per_period, nperiods))

all_coeffs = np.dot(All_params_basis, all_params).reshape(-1)

all_pos_direct = scipy.fft.ifft(all_coeffs)


ifft_f  = scipy.fft.ifft(All_params_basis, axis=0)

inter_array = np.zeros((ncoeffs_min,nperiods),dtype=np.complex128)

for iq in range(ncoeffs_min):
    
    for ip in range(nperiods):    
        
        w = np.exp((2j*m.pi*iq*ip)/nint)    
        
        inter_array[iq, ip] = w * np.matmul(ifft_f[iq,:], all_params[:,ip])
        

ifft_fg  = scipy.fft.ifft(inter_array, axis=1).T.reshape(-1)


print(np.linalg.norm(all_pos_direct - ifft_fg))


print()        
print("="*80)
print()


ncoeffs_min = 5
nparam_per_period = 15
nperiods = 11
nint = ncoeffs_min * nperiods

All_params_basis = np.random.random((ncoeffs_min, nparam_per_period)) + 1j * np.random.random((ncoeffs_min, nparam_per_period))
all_params = np.random.random((nparam_per_period, nperiods)) + 1j * np.random.random((nparam_per_period, nperiods))

all_coeffs = np.dot(All_params_basis, all_params).T.reshape(-1)

all_pos_direct = scipy.fft.ifft(all_coeffs)

ifft_g  = scipy.fft.ifft(all_params, axis=1)

inter_array = np.zeros((ncoeffs_min,nperiods),dtype=np.complex128)

for iq in range(ncoeffs_min):
    
    for ip in range(nperiods):        
        
        w = np.exp((2j*m.pi*iq*ip)/nint)
        
        inter_array[iq, ip] = w * np.matmul(All_params_basis[iq,:], ifft_g[:,ip])

ifft_fg  = scipy.fft.ifft(inter_array, axis=0).reshape(-1)
       
print(np.linalg.norm(all_pos_direct - ifft_fg))


print()        
print("="*80)
print()

# Without convolution, ON A SUBDOMAIN !!!!

ncoeffs_min = 2
nparam_per_period = 15
nperiods = 3
nint = ncoeffs_min * nperiods

All_params_basis = np.random.random((ncoeffs_min, nparam_per_period)) + 1j * np.random.random((ncoeffs_min, nparam_per_period))
all_params = np.random.random((nparam_per_period, nperiods)) + 1j * np.random.random((nparam_per_period, nperiods))

all_coeffs = np.dot(All_params_basis, all_params).T.reshape(-1)

all_pos_direct = scipy.fft.ifft(all_coeffs)

ifft_g  = scipy.fft.ifft(all_params, axis=1)

inter_array = np.zeros((ncoeffs_min,nperiods),dtype=np.complex128)

for iq in range(ncoeffs_min):
    
    for ip in range(nperiods):        
        
        w = np.exp((2j*m.pi*iq*ip)/nint)
        
        inter_array[iq, ip] = w * np.matmul(All_params_basis[iq,:], ifft_g[:,ip])

sum_fg = np.sum(inter_array, axis=0) / ncoeffs_min

ifft_fg  = scipy.fft.ifft(inter_array, axis=0).reshape(-1)
       
print(np.linalg.norm(all_pos_direct - ifft_fg))
print(np.linalg.norm(all_pos_direct[:nperiods] - sum_fg))

print()        
print("="*80)
print()

# Without convolution, on a subdomain, NO GLOBAL ARRAYS!!!!

ncoeffs_min = 3
nparam_per_period = 15
nperiods = 37
nint = ncoeffs_min * nperiods

All_params_basis = np.random.random((ncoeffs_min, nparam_per_period)) + 1j * np.random.random((ncoeffs_min, nparam_per_period))
all_params = np.random.random((nparam_per_period, nperiods)) + 1j * np.random.random((nparam_per_period, nperiods))

all_coeffs = np.dot(All_params_basis, all_params).T.reshape(-1)
all_pos_direct = scipy.fft.ifft(all_coeffs)

ifft_g  = scipy.fft.ifft(all_params, axis=1)

inter_array = np.zeros((nperiods),dtype=np.complex128)

ncoeffs_min_inv = 1.  / ncoeffs_min

for iq in range(ncoeffs_min):
    
    wo = np.exp((2j*m.pi*iq)/nint)
    
    w = ncoeffs_min_inv
    
    for ip in range(nperiods):        
        
        inter_array[ip] += w * np.matmul(All_params_basis[iq,:], ifft_g[:,ip])
        
        w *= wo

print(np.linalg.norm(all_pos_direct[:nperiods] - inter_array))

# IS THIS REALLY WHAT I WANT THOUGH ?
# DO THIS WITH ULTIMATELY REAL POSITIONS ?
# START WITH INVERSING THESE RELATIONS ? (<=> go from positions to parameters)


print()        
print("="*80)
print()

# IRFFT via ICFFT of double length

ncoeffs_min = 3
nparam_per_period = 15
nperiods = 5
nint = 2 * (ncoeffs_min * nperiods - 1)

All_params_basis = np.random.random((ncoeffs_min, nparam_per_period)) + 1j * np.random.random((ncoeffs_min, nparam_per_period))
all_params = np.random.random((nparam_per_period, nperiods))


""" This is probably actually not necessary
Making sure that the first and last coeffs are purely real. There might be a better way.
all_params[:,0] = 0.
all_params[:,-1] = 0.
 """
all_coeffs = np.dot(All_params_basis, all_params).T.reshape(-1)

all_pos_direct = scipy.fft.irfft(all_coeffs)

assert nint == all_pos_direct.shape[0]

all_coeffs_2 = np.zeros((nint), dtype=np.complex128)

all_coeffs_2[:all_coeffs.shape[0]] = 2*all_coeffs
all_coeffs_2[0] = all_coeffs[0]
all_coeffs_2[all_coeffs.shape[0]-1] = all_coeffs[all_coeffs.shape[0]-1]

all_pos_2 = scipy.fft.ifft(all_coeffs_2)
all_pos_2_real = all_pos_2.real.copy()

print(np.linalg.norm(all_pos_direct - all_pos_2_real))

all_params_3 = np.zeros((nparam_per_period, 2*nperiods))
all_params_3[:,:nperiods] = all_params[:,:]
all_coeffs_3 = np.dot(All_params_basis, all_params_3).T.reshape(-1)[0:-2].copy()
all_coeffs_3[1:all_coeffs.shape[0]-1] *= 2


print(np.linalg.norm(all_coeffs_2 - all_coeffs_3))


print()        
print("="*80)
print()

# IRFFT via ICFFT of double length with decomposable nint

ncoeffs_min = 3
nparam_per_period = 15
nperiods = 5
nint = 2 * (ncoeffs_min * nperiods)

All_params_basis = np.random.random((ncoeffs_min, nparam_per_period)) + 1j * np.random.random((ncoeffs_min, nparam_per_period))
all_params = np.random.random((nparam_per_period, nperiods))


# This is probably actually not necessary
# Making sure that the first and last coeffs are purely real. There might be a better way.
# all_params[:,0] = 0.
# all_params[:,-1] = 0.

 
 
all_coeffs = np.zeros((nint//2+1), dtype=np.complex128)
all_coeffs[:nint//2] = np.dot(All_params_basis, all_params).T.reshape(-1)

all_pos_direct = scipy.fft.irfft(all_coeffs)
assert nint == all_pos_direct.shape[0]

all_coeffs_2 = np.zeros((nint), dtype=np.complex128)

all_coeffs_2[:all_coeffs.shape[0]] = 2*all_coeffs
all_coeffs_2[0] = all_coeffs[0]
all_coeffs_2[all_coeffs.shape[0]-1] = all_coeffs[all_coeffs.shape[0]-1]

all_pos_2 = scipy.fft.ifft(all_coeffs_2)
all_pos_2_real = all_pos_2.real.copy()

print(np.linalg.norm(all_pos_direct - all_pos_2_real))

all_params_3 = np.zeros((nparam_per_period, 2*nperiods))
all_params_3[:,:nperiods] = all_params[:,:]
all_coeffs_3 = np.dot(All_params_basis, all_params_3).T.reshape(-1)
all_coeffs_3[1:all_coeffs.shape[0]-1] *= 2
print(np.linalg.norm(all_coeffs_2 - all_coeffs_3))






all_coeffs_4 = np.dot(All_params_basis, all_params_3).T.reshape(-1)
all_pos_4_direct = scipy.fft.ifft(all_coeffs_4)

# The average value is treated separately
meanval = -np.dot(All_params_basis[0,:],all_params_3[:,0]).real / nint
# All parameters are doubled here.
# The transform is a COMPLEX Inverse transform here
ifft_g  = scipy.fft.ifft(2*all_params_3, axis=1)

n_inter = nperiods
inter_array = np.full((n_inter), meanval)

ncoeffs_min_inv = 1.  / ncoeffs_min

for iq in range(ncoeffs_min):
    
    wo = np.exp((2j*m.pi*iq)/nint)
    
    w = ncoeffs_min_inv
    
    for ip in range(n_inter):        
        # Only the real part of the computation is needed here
        inter_array[ip] += (w * np.matmul(All_params_basis[iq,:], ifft_g[:,ip])).real
        
        w *= wo

# print(np.linalg.norm(all_pos_4_direct[:n_inter] - inter_array))
print(np.linalg.norm(all_pos_direct[:n_inter] - inter_array))
# print(all_pos_direct[:n_inter])
# print(inter_array)
# print(all_pos_direct[:n_inter] - inter_array)



print()        
print("="*80)
print()

# IRFFT via ICFFT of double length with decomposable nint and real transforms


ncoeffs_min = 7
nparam_per_period = 3
nperiods = 17
nint = 2 * (ncoeffs_min * nperiods)

All_params_basis = np.random.random((ncoeffs_min, nparam_per_period)) + 1j * np.random.random((ncoeffs_min, nparam_per_period))
all_params = np.random.random((nparam_per_period, nperiods))
 
all_coeffs = np.zeros((nint//2+1), dtype=np.complex128)
all_coeffs[:nint//2] = np.dot(All_params_basis, all_params).T.reshape(-1)

all_pos_direct = scipy.fft.irfft(all_coeffs)
assert nint == all_pos_direct.shape[0]



# The average value is treated separately
meanval = -np.dot(All_params_basis[0,:].real,all_params[:,0]) / nint
# All parameters are doubled here.
ifft_g  = scipy.fft.ihfft(all_params, axis=1, n=2*nperiods)

n_inter = nperiods + 1
inter_array = np.full((n_inter), meanval)

ncoeffs_min_inv = 2.  / ncoeffs_min

for iq in range(ncoeffs_min):
    
    wo = np.exp((2j*m.pi*iq)/nint)
    
    w = ncoeffs_min_inv
    
    for ip in range(n_inter):        
        # Only the real part of the computation is needed here
        inter_array[ip] += (w * np.matmul(All_params_basis[iq,:], ifft_g[:,ip])).real
        
        w *= wo

print(np.linalg.norm(all_pos_direct[:n_inter] - inter_array))





