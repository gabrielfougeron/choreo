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

