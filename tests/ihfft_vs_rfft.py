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


nint = 2000
ncoeff = nint // 2 + 1

x = np.random.random((nint))
c = np.fft.rfft(x)
x_inv = np.fft.irfft(c,n=nint)
print(np.linalg.norm(x-x_inv))


y = np.fft.irfft(c)*nint
z = np.fft.irfft(c,norm="forward")

print(np.linalg.norm(y-z))


ch = np.conj(np.fft.ihfft(x)*nint)

# print(c-ch)

print(np.linalg.norm(c-ch))




c = np.fft.ihfft(x)
ch = np.conj(np.fft.rfft(x)/nint)

# print(c-ch)

print(np.linalg.norm(c-ch))


c = np.fft.ihfft(x)
d = np.fft.ihfft(x,nint)

print(np.linalg.norm(c-d))




c = np.fft.ihfft(x)
ch = np.conj(np.fft.rfft(x,norm="forward"))
# ch = np.conj(np.fft.rfft(x,norm="backward"))

print(np.linalg.norm(c-ch))
