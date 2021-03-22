import numpy as np
import math as m
import scipy.optimize as opt
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import cnames
from matplotlib import animation
import copy
import os, shutil
import time


from Choreo_funs import *

# ~ all_coeffs = np.load('./Sniff_all/1x1x1/3.npy')
all_coeffs = np.load('./Sniff_all/4/1.npy')



nloop = all_coeffs.shape[0]
ncoeff = all_coeffs.shape[2]
nbody = np.array([12])
mass = np.ones((nloop))
nint = 2*ncoeff
n_idx,all_idx =  setup_idx(nloop,nbody,ncoeff)

Action,GradAction = Compute_action_timings(nloop,nbody,ncoeff,mass,nint,all_coeffs,n_idx,all_idx)
    
    
    
    
    
    
