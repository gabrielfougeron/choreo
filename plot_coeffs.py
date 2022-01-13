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

# ~ all_coeffs = np.load('./Target_res/3/4.npy')
all_coeffs = np.load('./Target_res/9/2.npy')
# ~ all_coeffs = np.load('./Target_res/6/2.npy')
# ~ all_coeffs = np.load('./Target_res/6/3.npy')


# ~ all_coeffs = np.load('./Sniff_all_sym/5/9.npy')
# ~ all_coeffs = np.load('./save_tests/9/8.npy')
# ~ all_coeffs = np.load('./save_tests/9/8.npy')
# ~ all_coeffs = np.load('./Sniff_all_sym/1/1.npy')
# ~ all_coeffs = np.load('./init.npy')


k_plot_max = 20


nloop = all_coeffs.shape[0]
# ~ ndim1 = all_coeffs1.shape[1]
ncoeff = all_coeffs.shape[2]

# ~ ncoeff_cap = 4800
ncoeff_cap = 1e10

ncoeff_plot = min(ncoeff,ncoeff_cap)


# ~ eps = 1e-18
eps = 0.

ampl = np.zeros((nloop,ncoeff_plot),dtype=np.float64)

for il in range(nloop):
    for k in range(ncoeff_plot):
        ampl[il,k] = np.linalg.norm(all_coeffs[il,:,k,:]) + eps

for il in range(nloop):
    for k in range(ncoeff_plot):
        if (k<k_plot_max):
            print(k,ampl[il,k])
        
        
        

ind = np.arange(ncoeff_plot) 


for il in range(nloop):
    
    fig = plt.figure()
    fig.set_size_inches(16, 12)
    ax = fig.add_subplot(111)

    ax.bar(ind,ampl[il,:])
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    filename = './bar_'+str(il)+'.png'
    
    plt.savefig(filename)
    
    plt.close()
