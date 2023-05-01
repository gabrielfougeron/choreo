import numpy as np
import math as m
import scipy.optimize as opt
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import cnames
from matplotlib import animation
import copy
import os, shutil, sys
import time

__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(__PROJECT_ROOT__)

import choreo 


store_folder = os.path.join(__PROJECT_ROOT__,'Sniff_all_sym','NumericalTank_tests')
all_pos_filename = os.path.join(store_folder,'NumericalTank_00001.npy')


all_pos = np.load(all_pos_filename)
c_coeffs = choreo.the_rfft(all_pos,axis=2,norm="forward") 

k_plot_max = 20


nloop = c_coeffs.shape[0]
ncoeff = c_coeffs.shape[2]

# ncoeff_cap = 4800
ncoeff_cap = 1e10

ncoeff_plot = min(ncoeff,ncoeff_cap)
ind = np.arange(ncoeff_plot) 


# eps = 1e-18
eps = 0.

ampl = np.zeros((nloop,ncoeff_plot),dtype=np.float64)
max_ampl = np.zeros((nloop,ncoeff_plot),dtype=np.float64)

for il in range(nloop):
    for k in range(ncoeff_plot):
        ampl[il,k] = np.linalg.norm(c_coeffs[il,:,k]) + eps

# for il in range(nloop):
#     for k in range(ncoeff_plot):
#         if (k<k_plot_max):
#             print(k,ampl[il,k])
        

# 
# for il in range(nloop):
#     
#     fig = plt.figure()
#     fig.set_size_inches(16, 12)
#     ax = fig.add_subplot(111)
# 
#     ax.bar(ind,ampl[il,:])
#     ax.set_yscale('log')
#     
#     plt.tight_layout()
#     
#     filename = os.path.join(store_folder,'bar_'+str(il)+'.png')
#     
#     plt.savefig(filename)
#     
#     plt.close()

ncoeff_plotm1 = ncoeff_plot - 1

for il in range(nloop):
    cur_max = 0.
    for k in range(ncoeff_plot):
        k_inv = ncoeff_plotm1 - k

        cur_max = max(cur_max,ampl[il,k_inv])
        max_ampl[il,k_inv] = cur_max

fig = plt.figure()
fig.set_size_inches(16, 12)
ax = fig.add_subplot(111)
for il in range(nloop):
    
    ax.plot(ind,max_ampl[il,:])


ax.set_yscale('log')
plt.tight_layout()

filename = os.path.join(store_folder,'bar_max.png')
plt.savefig(filename)
plt.close()
