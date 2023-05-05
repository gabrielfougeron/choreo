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
import functools

__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(__PROJECT_ROOT__)

import choreo 

# Cluster_plot = True
Cluster_plot = False

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
n_colors = len(colors)

store_folder = os.path.join(__PROJECT_ROOT__,'Sniff_all_sym','NumericalTank_tests')
file_basename = 'NumericalTank_00004_init'
file_basename = 'NumericalTank_00004'
all_pos_filename = os.path.join(store_folder,file_basename+'.npy')



all_pos = np.load(all_pos_filename)
c_coeffs = choreo.the_rfft(all_pos,axis=2,norm="forward") 

k_plot_max = 20

nloop = c_coeffs.shape[0]
ncoeff = c_coeffs.shape[2]

# ncoeff_cap = 4800
ncoeff_cap = 1e10

ncoeff_plot = min(ncoeff,ncoeff_cap)
ind = np.arange(ncoeff_plot) 


eps = 1e-18
# eps = 0.

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

if Cluster_plot:

    log_max_ampl_T = np.log(max_ampl).T

    degpolyfit = 1
    OneRangeCost = functools.partial(choreo.PolyCost,degpolyfit=degpolyfit)
    cuts,costs,hmin,hmax,hcmin,hcmax = choreo.FindFuses(log_max_ampl_T,OneRangeCost=OneRangeCost)

    n_clusters = 2

    color_clusters,depth_color_cluster = choreo.ColorClusters(log_max_ampl_T,ind,n_clusters,cuts)

    for i_clus in range(n_clusters):
        
        imin = color_clusters[i_clus]
        imax = color_clusters[i_clus+1]+1

        for il in range(nloop):
        
            ax.plot(ind[imin:imax],max_ampl[il,imin:imax],c=colors[i_clus%n_colors])

            z = np.polyfit(ind[imin:imax], log_max_ampl_T[imin:imax,il], degpolyfit, rcond=None, full=False)
            p = np.poly1d(z)
            
            ax.plot(ind[imin:imax],np.exp(p(ind[imin:imax])),c='k')

else:

    for il in range(nloop):
    
        ax.plot(ind,max_ampl[il,:],c=colors[il%n_colors])



ax.set_yscale('log')
plt.tight_layout()

filename = os.path.join(store_folder,file_basename+'_bar_max.png')
plt.savefig(filename)
plt.close()
