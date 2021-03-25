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


nbody = np.array([8])
# ~ nbody = np.array([4])
# ~ nbody = np.array([1,1,1,1,1])
# ~ nbody = np.array([3,3,3,3,3])


nloop=nbody.size

store_folder = './Sniff_all/'
store_folder = store_folder+str(nbody[0])
for i in range(len(nbody)-1):
    store_folder = store_folder+'x'+str(nbody[i+1])

for file_path in os.listdir(store_folder):
    file_path = os.path.join(store_folder, file_path)
    file_root, file_ext = os.path.splitext(os.path.basename(file_path))
    
    if (file_ext == '.mp4' ):
        
        if ('_no_trace' not in file_root):
            
            new_vid_filename = os.path.join(store_folder, file_root+'_no_trace.mp4')
            
            if not(os.path.isfile(new_vid_filename)):
                
                all_coeffs_filename = os.path.join(store_folder, file_root+'.npy')
                
                all_coeffs = np.load(all_coeffs_filename)
                
                print('Saving no trace video in '+new_vid_filename)
                nint_plot = 4000
                nperiod = 1/2
                plot_all_2D_anim(nloop,nbody,nint_plot,nperiod,all_coeffs,new_vid_filename,Plot_trace=False)
                
