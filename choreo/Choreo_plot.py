'''
Choreo_funs.py : Defines I/O and plot specific functions in the Choreographies2 project.

'''

import os
import itertools
import copy
import time
import pickle

import numpy as np
import math as m
import scipy.fft
import scipy.optimize
import scipy.linalg as la
import scipy.sparse as sp
import sparseqr
import networkx as nx
import random

try:
    import ffmpeg
except:
    pass

import inspect

import fractions

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import cnames
from matplotlib.collections import LineCollection
from matplotlib import animation

from choreo.Choreo_cython_funs import ndim,twopi,nhash,n
from choreo.Choreo_cython_funs import the_irfft,the_rfft,the_ihfft

from choreo.Choreo_funs import Compute_action,Compute_hash_action,Compute_Newton_err
from choreo.Choreo_funs import Compute_MinDist,Detect_Escape
from choreo.Choreo_funs import Unpackage_all_coeffs
from choreo.Choreo_funs import ComputeAllPos
    
Action_msg = 'Value of the Action : '
Action_msg_len = len(Action_msg)

Action_Hash_msg = 'Hash Action for duplicate detection : '
Action_Hash_msg_len = len(Action_Hash_msg)

def plot_Newton_Error(x,callfun,filename,fig_size=(8,5),color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']):
    
    nbody = callfun[0]['nbody']
    
    Newt_err = Compute_Newton_err(x,callfun)
    # print(np.linalg.norm(Newt_err)/(callfun[0]['nint_list'][callfun[0]['current_cvg_lvl']]*nbody))
    
    Newt_err = np.linalg.norm(Newt_err,axis=(1))
    
    fig = plt.figure()
    fig.set_size_inches(fig_size)
    ax = plt.gca()
    
    ncol = len(color_list)

    cb = []
    for ib in range(nbody):
        cb.append(color_list[ib%ncol])
    
    # for ib in range(nbody):
    for ib in range(1):
        ax.plot(Newt_err[ib,:],c=cb[ib])
        
    ax.set_yscale('log')
        
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_all_2D(x,nint_plot,callfun,filename,fig_size=(10,10),dpi=100,color=None,color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']):
    # Plots 2D trajectories and saves image under filename
    
    if isinstance(color,list):
        
        for the_color in color :
            
            file_bas,file_ext = os.path.splitext(filename)
            
            the_filename = file_bas+'_'+the_color+file_ext
            
            plot_all_2D(x=x,nint_plot=nint_plot,callfun=callfun,filename=the_filename,fig_size=fig_size,dpi=dpi,color=the_color,color_list=color_list)
    
    elif (color is None) or (color == "body") or (color == "loop"):
        
        plot_all_2D_cpb(x,nint_plot,callfun,filename,fig_size=fig_size,dpi=dpi,color=color,color_list=color_list)
        
    elif (color == "velocity"):
        
        plot_all_2D_cpv(x,nint_plot,callfun,filename,fig_size=fig_size,dpi=dpi)
        
    elif (color == "all"):
        
        plot_all_2D(x=x,nint_plot=nint_plot,callfun=callfun,filename=filename,fig_size=fig_size,dpi=dpi,color=["body","velocity"],color_list=color_list)

    else:
        
        raise ValueError("Unknown color scheme")

def plot_all_2D_cpb(x,nint_plot,callfun,filename,fig_size=(10,10),dpi=100,color=None,color_list = plt.rcParams['axes.prop_cycle'].by_key()['color'],xlim=None,extend=0.03):
    # Plots 2D trajectories with one color per body and saves image under filename
    
    args = callfun[0]
    
    all_coeffs = Unpackage_all_coeffs(x,callfun)
    
    nloop = args['nloop']
    nbody = args['nbody']
    loopnb = args['loopnb']
    Targets = args['Targets']
    SpaceRotsUn = args['SpaceRotsUn']
    
    c_coeffs = all_coeffs.view(dtype=np.complex128)[...,0]
    
    all_pos = np.zeros((nloop,ndim,nint_plot+1),dtype=np.float64)
    all_pos[:,:,0:nint_plot] = the_irfft(c_coeffs,n=nint_plot,axis=2)*nint_plot
    all_pos[:,:,nint_plot] = all_pos[:,:,0]
    
    all_pos_b = np.zeros((nbody,ndim,nint_plot+1),dtype=np.float64)
    
    for il in range(nloop):
        for ib in range(loopnb[il]):
            for iint in range(nint_plot+1):
                # exact time is irrelevant
                all_pos_b[Targets[il,ib],:,iint] = np.dot(SpaceRotsUn[il,ib,:,:],all_pos[il,:,iint])
    
    ncol = len(color_list)
    
    cb = ['b' for ib in range(nbody)]

    if (color is None) or (color == "body"):
        for ib in range(nbody):
            cb[ib] = color_list[ib%ncol]
        
    elif (color == "loop"):
        
        for il in range(nloop):
            for ib in range(loopnb[il]):
                ibb = Targets[il,ib]
                cb[ibb] = color_list[il%ncol]

    if xlim is None:

        xmin = all_pos_b[:,0,:].min()
        xmax = all_pos_b[:,0,:].max()
        ymin = all_pos_b[:,1,:].min()
        ymax = all_pos_b[:,1,:].max()

    else :

        xmin = xlim[0]
        xmax = xlim[1]
        ymin = xlim[2]
        ymax = xlim[3]
    
    xinf = xmin - extend*(xmax-xmin)
    xsup = xmax + extend*(xmax-xmin)
    
    yinf = ymin - extend*(ymax-ymin)
    ysup = ymax + extend*(ymax-ymin)

    # Plot-related
    fig = plt.figure()
    fig.set_size_inches(fig_size)
    fig.set_dpi(dpi)
    ax = plt.gca()
    # lines = sum([ax.plot([], [],'b-', antialiased=True)  for ib in range(nbody)], [])
    lines = sum([ax.plot([], [],'-',color=cb[ib] ,antialiased=True,zorder=-ib)  for ib in range(nbody)], [])
    points = sum([ax.plot([], [],'ko', antialiased=True)for ib in range(nbody)], [])
    
    # print(xinf,xsup)
    # print(yinf,ysup)
    
    ax.axis('off')
    ax.set_xlim([xinf, xsup])
    ax.set_ylim([yinf, ysup ])
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    
    for ib in range(nbody):

        lines[ib].set_data(all_pos_b[ib,0,:], all_pos_b[ib,1,:])

    plt.savefig(filename)
    
    plt.close()

def plot_all_2D_cpv(x,nint_plot,callfun,filename,fig_size=(10,10),dpi=100,xlim=None,extend=0.03):
    # Plots 2D trajectories with one color per body and saves image under filename
    
    args = callfun[0]
    
    all_coeffs = Unpackage_all_coeffs(x,callfun)
    
    nloop = args['nloop']
    nbody = args['nbody']
    loopnb = args['loopnb']
    Targets = args['Targets']
    SpaceRotsUn = args['SpaceRotsUn']
    
    c_coeffs = all_coeffs.view(dtype=np.complex128)[...,0]
    
    all_pos = np.zeros((nloop,ndim,nint_plot+1),dtype=np.float64)
    all_pos[:,:,0:nint_plot] = the_irfft(c_coeffs,n=nint_plot,axis=2)*nint_plot
    all_pos[:,:,nint_plot] = all_pos[:,:,0]
    
    all_coeffs_v = np.zeros(all_coeffs.shape)
    
    for k in range(args['ncoeff_list'][args["current_cvg_lvl"]]):
        all_coeffs_v[:,:,k,0] = -k * all_coeffs[:,:,k,1]
        all_coeffs_v[:,:,k,1] =  k * all_coeffs[:,:,k,0]
    
    c_coeffs_v = all_coeffs_v.view(dtype=np.complex128)[...,0]
    
    all_vel = np.zeros((nloop,nint_plot+1),dtype=np.float64)
    all_vel[:,0:nint_plot] = np.linalg.norm(the_irfft(c_coeffs_v,n=nint_plot,axis=2),axis=1)*nint_plot
    all_vel[:,nint_plot] = all_vel[:,0]
    
    all_pos_b = np.zeros((nbody,ndim,nint_plot+1),dtype=np.float64)
    
    for il in range(nloop):
        for ib in range(loopnb[il]):
            for iint in range(nint_plot+1):
                # exact time is irrelevant
                all_pos_b[Targets[il,ib],:,iint] = np.dot(SpaceRotsUn[il,ib,:,:],all_pos[il,:,iint])
    
    all_vel_b = np.zeros((nbody,nint_plot+1),dtype=np.float64)
    
    for il in range(nloop):
        for ib in range(loopnb[il]):
            for iint in range(nint_plot+1):
                # exact time is irrelevant
                all_vel_b[Targets[il,ib],iint] = all_vel[il,iint]
    
    if xlim is None:

        xmin = all_pos_b[:,0,:].min()
        xmax = all_pos_b[:,0,:].max()
        ymin = all_pos_b[:,1,:].min()
        ymax = all_pos_b[:,1,:].max()

    else :

        xmin = xlim[0]
        xmax = xlim[1]
        ymin = xlim[2]
        ymax = xlim[3]

    xinf = xmin - extend*(xmax-xmin)
    xsup = xmax + extend*(xmax-xmin)
    
    yinf = ymin - extend*(ymax-ymin)
    ysup = ymax + extend*(ymax-ymin)

    # Plot-related
    fig = plt.figure()
    fig.set_size_inches(fig_size)
    fig.set_dpi(dpi)
    ax = plt.gca()

    # cmap = None
    cmap = 'turbo'
    # cmap = 'rainbow'
    
    norm = plt.Normalize(0,all_vel_b.max())
    
    for ib in range(nbody-1,-1,-1):
                
        points = all_pos_b[ib,:,:].T.reshape(-1,1,2)
        segments = np.concatenate([points[:-1],points[1:]],axis=1)
        
        lc = LineCollection(segments,cmap=cmap,norm=norm)
        lc.set_array(all_vel_b[ib,:])
        
        ax.add_collection(lc)

    ax.axis('off')
    ax.set_xlim([xinf, xsup])
    ax.set_ylim([yinf, ysup ])
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()

    plt.savefig(filename)
    
    plt.close()
 
def plot_all_2D_anim(x,nint_plot,callfun,filename,nperiod=1,Plot_trace=True,fig_size=(5,5),dnint=1,all_pos_trace=None,all_pos_points=None,xlim=None,extend=0.03):
    # Creates a video of the bodies moving along their trajectories, and saves the file under filename
    
    args = callfun[0]
    
    all_coeffs = Unpackage_all_coeffs(x,callfun)
    
    nloop = args['nloop']
    nbody = args['nbody']
    loopnb = args['loopnb']
    Targets = args['Targets']
    SpaceRotsUn = args['SpaceRotsUn']
    TimeRevsUn = args['TimeRevsUn']
    TimeShiftNumUn = args['TimeShiftNumUn']
    TimeShiftDenUn = args['TimeShiftDenUn']
    
    maxloopnb = loopnb.max()
    
    nint_plot_img = nint_plot*dnint
    nint_plot_vid = nint_plot

    if (all_pos_trace is None) or (all_pos_points is None):

        all_pos_b = np.zeros((nbody,ndim,nint_plot_img+1),dtype=np.float64)
        all_pos_b = ComputeAllPos(x,callfun,nint=nint_plot_img)
        all_pos_b[:,:,nint_plot_img] = all_pos_b[:,:,0]

    if (all_pos_trace is None):
        all_pos_trace = all_pos_b

    if (all_pos_points is None):
        all_pos_points = all_pos_b


    if xlim is None:

        xmin = all_pos_trace[:,0,:].min()
        xmax = all_pos_trace[:,0,:].max()
        ymin = all_pos_trace[:,1,:].min()
        ymax = all_pos_trace[:,1,:].max()

    else :

        xmin = xlim[0]
        xmax = xlim[1]
        ymin = xlim[2]
        ymax = xlim[3]

    xinf = xmin - extend*(xmax-xmin)
    xsup = xmax + extend*(xmax-xmin)
    
    yinf = ymin - extend*(ymax-ymin)
    ysup = ymax + extend*(ymax-ymin)
    
    # Plot-related
    fig = plt.figure()
    fig.set_size_inches(fig_size)
    ax = plt.gca()
    lines = sum([ax.plot([], [],'-', antialiased=True,zorder=-ib)  for ib in range(nbody)], [])
    points = sum([ax.plot([], [],'ko', antialiased=True)for ib in range(nbody)], [])
    
    # print(xinf,xsup)
    # print(yinf,ysup)
    
    ax.axis('off')
    ax.set_xlim([xinf, xsup])
    ax.set_ylim([yinf, ysup ])
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    
    # TODO: Understand why this is needed / how to rationalize this use. Is it even legal python ?

    iint = [0]
    
    def init():
        
        if (Plot_trace):
            for ib in range(nbody):
                lines[ib].set_data(all_pos_trace[ib,0,:], all_pos_trace[ib,1,:])
        
        return lines + points

    def update(i):
        
        for ib in range(nbody):
            points[ib].set_data(all_pos_points[ib,0,iint[0]], all_pos_points[ib,1,iint[0]])
            
        iint[0] = ((iint[0]+dnint) % nint_plot_img)

        return lines + points
    
    anim = animation.FuncAnimation(fig, update, frames=int(nperiod*nint_plot_vid),init_func=init, blit=True)
                        
    # Save as mp4. This requires mplayer or ffmpeg to be installed
    # anim.save(filename, fps=30, extra_args=['-vcodec', 'libx264'])
    anim.save(filename, fps=30)
    
    plt.close()
    
def Images_to_video(input_folder,output_filename,ReverseEnd=False,img_file_ext='.png'):
    # Expects images files with consistent extension (default *.png).
    
    list_files = os.listdir(input_folder)

    png_files = []
    
    for the_file in list_files:
        
        the_file = input_folder+the_file
        
        if os.path.isfile(the_file):
            
            file_base , file_ext = os.path.splitext(the_file)
            
            if file_ext == img_file_ext :
                
                png_files.append(the_file)
        
    # Sorting files by alphabetical order
    png_files.sort()
        
    frames_filename = 'frames.txt'
    
    f = open(frames_filename,"w")
    for img_name in img_list:
        f.write('file \''+os.path.abspath(img_name)+'\'\n')
        f.write('duration 0.0333333 \n')
    
    if ReverseEnd:
        for img_name in reversed(img_list):
            f.write('file \''+os.path.abspath(img_name)+'\'\n')
            f.write('duration 0.0333333 \n')
        
    f.close()

    try:
        (
            ffmpeg
            .input(frames_filename,f='concat',safe='0')
            .output(output_filename,vcodec='h264',pix_fmt='yuv420p')
            .global_args('-y')
            .global_args('-loglevel','error')
            .run()
        )
    except:
        raise ModuleNotFoundError('Error: ffmpeg not found')
    
    os.remove(frames_filename)
 

def factor_squarest(n):
    x = m.ceil(m.sqrt(n))
    y = int(n/x)
    while ( (y * x) != float(n) ):
        x -= 1
        y = int(n/x)
    return max(x,y), min(x,y)

def VideoGrid(input_list,output_filename,nxy = None,ordering='RowMajor'):

    nvid = len(input_list)

    if nxy is None:
         nx,ny = factor_squarest(nvid)

    else:
        nx,ny = nxy
        if (nx*ny != nvid):
            raise(ValueError('The number of input video files is incorrect'))

    if ordering == 'RowMajor':
        layout_list = []
        for iy in range(ny):
            if iy == 0:
                ylayout='0'
            else:
                ylayout = 'h0'
                for iiy in range(1,iy):
                    ylayout = ylayout + '+h'+str(iiy)

            for ix in range(nx):
                if ix == 0:
                    xlayout='0'
                else:
                    xlayout = 'w0'
                    for iix in range(1,ix):
                        xlayout = xlayout + '+w'+str(iix)


                layout_list.append(xlayout + '_' + ylayout)

    elif ordering == 'ColMajor':
        layout_list = []
        for ix in range(nx):
            if ix == 0:
                xlayout='0'
            else:
                xlayout = 'w0'
                for iix in range(1,ix):
                    xlayout = xlayout + '+w'+str(iix)
            for iy in range(ny):
                if iy == 0:
                    ylayout='0'
                else:
                    ylayout = 'h0'
                    for iiy in range(1,iy):
                        ylayout = ylayout + '+h'+str(iiy)

                layout_list.append(xlayout + '_' + ylayout)
    else:
        raise(ValueError('Unknown ordering : '+ordering))

    layout = layout_list[0]
    for i in range(1,nvid):
        layout = layout + '|' + layout_list[i]

    try:
        
        ffmpeg_input_list = []
        for the_input in input_list:
            ffmpeg_input_list.append(ffmpeg.input(the_input))

        # ffmpeg_out = ( ffmpeg
        #     .filter(ffmpeg_input_list, 'hstack')
        # )
# 
        ffmpeg_out = ( ffmpeg
            .filter(
                ffmpeg_input_list,
                'xstack',
                inputs=nvid,
                layout=layout,
            )
        )

        ffmpeg_out = ( ffmpeg_out
            .output(output_filename,vcodec='h264',pix_fmt='yuv420p')
            .global_args('-y')
            .global_args('-loglevel','error')
        )

        ffmpeg_out.run()

    # except:
    #     raise ModuleNotFoundError('Error: ffmpeg not found')

    except BaseException as err:
        print(f"Unexpected {err=}, {type(err)=}")
        raise

def Write_Descriptor(x,callfun,filename):
    # Dumps a text file describing the current trajectories
    
    args = callfun[0]
    
    with open(filename,'w') as filename_write:
        
        filename_write.write('Number of bodies : {:d}\n'.format(args['nbody']))
        
        # filename_write.write('Number of loops : {:d}\n'.format(args['nloop']))
        
        filename_write.write('Mass of those bodies : ')
        for il in range(args['nbody']):
            filename_write.write(' {:f}'.format(args['mass'][il]))
        filename_write.write('\n')
        
        filename_write.write('Number of Fourier coefficients in each loop : {:d}\n'.format(args['ncoeff_list'][args["current_cvg_lvl"]]))
        filename_write.write('Number of integration points for the action : {:d}\n'.format(args['nint_list'][args["current_cvg_lvl"]]))
        
        Action,Gradaction = Compute_action(x,callfun)
        
        filename_write.write('Value of the Action : {:.16f}\n'.format(Action))
        filename_write.write('Value of the Norm of the Gradient of the Action : {:.10E}\n'.format(np.linalg.norm(Gradaction)))

        Newt_err = Compute_Newton_err(x,callfun)
        Newt_err_norm = np.linalg.norm(Newt_err)/(args['nint_list'][args["current_cvg_lvl"]]*args['nbody'])
        filename_write.write('Sum of Newton Errors : {:.10E}\n'.format(Newt_err_norm))
        
        dxmin = Compute_MinDist(x,callfun)
        filename_write.write('Minimum inter-body distance : {:.10E}\n'.format(dxmin))
        
        Hash_Action = Compute_hash_action(x,callfun)
        filename_write.write('Hash Action for duplicate detection : ')
        for ihash in range(nhash):
            filename_write.write(' {:.16f}'.format(Hash_Action[ihash]))
        filename_write.write('\n')
        
        Escaped,dists = Detect_Escape(x,callfun)
        
        filename_write.write('Escaped detection : ')
        if Escaped:
            filename_write.write(' True ')
        else:
            filename_write.write(' False ')
            
        for i in range(2):
            filename_write.write(' {:.10f}'.format(dists[i]))
        filename_write.write(' {:.10f}'.format(dists[1]/(args['nbody']*dists[0])))    
        
        filename_write.write('\n')
         
def ReadActionFromFile(filename):
    with open(filename,'r') as file_read:
        file_readlines = file_read.readlines()
        for iline in range(len(file_readlines)):
            line = file_readlines[iline]

            if (line[0:Action_msg_len] == Action_msg):
                This_Action = float(line[Action_msg_len:])
            
            elif (line[0:Action_Hash_msg_len] == Action_Hash_msg):
                split_nums = line[Action_Hash_msg_len:].split()
                
                This_Action_Hash = np.array([float(num_str) for num_str in split_nums])
    
    return This_Action,This_Action_Hash

def SelectFiles_Action(store_folder,hash_dict,Action_val=0,Action_Hash_val=np.zeros((nhash)),rtol=1e-5):
    # Creates a list of possible duplicates based on value of the action and hashes

    file_path_list = []
    for file_path in os.listdir(store_folder):
        file_path = os.path.join(store_folder, file_path)
        file_root, file_ext = os.path.splitext(os.path.basename(file_path))
        
        if (file_ext == '.txt' ):
            
            Saved_Action = hash_dict.get(file_root)
            
            if (Saved_Action is None) :
            
                # print(file_path)
                with open(file_path,'r') as file_read:
                    file_readlines = file_read.readlines()
                    for iline in range(len(file_readlines)):
                        line = file_readlines[iline]

                        if (line[0:Action_msg_len] == Action_msg):
                            This_Action = float(line[Action_msg_len:])
                        
                        elif (line[0:Action_Hash_msg_len] == Action_Hash_msg):
                            split_nums = line[Action_Hash_msg_len:].split()
                            
                            This_Action_Hash = np.array([float(num_str) for num_str in split_nums])
                            
                    
                    hash_dict[file_root] = [This_Action,This_Action_Hash]
                    
            else:
                
                 This_Action = Saved_Action[0]
                 This_Action_Hash = Saved_Action[1]
                        
            IsCandidate = (abs(This_Action-Action_val) < ((abs(This_Action)+abs(Action_val))*rtol))
            for ihash in range(nhash):
                IsCandidate = (IsCandidate and ((abs(This_Action_Hash[ihash]-Action_Hash_val[ihash])) < ((abs(This_Action_Hash[ihash])+abs(Action_Hash_val[ihash]))*rtol)))
            
            if IsCandidate:
                
                file_path_list.append(store_folder+'/'+file_root)
                    
    return file_path_list

def Check_Duplicates(x,callfun,hash_dict,store_folder,duplicate_eps):
    # Checks whether there is a duplicate of a given trajecory in the provided folder

    Action,Gradaction = Compute_action(x,callfun)
    Hash_Action = Compute_hash_action(x,callfun)

    file_path_list = SelectFiles_Action(store_folder,hash_dict,Action,Hash_Action,duplicate_eps)
    
    if (len(file_path_list) == 0):
        
        Found_duplicate = False
        file_path = ''
    
    else:
        Found_duplicate = True
        file_path = file_path_list[0]
    
    return Found_duplicate,file_path

    
