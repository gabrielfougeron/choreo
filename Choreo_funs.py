'''
Choreo_funs.py : Defines useful functions in the Choreographies2 project.

'''

import os
import itertools
import copy
import h5py
import time
import pickle

import numpy as np
import math as m
import scipy.optimize as opt
import scipy.linalg as la
import scipy.sparse as sp
import sparseqr
import networkx as nx

import logging
logging.disable(logging.WARNING)

import fractions

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import cnames
from matplotlib.collections import LineCollection
from matplotlib import animation

from Choreo_cython_funs import *

def plot_all_2D(x,nint_plot,callfun,filename,fig_size=(10,10),color=None,color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']):
    # Plots 2D trajectories and saves image under filename
    
    if isinstance(color,list):
        
        for the_color in color :
            
            file_bas,file_ext = os.path.splitext(filename)
            
            the_filename = file_bas+'_'+the_color+file_ext
            
            plot_all_2D(x=x,nint_plot=nint_plot,callfun=callfun,filename=the_filename,fig_size=fig_size,color=the_color,color_list=color_list)
    
    elif (color is None) or (color == "body") or (color == "loop"):
        
        plot_all_2D_cpb(x,nint_plot,callfun,filename,fig_size=(10,10),color=color,color_list=color_list)
        
    elif (color == "velocity"):
        
        plot_all_2D_cpv(x,nint_plot,callfun,filename,fig_size=(10,10))
        
    elif (color == "all"):
        
        plot_all_2D(x=x,nint_plot=nint_plot,callfun=callfun,filename=filename,fig_size=fig_size,color=["body","velocity"],color_list=color_list)

    else:
        
        raise ValueError("Unknown color scheme")

def plot_all_2D_cpb(x,nint_plot,callfun,filename,fig_size=(10,10),color=None,color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']):
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
    all_pos[:,:,0:nint_plot] = np.fft.irfft(c_coeffs,n=nint_plot,axis=2)*nint_plot
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



    xmin = all_pos_b[:,0,:].min()
    xmax = all_pos_b[:,0,:].max()
    ymin = all_pos_b[:,1,:].min()
    ymax = all_pos_b[:,1,:].max()
    
    r = 0.03
    
    xinf = xmin - r*(xmax-xmin)
    xsup = xmax + r*(xmax-xmin)
    
    yinf = ymin - r*(ymax-ymin)
    ysup = ymax + r*(ymax-ymin)

    # Plot-related
    fig = plt.figure()
    fig.set_size_inches(fig_size)
    ax = plt.gca()
    # ~ lines = sum([ax.plot([], [],'b-', antialiased=True)  for ib in range(nbody)], [])
    lines = sum([ax.plot([], [],'-',color=cb[ib] ,antialiased=True,zorder=-ib)  for ib in range(nbody)], [])
    points = sum([ax.plot([], [],'ko', antialiased=True)for ib in range(nbody)], [])
    
    # ~ print(xinf,xsup)
    # ~ print(yinf,ysup)
    
    ax.axis('off')
    ax.set_xlim([xinf, xsup])
    ax.set_ylim([yinf, ysup ])
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    
    for ib in range(nbody):

        lines[ib].set_data(all_pos_b[ib,0,:], all_pos_b[ib,1,:])

    plt.savefig(filename)
    
    plt.close()

def plot_all_2D_cpv(x,nint_plot,callfun,filename,fig_size=(10,10)):
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
    all_pos[:,:,0:nint_plot] = np.fft.irfft(c_coeffs,n=nint_plot,axis=2)*nint_plot
    all_pos[:,:,nint_plot] = all_pos[:,:,0]
    
    all_coeffs_v = np.zeros(all_coeffs.shape)
    
    for k in range(args['ncoeff_list'][args["current_cvg_lvl"]]):
        all_coeffs_v[:,:,k,0] = -k * all_coeffs[:,:,k,1]
        all_coeffs_v[:,:,k,1] =  k * all_coeffs[:,:,k,0]
    
    c_coeffs_v = all_coeffs_v.view(dtype=np.complex128)[...,0]
    
    all_vel = np.zeros((nloop,nint_plot+1),dtype=np.float64)
    all_vel[:,0:nint_plot] = np.linalg.norm(np.fft.irfft(c_coeffs_v,n=nint_plot,axis=2),axis=1)*nint_plot
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
    
    xmin = all_pos_b[:,0,:].min()
    xmax = all_pos_b[:,0,:].max()
    ymin = all_pos_b[:,1,:].min()
    ymax = all_pos_b[:,1,:].max()
    
    r = 0.03
    
    xinf = xmin - r*(xmax-xmin)
    xsup = xmax + r*(xmax-xmin)
    
    yinf = ymin - r*(ymax-ymin)
    ysup = ymax + r*(ymax-ymin)

    # Plot-related
    fig = plt.figure()
    fig.set_size_inches(fig_size)
    ax = plt.gca()

    # ~ cmap = None
    cmap = 'turbo'
    # ~ cmap = 'rainbow'
    
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
 
def plot_all_2D_anim(x,nint_plot,callfun,filename,nperiod=1,Plot_trace=True,fig_size=(5,5)):
    # Creates a vide of the bodies moving along their trajectories, and saves the file under filename
    
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
    
    c_coeffs = all_coeffs.view(dtype=np.complex128)[...,0]
    
    all_pos = np.zeros((nloop,ndim,nint_plot+1),dtype=np.float64)
    all_pos[:,:,0:nint_plot] = np.fft.irfft(c_coeffs,n=nint_plot,axis=2)*nint_plot
    all_pos[:,:,nint_plot] = all_pos[:,:,0]
    
    all_pos_b = np.zeros((nbody,ndim,nint_plot+1),dtype=np.float64)
    all_shiftsUn = np.zeros((nloop,maxloopnb),dtype=np.int_)
    
    xmin = all_coeffs[0,0,0,0]
    xmax = xmin
    ymin = all_coeffs[0,1,0,0]
    ymax = ymin
    
    for il in range(nloop):
        for ib in range(loopnb[il]):
                
            # ~ if not(((-TimeRevsUn[il,ib]*nint_plot*TimeShiftNumUn[il,ib]) % TimeShiftDenUn[il,ib]) == 0):
                # ~ print("WARNING : remainder in integer division")
                
            all_shiftsUn[il,ib] = ((-TimeRevsUn[il,ib]*nint_plot*TimeShiftNumUn[il,ib]) // TimeShiftDenUn[il,ib] ) % nint_plot

    for iint in range(nint_plot+1):    
        for il in range(nloop):
            for ib in range(loopnb[il]):

                all_pos_b[Targets[il,ib],:,iint] = np.dot(SpaceRotsUn[il,ib,:,:],all_pos[il,:,all_shiftsUn[il,ib]])

                all_shiftsUn[il,ib] = (all_shiftsUn[il,ib]+TimeRevsUn[il,ib]) % nint_plot
                
                xmin = min(xmin,all_pos_b[Targets[il,ib],0,iint])
                xmax = max(xmax,all_pos_b[Targets[il,ib],0,iint])
                ymin = min(ymin,all_pos_b[Targets[il,ib],1,iint])
                ymax = max(ymax,all_pos_b[Targets[il,ib],1,iint])
                              
    r = 0.03
    
    xinf = xmin - r*(xmax-xmin)
    xsup = xmax + r*(xmax-xmin)
    
    yinf = ymin - r*(ymax-ymin)
    ysup = ymax + r*(ymax-ymin)
    
    # Plot-related
    fig = plt.figure()
    fig.set_size_inches(fig_size)
    ax = plt.gca()
    lines = sum([ax.plot([], [],'-', antialiased=True,zorder=-ib)  for ib in range(nbody)], [])
    points = sum([ax.plot([], [],'ko', antialiased=True)for ib in range(nbody)], [])
    
    # ~ print(xinf,xsup)
    # ~ print(yinf,ysup)
    
    ax.axis('off')
    ax.set_xlim([xinf, xsup])
    ax.set_ylim([yinf, ysup ])
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    
    iint = [0]
    
    def init():
        
        if (Plot_trace):
            for ib in range(nbody):
                lines[ib].set_data(all_pos_b[ib,0,:], all_pos_b[ib,1,:])
        
        return lines + points

    def update(i):
        
        for ib in range(nbody):
            points[ib].set_data(all_pos_b[ib,0,iint[0]], all_pos_b[ib,1,iint[0]])
            
        iint[0] = ((iint[0]+1) % nint_plot)

        return lines + points
    
    anim = animation.FuncAnimation(fig, update, frames=int(nperiod*nint_plot),init_func=init, blit=True)
                        
    # Save as mp4. This requires mplayer or ffmpeg to be installed
    # ~ anim.save(filename, fps=30, extra_args=['-vcodec', 'libx264'])
    anim.save(filename, fps=30)
    
    plt.close()
 
def Package_all_coeffs(all_coeffs,callfun):
    # Transfers the Fourier coefficients of the generators to a single vector of parameters.
    # The packaging process projects the trajectory onto the space of constrraint satisfying trajectories.
    
    args = callfun[0]

    y = all_coeffs.reshape(-1)
    x = args['coeff_to_param_list'][args["current_cvg_lvl"]].dot(y)
    
    return x
    
def Unpackage_all_coeffs(x,callfun):
    # Computes the Fourier coefficients of the generator given the parameters.
    
    args=callfun[0]
    
    y = args['param_to_coeff_list'][args["current_cvg_lvl"]] * x
    all_coeffs = y.reshape(args['nloop'],ndim,args['ncoeff_list'][args["current_cvg_lvl"]],2)
    
    return all_coeffs

def Compute_action_onlygrad(x,callfun):
    # Wrapper function that returns ONLY the gradient of the action with respect to the parameters 
    
    J,y = Compute_action(x,callfun)
    
    return y
    
def Compute_action_onlygrad_escape(x,callfun):
    # Cumputes the action and its gradient with respect to the parameters at a given value of the parameters

    args=callfun[0]
    
    y = args['param_to_coeff_list'][args["current_cvg_lvl"]] * x
    all_coeffs = y.reshape(args['nloop'],ndim,args['ncoeff_list'][args["current_cvg_lvl"]],2)

    # ~ rms_dist = Compute_Loop_Dist_Cython(
        # ~ args['nloop']           ,
        # ~ args['ncoeff_list'][args["current_cvg_lvl"]]          ,
        # ~ args['nint_list'][args["current_cvg_lvl"]]            ,
        # ~ args['mass']            ,
        # ~ args['loopnb']          ,
        # ~ args['Targets']         ,
        # ~ args['MassSum']         ,
        # ~ args['SpaceRotsUn']     ,
        # ~ args['TimeRevsUn']      ,
        # ~ args['TimeShiftNumUn']  ,
        # ~ args['TimeShiftDenUn']  ,
        # ~ args['loopnbi']         ,
        # ~ args['ProdMassSumAll']  ,
        # ~ args['SpaceRotsBin']    ,
        # ~ args['TimeRevsBin']     ,
        # ~ args['TimeShiftNumBin'] ,
        # ~ args['TimeShiftDenBin'] ,
        # ~ all_coeffs
        # ~ )

    rms_dist = Compute_Loop_Dist_btw_avg_Cython(
        args['nloop']           ,
        args['ncoeff_list'][args["current_cvg_lvl"]]          ,
        args['nint_list'][args["current_cvg_lvl"]]            ,
        args['mass']            ,
        args['loopnb']          ,
        args['Targets']         ,
        args['MassSum']         ,
        args['SpaceRotsUn']     ,
        args['TimeRevsUn']      ,
        args['TimeShiftNumUn']  ,
        args['TimeShiftDenUn']  ,
        args['loopnbi']         ,
        args['ProdMassSumAll']  ,
        args['SpaceRotsBin']    ,
        args['TimeRevsBin']     ,
        args['TimeShiftNumBin'] ,
        args['TimeShiftDenBin'] ,
        all_coeffs
        )

    escape_pen = 1 + args['escape_fac'] * abs(rms_dist)**args['escape_pow']
    
    # ~ print("escape_pen = ",escape_pen)

    nint = args['nint_list'][args["current_cvg_lvl"]]
    c_coeffs = all_coeffs.view(dtype=np.complex128)[...,0]
    all_pos = np.fft.irfft(c_coeffs,n=nint,axis=2)*nint

    J,GradJ =  Compute_action_Cython(
        args['nloop']           ,
        args['ncoeff_list'][args["current_cvg_lvl"]]          ,
        args['nint_list'][args["current_cvg_lvl"]]            ,
        args['mass']            ,
        args['loopnb']          ,
        args['Targets']         ,
        args['MassSum']         ,
        args['SpaceRotsUn']     ,
        args['TimeRevsUn']      ,
        args['TimeShiftNumUn']  ,
        args['TimeShiftDenUn']  ,
        args['loopnbi']         ,
        args['ProdMassSumAll']  ,
        args['SpaceRotsBin']    ,
        args['TimeRevsBin']     ,
        args['TimeShiftNumBin'] ,
        args['TimeShiftDenBin'] ,
        all_coeffs              ,
        all_pos
        )

    GJ = GradJ.reshape(-1)
    GJparam = (GJ * args['param_to_coeff_list'][args["current_cvg_lvl"]]) * escape_pen
    
    return GJparam
    
def Compute_action_hess_mul(x,dx,callfun):
    # Returns the Hessian of the action (computed wrt the parameters) times a test vector of parameter deviations.
    
    args=callfun[0]

    dy = args['param_to_coeff_list'][args["current_cvg_lvl"]] * dx
    all_coeffs_d = dy.reshape(args['nloop'],ndim,args['ncoeff_list'][args["current_cvg_lvl"]],2)
    
    if args["Do_Pos_FFT"]:
        
        y = args['param_to_coeff_list'][args["current_cvg_lvl"]] * x
        args['last_all_coeffs'] = y.reshape(args['nloop'],ndim,args['ncoeff_list'][args["current_cvg_lvl"]],2)
        
        nint = args['nint_list'][args["current_cvg_lvl"]]
        c_coeffs = args['last_all_coeffs'].view(dtype=np.complex128)[...,0]
        args['last_all_pos'] = np.fft.irfft(c_coeffs,n=nint,axis=2)*nint
    
    HessJdx =  Compute_action_hess_mul_Cython(
        args['nloop']           ,
        args['ncoeff_list'][args["current_cvg_lvl"]]          ,
        args['nint_list'][args["current_cvg_lvl"]]            ,
        args['mass']            ,
        args['loopnb']          ,
        args['Targets']         ,
        args['MassSum']         ,
        args['SpaceRotsUn']     ,
        args['TimeRevsUn']      ,
        args['TimeShiftNumUn']  ,
        args['TimeShiftDenUn']  ,
        args['loopnbi']         ,
        args['ProdMassSumAll']  ,
        args['SpaceRotsBin']    ,
        args['TimeRevsBin']     ,
        args['TimeShiftNumBin'] ,
        args['TimeShiftDenBin'] ,
        args['last_all_coeffs'] ,
        all_coeffs_d            ,
        args['last_all_pos']    ,
        )

    HJdx = HessJdx.reshape(-1)
    
    z = HJdx * args['param_to_coeff_list'][args["current_cvg_lvl"]]
    
    return z
    
def Compute_action_hess_LinOpt(x,callfun):
    # Defines the Hessian of the action wrt parameters at a given point as a Scipy LinearOperator

    args=callfun[0]

    return sp.linalg.LinearOperator((args['coeff_to_param_list'][args["current_cvg_lvl"]].shape[0],args['coeff_to_param_list'][args["current_cvg_lvl"]].shape[0]),
        matvec =  (lambda dx,xl=x,callfunl=callfun : Compute_action_hess_mul(xl,dx,callfunl)),
        rmatvec = (lambda dx,xl=x,callfunl=callfun : Compute_action_hess_mul(xl,dx,callfunl)))
    
def null_space_sparseqr(AT):
    # Returns a basis of the null space of a matrix A.
    # AT must be in COO format
    # The nullspace of the TRANSPOSE of AT will be returned

    # ~ tolerance = 1e-5
    tolerance = None

    Q, R, E, rank = sparseqr.qr( AT, tolerance=tolerance )

    nrow = AT.shape[0]
    
    if (nrow <= rank):
        # ~ raise ValueError("Kernel is empty")
        
        return sp.coo_matrix(([],([],[])),shape=(nrow,0))
    
    else:
        
        iker = 0
        while(Q.col[iker] < rank):
            iker+=1
            
        return sp.coo_matrix((Q.data[iker:],(Q.row[iker:],Q.col[iker:]-rank)),shape=(nrow,nrow-rank))

class current_best:
    # Class meant to store the best solution during scipy optimization / root finding
    # Useful since scipy does not return the best solution, but rathe the solution at the last iteration.
    
    def __init__(self,x,f):
        
        self.x = x
        self.f = f
        self.f_norm = np.linalg.norm(f)
        
    def update(self,x,f):
        
        f_norm = np.linalg.norm(f)
        
        if (f_norm < self.f_norm):
            self.x = x
            self.f = f
            self.f_norm = f_norm

    def get_best(self):
        return self.x,self.f,self.f_norm
        
class ChoreoSym():
    # This class defines the symmetries of the action
    # Useful to detect loops and constraints.
    #
    # Syntax : Giving one ChoreoSym to setup_changevar prescrbes the following symmetry / constraint :
    # x_LoopTarget(t) = SpaceRot * x_LoopSource (TimeRev * (t - TimeShift))
    #
    # Where SpaceRot is assumed orthogonal (never actually checked, so beware)
    # and TimeShift is defined as a rational fraction.
    
    def __init__(
            self,
            LoopTarget=0,
            LoopSource=0,
            SpaceRot=np.identity(ndim,dtype=np.float64),
            TimeRev=1,
            TimeShift=fractions.Fraction(numerator=0,denominator=1)
            ):

        self.LoopTarget = LoopTarget
        self.LoopSource = LoopSource
        self.SpaceRot = SpaceRot
        self.TimeRev = TimeRev
        self.TimeShift = TimeShift
        
    def Inverse(self):
        
        return ChoreoSym(
            LoopTarget=self.LoopSource,
            LoopSource=self.LoopTarget,
            SpaceRot = self.SpaceRot.transpose(),
            TimeRev = self.TimeRev,         
            TimeShift = fractions.Fraction(numerator=((-int(self.TimeRev)*self.TimeShift.numerator) % self.TimeShift.denominator),denominator=self.TimeShift.denominator)
            )

    def ComposeLight(B,A):
        # Composition B o A, i.e. applies A then B, ignoring that target A might be different from source B
        
        tshift = B.TimeShift + fractions.Fraction(numerator=int(B.TimeRev)*A.TimeShift.numerator,denominator=A.TimeShift.denominator)
        tshiftmod = fractions.Fraction(numerator=tshift.numerator % tshift.denominator,denominator =tshift.denominator)
    
        return ChoreoSym(
            LoopTarget = B.LoopTarget,
            LoopSource = A.LoopSource,
            SpaceRot = np.dot(B.SpaceRot,A.SpaceRot),
            TimeRev = (B.TimeRev * A.TimeRev),
            TimeShift = tshiftmod
            )

    def Compose(B,A):
        # Composition B o A, i.e. applies A then B
        
        if (A.LoopTarget == B.LoopSource):
            return B.ComposeLight(A)
            
        else:
            
            print(B.LoopTarget,B.LoopSource)
            print(A.LoopTarget,A.LoopSource)
            
            raise ValueError("Symmetries cannot be composed")

    def IsIdentity(self):
        
        atol = 1e-10

        if ((abs(self.TimeShift) < atol) and (self.TimeRev == 1) and (self.LoopTarget == self.LoopSource)):
            
            return np.allclose(self.SpaceRot,np.identity(ndim,dtype=np.float64),rtol=0.,atol=atol)
            
        else:
        
            return False
            
    def IsSame(self,other):
        
        return ((self.Inverse()).ComposeLight(other)).IsIdentity()

def Make2DChoreoSym(SymType,ib_list):
    # Defines symmetries of a 2-D system of bodies as classfied in [1] 
    
    # Classification :
    # C(n,k,l) with k and l relative primes
    # D(n,k,l) with k and l relative primes
    # Cp(n,2,#) 
    # Dp(n,1,#) 
    # Dp(n,2,#) 
    # Those are exhaustive for 2-D purely choreographic symmetries (i.e. 1 loop with 1 path)
    
    # I also added p and q for space rotation. This might however not be exhaustive.
    
    # SymType  => Dictionary containing the following keys :
        # 'name'
        # 'n'
        # 'k'
        # 'l'
        # 'p'
        # 'q'
        
    # [1] : https://arxiv.org/abs/1305.0470

    # ~ if (len(ib_list) != SymType['n']):
        # ~ print("Warning : SymType and LoopLength are inconsistent")
        
    SymGens = []
    
    if (SymType['name'] in ['C','D','Cp','Dp']):
        
        rot_angle =  twopi * SymType['p'] /  SymType['q']
        s = 1
        
        for ib_rel in range(len(ib_list)-1):
            SymGens.append(ChoreoSym(
                LoopTarget=ib_list[ib_rel+1],
                LoopSource=ib_list[ib_rel  ],
                SpaceRot = np.array([[s*np.cos(rot_angle),-s*np.sin(rot_angle)],[np.sin(rot_angle),np.cos(rot_angle)]],dtype=np.float64),
                TimeRev=1,
                TimeShift=fractions.Fraction(numerator=-1,denominator=SymType['n'])
                ))

    if ((SymType['name'] == 'C') or (SymType['name'] == 'D')):
        
        rot_angle = twopi * SymType['l'] /  SymType['k']
        s = 1
        
        SymGens.append(ChoreoSym(
            LoopTarget=ib_list[0],
            LoopSource=ib_list[0],
            SpaceRot = np.array([[s*np.cos(rot_angle),-s*np.sin(rot_angle)],[np.sin(rot_angle),np.cos(rot_angle)]],dtype=np.float64),
            TimeRev=1,
            TimeShift=fractions.Fraction(numerator=1,denominator=SymType['k'])
            ))

    if (SymType['name'] == 'D'):
        
        rot_angle = 0
        s = -1

        SymGens.append(ChoreoSym(
            LoopTarget=ib_list[0],
            LoopSource=ib_list[0],
            SpaceRot = np.array([[s*np.cos(rot_angle),-s*np.sin(rot_angle)],[np.sin(rot_angle),np.cos(rot_angle)]],dtype=np.float64),
            TimeRev=-1,
            TimeShift=fractions.Fraction(numerator=0,denominator=1)
            ))
        
    if ((SymType['name'] == 'Cp') or ((SymType['name'] == 'Dp') and (SymType['k'] == 2))):
        
        rot_angle = 0
        s = -1

        SymGens.append(ChoreoSym(
            LoopTarget=ib_list[0],
            LoopSource=ib_list[0],
            SpaceRot = np.array([[s*np.cos(rot_angle),-s*np.sin(rot_angle)],[np.sin(rot_angle),np.cos(rot_angle)]],dtype=np.float64),
            TimeRev=1,
            TimeShift=fractions.Fraction(numerator=1,denominator=2)
            ))

    if (SymType['name'] == 'Dp'):
        
        rot_angle =  np.pi
        s = 1

        SymGens.append(ChoreoSym(
            LoopTarget=ib_list[0],
            LoopSource=ib_list[0],
            SpaceRot = np.array([[s*np.cos(rot_angle),-s*np.sin(rot_angle)],[np.sin(rot_angle),np.cos(rot_angle)]],dtype=np.float64),
            TimeRev=-1,
            TimeShift=fractions.Fraction(numerator=0,denominator=1)
            ))
    
    return SymGens

def Make2DChoreoSymManyLoops(nloop=None,nbpl=None,SymName=None):

    if nloop is None :
        if nbpl is None :
            raise ValueError("1")
        else:
            if isinstance(nbpl,list):
                nloop = len(nbpl)
            else:
                raise ValueError("2")
                
    else:
        if nbpl is None :
            raise ValueError("3")
        else:
            if isinstance(nbpl,int):
                nloop = [ nbpl for il in range(nloop) ]
            elif isinstance(nbpl,list):
                    if nloop != len(nbpl):
                        raise ValueError("4")
            else:
                raise ValueError("5")

    if (SymName is None):
        SymName = ['C' for il in range(nloop)]
    elif isinstance(SymName,str):
        SymName = [SymName for il in range(nloop)]

    SymGens = []

    the_lcm = m.lcm(*nbpl)

    istart = 0
    for il in range(nloop):

        SymType = {
            'name'  : SymName[il],
            'n'     : -the_lcm,
            'k'     : 1,
            'l'     : 1 ,
            'p'     : 0 ,
            'q'     : 1 ,
        }

        SymGens.extend(Make2DChoreoSym(SymType,[(i+istart) for i in range(nbpl[il])]))
                
        SymGens.append(ChoreoSym(
                        LoopTarget=istart,
                        LoopSource=istart,
                        SpaceRot = np.identity(ndim,dtype=np.float64),
                        TimeRev=1,
                        TimeShift=fractions.Fraction(numerator=1,denominator=the_lcm//nbpl[il])
                        ))
        
        istart += nbpl[il]
        
    nbody = istart
        
    return SymGens,nbody

def setup_changevar(nbody,ncoeff_init,mass,n_reconverge_it_max=6,MomCons=True,n_grad_change=1.,Sym_list=[]):
    # This function returns the callfun dictionnary to be given as input to virtually all other function.
    # It detects loops and constraints based on symmetries.
    # It defines parameters according to given constraints and diagonal change of variable
    # It computes useful objects to optimize the computation of the action :
    #   - Exhaustive list of unary transformation for generator to body
    #   - Exhaustive list of binary transformations from generator within each loop.
    
    SymGraph = nx.Graph()
    for i in range(nbody):
        SymGraph.add_node(i,Constraint_list=[])
    
    for Sym in Sym_list:
        
        if (Sym.LoopTarget > Sym.LoopSource):
            Sym=Sym.Inverse()
        
        if (Sym.LoopTarget == Sym.LoopSource):
            # Add constraint
            if not(Sym.IsIdentity()):
                SymGraph.nodes[Sym.LoopSource]["Constraint_list"].append(Sym)
            
        else:
            
            edge = (Sym.LoopTarget,Sym.LoopSource)
            
            if edge in SymGraph.edges: # adds constraint instead of adding parallel edge
                
                Constraint = Sym.Inverse().Compose(SymGraph.edges[edge]["Sym"])
                
                if not(Constraint.IsIdentity()):
                    SymGraph.nodes[Constraint.LoopSource]["Constraint_list"].append(Constraint)
            
            else: # Really adds edge
                
                SymGraph.add_edge(*edge,Sym=Sym)
            
    Cycles = list(nx.cycle_basis(SymGraph))    
    # Aggregate cycles symmetries into constraints
    
    for Cycle in Cycles:

        Constraint = ChoreoSym(LoopTarget=Cycle[0],LoopSource=Cycle[0])
        
        Cycle_len = len(Cycle)
        
        for iedge in range(Cycle_len):
            
            ibeg = Cycle[iedge]
            iend = Cycle[(iedge+1)%Cycle_len]
            
            if (ibeg < iend):

                Constraint = Constraint.Compose(SymGraph.edges[(ibeg,iend)]["Sym"])
                
            else:
                
                Constraint = Constraint.Compose(SymGraph.edges[(iend,ibeg)]["Sym"].Inverse())
            
        if not(Constraint.IsIdentity()):
            
            SymGraph.nodes[Cycle[0]]["Constraint_list"].append(Constraint)
            
    # Choose one representative per connected component
    # And aggregate path from the representative to each node of cycle
    # Then bring all constraints back to this node, and aggregate constraints
    
    ConnectedComponents = list(nx.connected_components(SymGraph))    

    nloop = len(ConnectedComponents)

    maxlooplen = 0
    for il in range(nloop):
        looplen = len(ConnectedComponents[il])
        if (looplen > maxlooplen):
            maxlooplen = looplen

    loopgen = np.zeros((nloop),dtype=int)
    loopnb = np.zeros((nloop),dtype=int)
    loopnbi = np.zeros((nloop),dtype=int)
    
    loop_gen_to_target = []
    
    Targets = np.zeros((nloop,maxlooplen),dtype=int)
    MassSum = np.zeros((nloop),dtype=np.float64)
    ProdMassSumAll_list = []
    UniqueSymsAll_list = []
    
    SpaceRotsUn = np.zeros((nloop,maxlooplen,ndim,ndim),dtype=np.float64)
    TimeRevsUn = np.zeros((nloop,maxlooplen),dtype=int)
    TimeShiftNumUn = np.zeros((nloop,maxlooplen),dtype=int)
    TimeShiftDenUn = np.zeros((nloop,maxlooplen),dtype=int)

    for il in range(len(ConnectedComponents)):
        
        loopgen[il] = ConnectedComponents[il].pop()
        
        paths_to_gen = nx.shortest_path(SymGraph, target=loopgen[il])
        
        ib = 0
        
        gen_to_target = []
        
        for istart,path in paths_to_gen.items():
            
            MassSum[il] += mass[istart]
            
            Sym = ChoreoSym(LoopTarget=istart,LoopSource=istart)
            
            path_len = len(path)
            
            for iedge in range(path_len-1):
                
                ibeg = path[iedge]
                iend = path[iedge+1]
                
                if (ibeg < iend):
                    
                    Sym = Sym.Compose(SymGraph.edges[(ibeg,iend)]["Sym"])

                else:
                    
                    Sym = Sym.Compose(SymGraph.edges[(iend,ibeg)]["Sym"].Inverse())
            
            Targets[il,ib] = istart
                    
            SpaceRotsUn[il,ib,:,:] = Sym.SpaceRot
            TimeRevsUn[il,ib] = Sym.TimeRev
            TimeShiftNumUn[il,ib] = Sym.TimeShift.numerator
            TimeShiftDenUn[il,ib] = Sym.TimeShift.denominator
            
            if (Sym.LoopTarget != loopgen[il]):
                
                for Constraint in SymGraph.nodes[Sym.LoopTarget]["Constraint_list"]:

                    Constraint = (Sym.Inverse()).Compose(Constraint.Compose(Sym))

                    if not(Constraint.IsIdentity()):
                        SymGraph.nodes[loopgen[il]]["Constraint_list"].append(Constraint)
            
            gen_to_target.append(Sym)
            
            ib+=1

        loopnb[il] = ib
        
        nbi = 0

        UniqueSyms = []
        ProdMassSum = []
        # Count unique pair transformations
        for ib in range(loopnb[il]-1):
            for ibp in range(ib+1,loopnb[il]):                
                
                Sym = (gen_to_target[ibp].Inverse()).ComposeLight(gen_to_target[ib])
                
                if Sym.IsIdentity():
                    raise ValueError("Two bodies have identical trajectories")
                
                IsUnique = True
                for isym in range(len(UniqueSyms)):

                    IsUnique = not(Sym.IsSame(UniqueSyms[isym]))

                    if not(IsUnique):
                        break

                    Sym = Sym.Inverse()
                    IsUnique = not(Sym.IsSame(UniqueSyms[isym]))

                    if not(IsUnique):
                        break

                if IsUnique:
                    UniqueSyms.append(Sym)
                    ProdMassSum.append(mass[Targets[il,ib]]*mass[Targets[il,ibp]])
                    loopnbi[il]+=1
                else:
                    ProdMassSum[isym]+=mass[Targets[il,ib]]*mass[Targets[il,ibp]]
                    
        UniqueSymsAll_list.append(UniqueSyms)
        ProdMassSumAll_list.append(ProdMassSum)

    maxloopnbi = loopnbi.max()
    
    ProdMassSumAll = np.zeros((nloop,maxloopnbi),dtype=np.float64)
    SpaceRotsBin = np.zeros((nloop,maxloopnbi,ndim,ndim),dtype=np.float64)
    TimeRevsBin = np.zeros((nloop,maxloopnbi),dtype=int)
    TimeShiftNumBin = np.zeros((nloop,maxloopnbi),dtype=int)
    TimeShiftDenBin = np.zeros((nloop,maxloopnbi),dtype=int)

    for il in range(nloop):
        for ibi in range(loopnbi[il]):
            
            ProdMassSumAll[il,ibi] = ProdMassSumAll_list[il][ibi]

            SpaceRotsBin[il,ibi,:,:] = UniqueSymsAll_list[il][ibi].SpaceRot
            TimeRevsBin[il,ibi] = UniqueSymsAll_list[il][ibi].TimeRev
            TimeShiftNumBin[il,ibi] = UniqueSymsAll_list[il][ibi].TimeShift.numerator
            TimeShiftDenBin[il,ibi] = UniqueSymsAll_list[il][ibi].TimeShift.denominator

    # Count constraints
    
    loopncstr = np.zeros((nloop),dtype=int)
    
    for il in range(nloop):
        loopncstr[il] = len(SymGraph.nodes[loopgen[il]]["Constraint_list"])
    
    maxloopncstr = loopncstr.max()
    
    SpaceRotsCstr = np.zeros((nloop,maxloopncstr,ndim,ndim),dtype=np.float64)
    TimeRevsCstr = np.zeros((nloop,maxloopncstr),dtype=int)
    TimeShiftNumCstr = np.zeros((nloop,maxloopncstr),dtype=int)
    TimeShiftDenCstr = np.zeros((nloop,maxloopncstr),dtype=int)
    
    for il in range(nloop):
        for i in range(loopncstr[il]):
            
            SpaceRotsCstr[il,i,:,:] = SymGraph.nodes[loopgen[il]]["Constraint_list"][i].SpaceRot
            TimeRevsCstr[il,i] = SymGraph.nodes[loopgen[il]]["Constraint_list"][i].TimeRev
            TimeShiftNumCstr[il,i] = SymGraph.nodes[loopgen[il]]["Constraint_list"][i].TimeShift.numerator
            TimeShiftDenCstr[il,i] = SymGraph.nodes[loopgen[il]]["Constraint_list"][i].TimeShift.denominator

    # Now detect parameters and build change of variables

    ncoeff_list = []
    nint_list = []
    param_to_coeff_list = []
    coeff_to_param_list = []

    for i in range(n_reconverge_it_max+1):
        
        ncoeff_list.append(ncoeff_init * (2**i))
        nint_list.append(2*ncoeff_list[i])

        cstrmat_sp = Assemble_Cstr_Matrix(
        nloop               ,
        ncoeff_list[i]      ,
        MomCons             ,
        mass                ,
        loopnb              ,
        Targets             ,
        MassSum             ,
        SpaceRotsUn         ,
        TimeRevsUn          ,
        TimeShiftNumUn      ,
        TimeShiftDenUn      ,
        loopncstr           ,
        SpaceRotsCstr       ,
        TimeRevsCstr        ,
        TimeShiftNumCstr    ,
        TimeShiftDenCstr    
        )

        param_to_coeff_list.append(null_space_sparseqr(cstrmat_sp))
        coeff_to_param_list.append(param_to_coeff_list[i].transpose(copy=True))

        diag_changevar(
            param_to_coeff_list[i].nnz,
            ncoeff_list[i],
            -n_grad_change,
            param_to_coeff_list[i].row,
            param_to_coeff_list[i].data,
            MassSum
            )
        
        diag_changevar(
            coeff_to_param_list[i].nnz,
            ncoeff_list[i],
            n_grad_change,
            coeff_to_param_list[i].col,
            coeff_to_param_list[i].data,
            MassSum
            )

    callfun = [{
    "nbody"                 :   nbody               ,
    "nloop"                 :   nloop               ,
    "mass"                  :   mass                ,
    "loopnb"                :   loopnb              ,
    "loopgen"               :   loopgen             ,
    "Targets"               :   Targets             ,
    "MassSum"               :   MassSum             ,
    "SpaceRotsUn"           :   SpaceRotsUn         ,
    "TimeRevsUn"            :   TimeRevsUn          ,
    "TimeShiftNumUn"        :   TimeShiftNumUn      ,
    "TimeShiftDenUn"        :   TimeShiftDenUn      ,
    "loopnbi"               :   loopnbi             ,
    "ProdMassSumAll"        :   ProdMassSumAll      ,
    "SpaceRotsBin"          :   SpaceRotsBin        ,
    "TimeRevsBin"           :   TimeRevsBin         ,
    "TimeShiftNumBin"       :   TimeShiftNumBin     ,
    "TimeShiftDenBin"       :   TimeShiftDenBin     ,
    "ncoeff_list"           :   ncoeff_list         ,
    "nint_list"             :   nint_list           ,
    "param_to_coeff_list"   :   param_to_coeff_list ,
    "coeff_to_param_list"   :   coeff_to_param_list ,
    "current_cvg_lvl"       :   0                   ,
    "last_all_coeffs"       :   None                ,
    "last_all_pos"          :   None                ,
    "Do_Pos_FFT"            :   True                ,
    }]

    return callfun
    
def Compute_action(x,callfun):
    # Cumputes the action and its gradient with respect to the parameters at a given value of the parameters

    args=callfun[0]
    
    if args["Do_Pos_FFT"]:
        
        y = args['param_to_coeff_list'][args["current_cvg_lvl"]] * x
        args['last_all_coeffs'] = y.reshape(args['nloop'],ndim,args['ncoeff_list'][args["current_cvg_lvl"]],2)
        
        nint = args['nint_list'][args["current_cvg_lvl"]]
        c_coeffs = args['last_all_coeffs'].view(dtype=np.complex128)[...,0]
        args['last_all_pos'] = np.fft.irfft(c_coeffs,n=nint,axis=2)*nint
    
    J,GradJ =  Compute_action_Cython(
        args['nloop']           ,
        args['ncoeff_list'][args["current_cvg_lvl"]]          ,
        args['nint_list'][args["current_cvg_lvl"]]            ,
        args['mass']            ,
        args['loopnb']          ,
        args['Targets']         ,
        args['MassSum']         ,
        args['SpaceRotsUn']     ,
        args['TimeRevsUn']      ,
        args['TimeShiftNumUn']  ,
        args['TimeShiftDenUn']  ,
        args['loopnbi']         ,
        args['ProdMassSumAll']  ,
        args['SpaceRotsBin']    ,
        args['TimeRevsBin']     ,
        args['TimeShiftNumBin'] ,
        args['TimeShiftDenBin'] ,
        args['last_all_coeffs'] ,
        args['last_all_pos'] 
        )

    GJ = GradJ.reshape(-1)
    y = GJ * args['param_to_coeff_list'][args["current_cvg_lvl"]]
    
    return J,y

def Compute_hash_action(x,callfun):
    # Returns an invariant hash of the trajectories.
    # Useful for duplicate detection

    args=callfun[0]
    
    y = args['param_to_coeff_list'][args["current_cvg_lvl"]] * x
    all_coeffs = y.reshape(args['nloop'],ndim,args['ncoeff_list'][args["current_cvg_lvl"]],2)
    
    Hash_Action =  Compute_hash_action_Cython(
        args['nloop']           ,
        args['ncoeff_list'][args["current_cvg_lvl"]]          ,
        args['nint_list'][args["current_cvg_lvl"]]            ,
        args['mass']            ,
        args['loopnb']          ,
        args['Targets']         ,
        args['MassSum']         ,
        args['SpaceRotsUn']     ,
        args['TimeRevsUn']      ,
        args['TimeShiftNumUn']  ,
        args['TimeShiftDenUn']  ,
        args['loopnbi']         ,
        args['ProdMassSumAll']  ,
        args['SpaceRotsBin']    ,
        args['TimeRevsBin']     ,
        args['TimeShiftNumBin'] ,
        args['TimeShiftDenBin'] ,
        all_coeffs
        )

    return Hash_Action
    
def Compute_Newton_err(x,callfun):
    # Computes the Newton error at a certain value of parameters
    # WARNING : DOUBLING NUMBER OF INTEGRATION POINTS

    args=callfun[0]
    
    y = args['param_to_coeff_list'][args["current_cvg_lvl"]] * x
    all_coeffs = y.reshape(args['nloop'],ndim,args['ncoeff_list'][args["current_cvg_lvl"]],2)
    
    all_Newt_err =  Compute_Newton_err_Cython(
        args['nbody']           ,
        args['nloop']           ,
        args['ncoeff_list'][args["current_cvg_lvl"]]          ,
        args['nint_list'][args["current_cvg_lvl"]]*2          ,
        args['mass']            ,
        args['loopnb']          ,
        args['Targets']         ,
        args['SpaceRotsUn']     ,
        args['TimeRevsUn']      ,
        args['TimeShiftNumUn']  ,
        args['TimeShiftDenUn']  ,
        all_coeffs
        )

    return all_Newt_err
    
def Compute_Loop_Size_Dist(x,callfun):
    # Computes sizes of trajetories and distance between center of trajectories
    # Useful to detect escape.
    # For checks only. There is a Cython version now
    
    args = callfun[0]
    
    all_coeffs = Unpackage_all_coeffs(x,callfun)
    
    max_loop_size = 0.
    for il in range(args['nloop']):
        loop_size = np.linalg.norm(all_coeffs[il,:,1:args['ncoeff_list'][args["current_cvg_lvl"]],:])
        max_loop_size = max(loop_size,max_loop_size)
    
    max_loop_dist = 0.
    for il in range(args['nloop']-1):
        for ilp in range(il,args['nloop']):
            
            for ib in range(args['loopnb'][il]):
                for ibp in range(args['loopnb'][ilp]):

                    loop_dist = np.linalg.norm(np.dot(args['SpaceRotsUn'][il,ib,:,:],all_coeffs[il,:,0,0]) - np.dot(args['SpaceRotsUn'][ilp,ibp,:,:],all_coeffs[ilp,:,0,0]))
                    max_loop_dist = max(loop_dist,max_loop_dist)
                    
    for il in range(args['nloop']):
        for ibi in range(args['loopnbi'][il]):
                
            loop_dist = np.linalg.norm(np.dot(args['SpaceRotsBin'][il,ibi,:,:],all_coeffs[il,:,0,0]) - all_coeffs[il,:,0,0])
            max_loop_dist = max(loop_dist,max_loop_dist)
    

    return max_loop_size,max_loop_dist
    
def Detect_Escape(x,callfun):
    # Returns True if the trajectories are so far that they are likely to never interact again
    
    # ~ max_loop_size,max_loop_dist = Compute_Loop_Size_Dist(x,callfun)
    
    args=callfun[0]
    
    y = args['param_to_coeff_list'][args["current_cvg_lvl"]] * x
    all_coeffs = y.reshape(args['nloop'],ndim,args['ncoeff_list'][args["current_cvg_lvl"]],2)
    
    res = Compute_Loop_Size_Dist_Cython(
        args['nloop']           ,
        args['ncoeff_list'][args["current_cvg_lvl"]]          ,
        args['nint_list'][args["current_cvg_lvl"]]            ,
        args['mass']            ,
        args['loopnb']          ,
        args['Targets']         ,
        args['MassSum']         ,
        args['SpaceRotsUn']     ,
        args['TimeRevsUn']      ,
        args['TimeShiftNumUn']  ,
        args['TimeShiftDenUn']  ,
        args['loopnbi']         ,
        args['ProdMassSumAll']  ,
        args['SpaceRotsBin']    ,
        args['TimeRevsBin']     ,
        args['TimeShiftNumBin'] ,
        args['TimeShiftDenBin'] ,
        all_coeffs
        )
    
    # ~ print("Compare SIZES in ESCAPE")
    # ~ print(abs(max_loop_size-res[0])/max_loop_size , abs(max_loop_dist-res[1])/max_loop_dist )
    
    # ~ print(max_loop_size,max_loop_dist)
    # ~ print(res)
    
    # ~ print("")
    # ~ print("")
    # ~ print("")
    
    # ~ return (max_loop_dist > (4.5 * callfun[0]['nbody'] * max_loop_size))
    return (res[1] > (4.5 * callfun[0]['nbody'] * res[0])),res
    
def Compute_MinDist(x,callfun):
    # Returns the minimum inter-body distance along a set of trajectories
    
    args=callfun[0]
    
    y = args['param_to_coeff_list'][args["current_cvg_lvl"]] * x
    all_coeffs = y.reshape(args['nloop'],ndim,args['ncoeff_list'][args["current_cvg_lvl"]],2)
    
    MinDist =  Compute_MinDist_Cython(
        args['nloop']           ,
        args['ncoeff_list'][args["current_cvg_lvl"]]          ,
        args['nint_list'][args["current_cvg_lvl"]]            ,
        args['mass']            ,
        args['loopnb']          ,
        args['Targets']         ,
        args['MassSum']         ,
        args['SpaceRotsUn']     ,
        args['TimeRevsUn']      ,
        args['TimeShiftNumUn']  ,
        args['TimeShiftDenUn']  ,
        args['loopnbi']         ,
        args['ProdMassSumAll']  ,
        args['SpaceRotsBin']    ,
        args['TimeRevsBin']     ,
        args['TimeShiftNumBin'] ,
        args['TimeShiftDenBin'] ,
        all_coeffs
        )
    
    return MinDist
    
def Write_Descriptor(x,callfun,filename):
    # Dumps a text file describing the current trajectories
    
    args = callfun[0]
    
    with open(filename,'w') as filename_write:
        
        filename_write.write('Number of bodies : {:d}\n'.format(args['nbody']))
        
        # ~ filename_write.write('Number of loops : {:d}\n'.format(args['nloop']))
        
        filename_write.write('Mass of those bodies : ')
        for il in range(args['nbody']):
            filename_write.write(' {:f}'.format(args['mass'][il]))
        filename_write.write('\n')
        
        filename_write.write('Number of Fourier coefficients in each loop : {:d}\n'.format(args['ncoeff_list'][args["current_cvg_lvl"]]))
        filename_write.write('Number of integration points for the action : {:d}\n'.format(args['nint_list'][args["current_cvg_lvl"]]))
        
        Action,Gradaction = Compute_action(x,callfun)
        
        filename_write.write('Value of the Action : {:.10f}\n'.format(Action))
        filename_write.write('Value of the Norm of the Gradient of the Action : {:.10E}\n'.format(np.linalg.norm(Gradaction)))

        Newt_err = Compute_Newton_err(x,callfun)
        Newt_err_norm = np.linalg.norm(Newt_err)/args['nint_list'][args["current_cvg_lvl"]]
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
         
def SelectFiles_Action(store_folder,hash_dict,Action_val=0,Action_Hash_val=np.zeros((nhash)),rtol=1e-5):
    # Creates a list of possible duplicates based on value of the action and hashes
    
    Action_msg = 'Value of the Action : '
    Action_msg_len = len(Action_msg)
    
    Action_Hash_msg = 'Hash Action for duplicate detection : '
    Action_Hash_msg_len = len(Action_Hash_msg)
    
    file_path_list = []
    for file_path in os.listdir(store_folder):
        file_path = os.path.join(store_folder, file_path)
        file_root, file_ext = os.path.splitext(os.path.basename(file_path))
        
        if (file_ext == '.txt' ):
            
            Saved_Action = hash_dict.get(file_root)
            
            if (Saved_Action is None) :
            
                # ~ print(file_path)
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
                IsCandidate = (IsCandidate and (abs(This_Action_Hash[ihash]-Action_Hash_val[ihash]) < (abs(This_Action_Hash[ihash])+abs(Action_Hash_val[ihash]))*rtol))
            
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

def Init_deflation(callfun,defl_cvg_lvl = 0):
    
    callfun[0]['defl_vec_list'] = []
    callfun[0]['defl_cvg_lvl'] = defl_cvg_lvl
    
def Add_deflation_coeffs(all_coeffs,callfun):
    
    all_coeffs_new = np.zeros((callfun[0]['nloop'],ndim,callfun[0]['ncoeff_list'][callfun[0]['defl_cvg_lvl']],2),dtype=np.float64)

    ncoeff_min = min(all_coeffs_new.shape[2],all_coeffs.shape[2])
    
    for k in range(ncoeff_min):
        all_coeffs_new[:,:,k,:] = all_coeffs[:,:,k,:]  

    y = all_coeffs_new.reshape(-1)
    callfun[0]['defl_vec_list'].append(callfun[0]['coeff_to_param_list'][callfun[0]["defl_cvg_lvl"]].dot(y))

def Load_all_defl(dirname,callfun):
    
    for file_path in os.listdir(dirname):
        file_path = os.path.join(dirname, file_path)
        file_root, file_ext = os.path.splitext(os.path.basename(file_path))
        
        if (file_ext == '.npy' ):
            
            all_coeffs = np.load(file_path)
            Add_deflation_coeffs(all_coeffs,callfun)

def Compute_action_defl(x,callfun):
    # Wrapper function that returns a deflated version of the gradient of the action.
    
    J,y = Compute_action(x,callfun)
    
    kfac = Compute_defl_fac(x,callfun)
    
    y = y*kfac
    
    return y

def Compute_defl_fac(x,callfun):
    
    kfac = 1.
    for defl_vec in callfun[0]['defl_vec_list']:
        
        dist2 = Compute_square_dist(defl_vec,x,callfun[0]['coeff_to_param_list'][callfun[0]["defl_cvg_lvl"]].shape[0])
        
        kfac*= dist2 * 50
        
        
    kfac = 1. / np.sqrt(kfac) + 1. 
    
    return kfac

class UniformRandom():
    def __init__(self, d):
        self.d = d

    def random(self):
        return np.random.random((self.d))

def Transform_Coeffs(SpaceRots, TimeRevs, TimeShiftNum, TimeShiftDen, all_coeffs):
    # Transforms coeffs defining a path and returns updated coeffs
    
    nloop = all_coeffs.shape[0]
    ncoeff = all_coeffs.shape[2]
        
    cs = np.zeros((2))
    all_coeffs_new = np.zeros(all_coeffs.shape)

    for il in range(nloop):
        for k in range(ncoeff):
            
            dt = TimeShiftNum[il] / TimeShiftDen[il]
            cs[0] = m.cos( - twopi * k*dt)
            cs[1] = m.sin( - twopi * k*dt)  
                
            v = all_coeffs[il,:,k,0] * cs[0] - TimeRevs[il] * all_coeffs[il,:,k,1] * cs[1]
            w = all_coeffs[il,:,k,0] * cs[1] + TimeRevs[il] * all_coeffs[il,:,k,1] * cs[0]
                
            all_coeffs_new[il,:,k,0] = SpaceRots[il,:,:].dot(v)
            all_coeffs_new[il,:,k,1] = SpaceRots[il,:,:].dot(w)
        
    return all_coeffs_new

def Compose_Two_Paths(nTf,nbs,nbf,ncoeff,all_coeffs_slow,all_coeffs_fast,Rotate_fast_with_slow=True):
    # Composes a "slow" with a "fast" path
    
    k_fac_slow = nbf
    k_fac_fast = nTf
    
    phys_exp = 2*(1-n)

    rfac_slow = (k_fac_slow)**(-1./phys_exp)
    rfac_fast = (k_fac_fast)**(-2./phys_exp)
    
    ncoeff_slow = all_coeffs_slow.shape[2]
    ncoeff_fast = all_coeffs_fast.shape[2]
    
    if (all_coeffs_slow.shape[0] != all_coeffs_fast.shape[0] ):
        raise ValueError("Fast and slow have different number of loops")
    else:
        nloop = all_coeffs_slow.shape[0]
    
    all_coeffs_slow_mod = np.zeros((nloop,ndim,ncoeff,2),dtype=np.float64)
    all_coeffs_fast_mod = np.zeros((nloop,ndim,ncoeff,2),dtype=np.float64)
        
    for il in range(nloop):
        for idim in range(ndim):
            for k in range(1,min(ncoeff//k_fac_slow,ncoeff_slow)):
                
                all_coeffs_slow_mod[il,idim,k*k_fac_slow,:]  = rfac_slow * all_coeffs_slow[il,idim,k,:]

    for il in range(nloop):
        for idim in range(ndim):
            for k in range(1,min(ncoeff//k_fac_fast,ncoeff_fast)):
                
                all_coeffs_fast_mod[il,idim,k*k_fac_fast,:]  = rfac_fast * all_coeffs_fast[il,idim,k,:]
    

    if Rotate_fast_with_slow :
        
        nint = 2*ncoeff

        c_coeffs_slow = all_coeffs_slow_mod.view(dtype=np.complex128)[...,0]
        all_pos_slow = np.fft.irfft(c_coeffs_slow,n=nint,axis=2)

        c_coeffs_fast = all_coeffs_fast_mod.view(dtype=np.complex128)[...,0]
        all_pos_fast = np.fft.irfft(c_coeffs_fast,n=nint,axis=2)

        all_coeffs_slow_mod_speed = np.zeros((nloop,ndim,ncoeff,2),dtype=np.float64)

        for il in range(nloop):
            for idim in range(ndim):
                for k in range(ncoeff):

                    all_coeffs_slow_mod_speed[il,idim,k,0] = k * all_coeffs_slow_mod[il,idim,k,1] 
                    all_coeffs_slow_mod_speed[il,idim,k,1] = -k * all_coeffs_slow_mod[il,idim,k,0] 
                
        c_coeffs_slow_mod_speed = all_coeffs_slow_mod_speed.view(dtype=np.complex128)[...,0]
        all_pos_slow_mod_speed = np.fft.irfft(c_coeffs_slow_mod_speed,n=nint,axis=2)
        
        all_pos_avg = np.zeros((nloop,ndim,nint),dtype=np.float64)

        for il in range(nloop):
            for iint in range(nint):
                
                v = all_pos_slow_mod_speed[il,:,iint]
                v = v / np.linalg.norm(v)

                SpRotMat = np.array( [[v[0] , -v[1]] , [v[1],v[0]]])
                # ~ SpRotMat = np.array( [[v[0] , v[1]] , [-v[1],v[0]]])
                
                all_pos_avg[il,:,iint] = all_pos_slow[il,:,iint] + SpRotMat.dot(all_pos_fast[il,:,iint])

        c_coeffs_avg = np.fft.rfft(all_pos_avg,n=nint,axis=2)
        all_coeffs_composed = np.zeros((nloop,ndim,ncoeff,2),dtype=np.float64)

        for il in range(nloop):
            for idim in range(ndim):
                for k in range(min(ncoeff,ncoeff_slow)):
                    all_coeffs_composed[il,idim,k,0] = c_coeffs_avg[il,idim,k].real
                    all_coeffs_composed[il,idim,k,1] = c_coeffs_avg[il,idim,k].imag
                    
                    
    else :
        
        all_coeffs_composed = all_coeffs_fast_mod[:,:,0:ncoeff,:]  + all_coeffs_slow_mod[:,:,0:ncoeff,:] 

    return all_coeffs_composed
