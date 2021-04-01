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

import fractions

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import cnames
from matplotlib import animation

from Choreo_cython_funs import *

def plot_all_2D(x,nint_plot,callfun,filename,fig_size=(10,10)):
    # Plots 2D trajectories and saves image under filename
    
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
    lines = sum([ax.plot([], [],'-', antialiased=True,zorder=-ib)  for ib in range(nbody)], [])
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

def Compute_action_onlygrad(x,callfun):
    # Wrapper function that returns ONLY the gradient of the action with respect to the parameters 
    
    J,y = Compute_action(x,callfun)
    
    return y
    
def Compute_action_hess_mul(x,dx,callfun):
    # Returns the Hessian of the action (computed wrt the parameters) times a test vector of parameter deviations.
    
    args=callfun[0]
    
    y = args['param_to_coeff_list'][args["current_cvg_lvl"]] * x
    all_coeffs = y.reshape(args['nloop'],ndim,args['ncoeff_list'][args["current_cvg_lvl"]],2)
    
    dy = args['param_to_coeff_list'][args["current_cvg_lvl"]] * dx
    all_coeffs_d = dy.reshape(args['nloop'],ndim,args['ncoeff_list'][args["current_cvg_lvl"]],2)
    
    HessJdx =  Compute_action_hess_mul_Cython(
        args['nloop']           ,
        args['ncoeff_list'][args["current_cvg_lvl"]]          ,
        args['nint_list']            ,
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
        all_coeffs_d            ,
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

    Q, R, E, rank = sparseqr.qr( AT )

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

    if (len(ib_list) != SymType['n']):
        print("Warning : SymType and LoopLength are inconsistent")
        
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
            param_to_coeff_list[i].data
            )
        
        diag_changevar(
            coeff_to_param_list[i].nnz,
            ncoeff_list[i],
            n_grad_change,
            coeff_to_param_list[i].col,
            coeff_to_param_list[i].data
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
    }]

    return callfun
    
def Compute_action(x,callfun):
    # Cumputes the action and its gradient with respect to the parameters at a given value of the parameters

    args=callfun[0]
    
    y = args['param_to_coeff_list'][args["current_cvg_lvl"]] * x
    all_coeffs = y.reshape(args['nloop'],ndim,args['ncoeff_list'][args["current_cvg_lvl"]],2)
    
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
        all_coeffs
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
    
def Compute_Loop_Dist_Size(x,callfun):
    # Computes sizes of trajetories and distance between center of trajectories
    # Useful to detect escape.
    
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
    
    return max_loop_dist,max_loop_size
    
def Detect_Escape(x,callfun):
    # Returns True if the trajectories are so far that they are likely to never interact again
    
    max_loop_dist,max_loop_size = Compute_Loop_Dist_Size(x,callfun)
    
    return (max_loop_dist > (4.5 * callfun[0]['nbody'] * max_loop_size))
    
def Compute_MinDist(x,callfun):
    # Returns the minimum inter-body distance along a set of trajecctories
    
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
            filename_write.write(' {:.10f}'.format(Hash_Action[ihash]))
        filename_write.write('\n')
        

def SelectFiles_Action(store_folder,Action_val,Action_Hash_val,rtol):
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
                        
                IsCandidate = (abs(This_Action-Action_val) < ((abs(This_Action)+abs(Action_val))*rtol))
                for ihash in range(nhash):
                    IsCandidate = (IsCandidate and (abs(This_Action_Hash[ihash]-Action_Hash_val[ihash]) < (abs(This_Action_Hash[ihash])+abs(Action_Hash_val[ihash]))*rtol))
                
                if IsCandidate:
                    
                    file_path_list.append(store_folder+'/'+file_root)
                    
    return file_path_list

def Check_Duplicates(x,callfun,store_folder,duplicate_eps,rtol=1e-5):
    # Checks whether there is a duplicate of a given trajecory in the provided folder

    Action,Gradaction = Compute_action(x,callfun)
    Hash_Action = Compute_hash_action(x,callfun)

    file_path_list = SelectFiles_Action(store_folder,Action,Hash_Action,rtol)
    
    if (len(file_path_list) == 0):
        
        Found_duplicate = False
        file_path = ''
    
    else:
        Found_duplicate = True
        file_path = file_path_list[0]
    
    return Found_duplicate,file_path
