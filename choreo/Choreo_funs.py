'''
Choreo_funs.py : Defines useful functions in the Choreographies2 project.

'''

import os
import itertools
import copy
import time
import pickle
import warnings

import numpy as np
import math as m
import scipy.fft
import scipy.optimize
import scipy.linalg as la
import scipy.sparse as sp
import sparseqr
import networkx as nx
import random

import inspect

import fractions

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import cnames
from matplotlib.collections import LineCollection
from matplotlib import animation

from choreo.Choreo_cython_funs import ndim,twopi,nhash,n
from choreo.Choreo_cython_funs import Compute_action_Cython,Compute_action_hess_mul_Cython,Compute_hash_action_Cython,Compute_Newton_err_Cython
from choreo.Choreo_cython_funs import Assemble_Cstr_Matrix,diag_changevar
from choreo.Choreo_cython_funs import Compute_MinDist_Cython,Compute_Loop_Dist_btw_avg_Cython,Compute_square_dist,Compute_Loop_Size_Dist_Cython
from choreo.Choreo_cython_funs import Compute_Forces_Cython,Compute_JacMat_Forces_Cython,Compute_JacMul_Forces_Cython
from choreo.Choreo_cython_funs import the_irfft,the_rfft

from choreo.Choreo_scipy_plus import *

def Pick_Named_Args_From_Dict(fun,the_dict,MissingArgsAreNone = True):
    
    # list_of_args = inspect.getargspec(fun).args
    list_of_args = inspect.getfullargspec(fun).args
    
    if MissingArgsAreNone:
        
        all_kwargs = {k:the_dict.get(k,None) for k in list_of_args}
        
    else:
        
        all_kwargs = {k:the_dict[k] for k in list_of_args}
    
    return all_kwargs

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
    
    y = args['param_to_coeff_list'][args["current_cvg_lvl"]].dot(x)
    all_coeffs = y.reshape(args['nloop'],ndim,args['ncoeff_list'][args["current_cvg_lvl"]],2)
    
    return all_coeffs

def RemoveSym(x,callfun):
    # Removes symmetries and gives coeffs for all bodies

    all_coeffs = Unpackage_all_coeffs(x,callfun)

    args = callfun[0]
    nbody = args['nbody']
    nloop = args['nloop']
    ncoeff = args['ncoeff_list'][args["current_cvg_lvl"]]
    loopnb = args['loopnb']

    all_coeffs_nosym = np.zeros((nbody,ndim,ncoeff,2),dtype=np.float64)

    for il in range(nloop):
        for ib in range(loopnb[il]):

            ibody = args['Targets'][il,ib]

            SpaceRot = args['SpaceRotsUn'][il,ib,:,:]
            TimeRev = args['TimeRevsUn'][il,ib]
            TimeShiftNum = args['TimeShiftNumUn'][il,ib]
            TimeShiftDen = args['TimeShiftDenUn'][il,ib]

            all_coeffs_nosym[ibody,:,:,:] = Transform_Coeffs_Single_Loop(SpaceRot, TimeRev, TimeShiftNum, TimeShiftDen, all_coeffs[il,:,:])

    return all_coeffs_nosym

def ComputeAllPos(x,callfun,nint=None):
    # Returns the positions of all bodies.

    if nint is None:
        args=callfun[0]
        nint = args['nint_list'][args["current_cvg_lvl"]]

    all_coeffs_nosym = RemoveSym(x,callfun).view(dtype=np.complex128)[...,0]
    all_pos_b = the_irfft(all_coeffs_nosym,n=nint,axis=2,norm="forward")

    return all_pos_b

def ComputeAllLoopPos(x,callfun,nint=None):
    # Returns the positions of all loops, not bodies.

    if nint is None:
        args=callfun[0]
        nint = args['nint_list'][args["current_cvg_lvl"]]

    all_coeffs_c = Unpackage_all_coeffs(x,callfun).view(dtype=np.complex128)[...,0]
    all_pos = the_irfft(all_coeffs_c,n=nint,axis=2,norm="forward")

    return all_pos

def ComputeAllPosVel(x,callfun,nint=None):
    # Returns the positions and velocities of all bodies along the path.

    if nint is None:
        args=callfun[0]
        nint = args['nint_list'][args["current_cvg_lvl"]]

    all_coeffs_nosym = RemoveSym(x,callfun).view(dtype=np.complex128)[...,0]
    all_pos_b = the_irfft(all_coeffs_nosym,n=nint,axis=2,norm="forward")

    ncoeff = all_coeffs_nosym.shape[2]
    for k in range(ncoeff):
        all_coeffs_nosym[:,:,k] *= twopi*1j*k

    all_vel_b = the_irfft(all_coeffs_nosym,n=nint,axis=2,norm="forward")

    return np.stack((all_pos_b,all_vel_b),axis=0)

def Compute_xlim(x,callfun,extend=0.):

    all_pos_b = ComputeAllPos(x,callfun)

    xmin = np.amin(all_pos_b,axis=(0,2))
    xmax = np.amax(all_pos_b,axis=(0,2))

    xmin -= extend*(xmax-xmin)
    xmax += extend*(xmax-xmin)

    return np.stack((xmin,xmax),axis=1).reshape(-1)

def Compute_init_pos_vel(x,callfun):
    # I litterally do not know of any more efficient way to compute the initial positions and velocities.

    all_pos_vel = ComputeAllPosVel(x,callfun)

    return np.ascontiguousarray(all_pos_vel[:,:,:,0])

def Compute_action_onlygrad(x,callfun):
    # Wrapper function that returns ONLY the gradient of the action with respect to the parameters 
    
    J,y = Compute_action(x,callfun)
    
    return y
    
def Compute_action_onlygrad_escape(x,callfun):
    # Cumputes the action and its gradient with respect to the parameters at a given value of the parameters

    args=callfun[0]
    
    y = args['param_to_coeff_list'][args["current_cvg_lvl"]].dot(x)
    all_coeffs = y.reshape(args['nloop'],ndim,args['ncoeff_list'][args["current_cvg_lvl"]],2)

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
    
    # print("escape_pen = ",escape_pen)

    
    if args["Do_Pos_FFT"]:
        
        y = args['param_to_coeff_list'][args["current_cvg_lvl"]].dot(x)
        args['last_all_coeffs'] = y.reshape(args['nloop'],ndim,args['ncoeff_list'][args["current_cvg_lvl"]],2)
        
        nint = args['nint_list'][args["current_cvg_lvl"]]
        c_coeffs = args['last_all_coeffs'].view(dtype=np.complex128)[...,0]
        args['last_all_pos'] = the_irfft(c_coeffs,n=nint,axis=2,norm="forward")

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
    GJparam = (args['param_to_coeff_T_list'][args["current_cvg_lvl"]].dot(GJ)) * escape_pen
    
    return GJparam
    
def Compute_action_hess_mul(x,dx,callfun):
    # Returns the Hessian of the action (computed wrt the parameters) times a test vector of parameter deviations.
    
    args=callfun[0]

    dy = args['param_to_coeff_list'][args["current_cvg_lvl"]].dot(dx)
    all_coeffs_d = dy.reshape(args['nloop'],ndim,args['ncoeff_list'][args["current_cvg_lvl"]],2)
    
    if args["Do_Pos_FFT"]:
        
        y = args['param_to_coeff_list'][args["current_cvg_lvl"]].dot(x)
        args['last_all_coeffs'] = y.reshape(args['nloop'],ndim,args['ncoeff_list'][args["current_cvg_lvl"]],2)
        
        nint = args['nint_list'][args["current_cvg_lvl"]]
        c_coeffs = args['last_all_coeffs'].view(dtype=np.complex128)[...,0]
        args['last_all_pos'] = the_irfft(c_coeffs,n=nint,axis=2,norm="forward")

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
    
    z =  args['param_to_coeff_T_list'][args["current_cvg_lvl"]].dot(HJdx)
    
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

    # tolerance = 1e-5
    tolerance = None

    Q, R, E, rank = sparseqr.qr( AT, tolerance=tolerance )

    nrow = AT.shape[0]
    
    if (nrow <= rank):
        # raise ValueError("Kernel is empty")
        
        return sp.coo_matrix(([],([],[])),shape=(nrow,0))
    
    else:

        mask = []
        iker = 0
        while (iker < Q.nnz):
            if (Q.col[iker] >= rank):
                mask.append(iker)
            iker += 1
            
        return sp.coo_matrix((Q.data[mask],(Q.row[mask],Q.col[mask]-rank)),shape=(nrow,nrow-rank))
     
class ChoreoSym():
    r"""
    This class defines the symmetries of the action
    Useful to detect loops and constraints.

    Syntax : Giving one ChoreoSym to setup_changevar prescribes the following symmetry / constraint :

    .. math::
        x_{\text{LoopTarget}}(t) = \text{SpaceRot} \cdot x_{\text{LoopSource}} (\text{TimeRev} * (t - \text{TimeShift}))

    Where SpaceRot is assumed orthogonal (never actually checked, so beware)
    and TimeShift is defined as a rational fraction.
    """

    def __init__(
        self,
        LoopTarget=0,
        LoopSource=0,
        SpaceRot=np.identity(ndim,dtype=np.float64),
        TimeRev=1,
        TimeShift=fractions.Fraction(numerator=0,denominator=1)
        ):
        r"""
        Class constructor
        """

        self.LoopTarget = LoopTarget
        self.LoopSource = LoopSource
        self.SpaceRot = SpaceRot
        self.TimeRev = TimeRev
        self.TimeShift = TimeShift
        
    def Inverse(self):
        r"""
        Returns the inverse of a symmetry transformation
        """

        return ChoreoSym(
            LoopTarget=self.LoopSource,
            LoopSource=self.LoopTarget,
            SpaceRot = self.SpaceRot.transpose(),
            TimeRev = self.TimeRev,         
            TimeShift = fractions.Fraction(numerator=((-int(self.TimeRev)*self.TimeShift.numerator) % self.TimeShift.denominator),denominator=self.TimeShift.denominator)
            )

    def ComposeLight(B,A):
        r"""
        Returns the composition of two transformations ignoring sources and targets.

        B.ComposeLight(A) returens the composition B o A, i.e. applies A then B, ignoring that target A might be different from source B.
        """
        
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
        r"""
        Returns the composition of two transformations.
        """
        # Composition B o A, i.e. applies A then B
        
        if (A.LoopTarget == B.LoopSource):
            return B.ComposeLight(A)
            
        else:
            
            print(B.LoopTarget,B.LoopSource)
            print(A.LoopTarget,A.LoopSource)
            
            raise ValueError("Symmetries cannot be composed")

    def IsIdentity(self,atol = 1e-10):
        r"""
        Returns True if the transformation is close to identity.
        """        

        if ((abs(self.TimeShift % 1) < atol) and (self.TimeRev == 1) and (self.LoopTarget == self.LoopSource)):
            
            return np.allclose(self.SpaceRot,np.identity(ndim,dtype=np.float64),rtol=0.,atol=atol)
            
        else:
        
            return False
            
    def IsSame(self,other):
        r"""
        Returns True if the two transformations are almost identical.
        """   
        return ((self.Inverse()).ComposeLight(other)).IsIdentity()

def setup_changevar(nbody,ncoeff_init,mass,n_reconverge_it_max=6,MomCons=True,n_grad_change=1.,Sym_list=[],CrashOnIdentity=True):
    # This function returns the callfun dictionnary to be given as input to virtually all other function.
    # It detects loops and constraints based on symmetries.
    # It defines parameters according to given constraints and diagonal change of variable
    # It computes useful objects to optimize the computation of the action :
    #  - Exhaustive list of unary transformation for generator to body
    #  - Exhaustive list of binary transformations from generator within each loop.
    
    Identity_detected = False

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

                    if CrashOnIdentity:
                        raise ValueError("Two bodies have identical trajectories")
                    else:
                        if not(Identity_detected):
                            print("Two bodies have identical trajectories")
                            # warnings.warn("Two bodies have identical trajectories", stacklevel=2)
                        
                    Identity_detected = True

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

    # Count how many unique paths need to be displayed
    RequiresLoopDispUn = np.zeros((nloop,maxlooplen),dtype=bool)

    eps_rot = 1e-10
    for il in range(nloop):

        loop_rots = []

        for ib in range(loopnb[il]): 

            Add_to_loop_rots = True

            for irot in range(len(loop_rots)):
                    
                dist_ij = np.linalg.norm(SpaceRotsUn[il,ib,:,:] - loop_rots[irot])

                Add_to_loop_rots = (Add_to_loop_rots and (dist_ij > eps_rot))

            RequiresLoopDispUn[il,ib] = Add_to_loop_rots

            if Add_to_loop_rots:

                loop_rots.append(SpaceRotsUn[il,ib,:,:])


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

    param_to_coeff_T_list = []
    coeff_to_param_T_list = []

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

        # TODO : THIS IS PROBABLY WHY I HAVE CONDITIONNING ISSUES FOR DIFFERENT MASSES !!!

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

        param_to_coeff_T_list.append(param_to_coeff_list[i].transpose(copy=True))
        coeff_to_param_T_list.append(coeff_to_param_list[i].transpose(copy=True))


    callfun = [{
    "nbody"                 :   nbody                   ,
    "nloop"                 :   nloop                   ,
    "mass"                  :   mass                    ,
    "loopnb"                :   loopnb                  ,
    "loopgen"               :   loopgen                 ,
    "Targets"               :   Targets                 ,
    "MassSum"               :   MassSum                 ,
    "SpaceRotsUn"           :   SpaceRotsUn             ,
    "TimeRevsUn"            :   TimeRevsUn              ,
    "TimeShiftNumUn"        :   TimeShiftNumUn          ,
    "TimeShiftDenUn"        :   TimeShiftDenUn          ,
    "RequiresLoopDispUn"    :   RequiresLoopDispUn      ,
    "loopnbi"               :   loopnbi                 ,
    "ProdMassSumAll"        :   ProdMassSumAll          ,
    "SpaceRotsBin"          :   SpaceRotsBin            ,
    "TimeRevsBin"           :   TimeRevsBin             ,
    "TimeShiftNumBin"       :   TimeShiftNumBin         ,
    "TimeShiftDenBin"       :   TimeShiftDenBin         ,
    "ncoeff_list"           :   ncoeff_list             ,
    "nint_list"             :   nint_list               ,
    "param_to_coeff_list"   :   param_to_coeff_list     ,
    "coeff_to_param_list"   :   coeff_to_param_list     ,
    "param_to_coeff_T_list" :   param_to_coeff_T_list   ,
    "coeff_to_param_T_list" :   coeff_to_param_T_list   ,
    "current_cvg_lvl"       :   0                       ,
    "last_all_coeffs"       :   None                    ,
    "last_all_pos"          :   None                    ,
    "Do_Pos_FFT"            :   True                    ,
    }]

    return callfun
    
def Compute_action(x,callfun):
    # Computes the action and its gradient with respect to the parameters at a given value of the parameters

    args=callfun[0]

    if args["Do_Pos_FFT"]:
        
        y = args['param_to_coeff_list'][args["current_cvg_lvl"]].dot(x)
        args['last_all_coeffs'] = y.reshape(args['nloop'],ndim,args['ncoeff_list'][args["current_cvg_lvl"]],2)
        
        nint = args['nint_list'][args["current_cvg_lvl"]]
        c_coeffs = args['last_all_coeffs'].view(dtype=np.complex128)[...,0]
        args['last_all_pos'] = the_irfft(c_coeffs,n=nint,axis=2,norm="forward")

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
    y = args['param_to_coeff_T_list'][args["current_cvg_lvl"]].dot(GJ)
    
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
    
    y = args['param_to_coeff_list'][args["current_cvg_lvl"]].dot(x)
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
    
    args=callfun[0]
    
    y = args['param_to_coeff_list'][args["current_cvg_lvl"]].dot(x)
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
    
    # return (max_loop_dist > (4.5 * callfun[0]['nbody'] * max_loop_size))
    return (res[1] > (4.5 * callfun[0]['nbody'] * res[0])),res
    
def Compute_MinDist(x,callfun):
    # Returns the minimum inter-body distance along a set of trajectories
    
    args=callfun[0]
    
    y = args['param_to_coeff_list'][args["current_cvg_lvl"]].dot(x)
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

def Compute_MaxPathLength(x,callfun):
    # Computes the maximum path length for speed sync

    args=callfun[0]

    nint = args['nint_list'][args["current_cvg_lvl"]]

    if args["Do_Pos_FFT"]:
        
        y = args['param_to_coeff_list'][args["current_cvg_lvl"]].dot(x)
        args['last_all_coeffs'] = y.reshape(args['nloop'],ndim,args['ncoeff_list'][args["current_cvg_lvl"]],2)
        
        c_coeffs = args['last_all_coeffs'].view(dtype=np.complex128)[...,0]
        args['last_all_pos'] = the_irfft(c_coeffs,n=nint,axis=2,norm="forward")

    dx = args['last_all_pos'].copy()
    dx[:,:,0:(nint-1)] -= args['last_all_pos'][:,:,1:nint]
    dx[:,:,nint-1] -= args['last_all_pos'][:,:,0]
    
    max_path_length = np.linalg.norm(dx,axis=1).sum(axis=1).max(axis=0)

    return max_path_length

class UniformRandom():
    def __init__(self, d):
        self.d = d
        self.rdn = np.random.RandomState(np.int64(time.time_ns()) % np.int64(2**32))

    def random(self):
        return self.rdn.random_sample((self.d))

def Transform_Coeffs_Single_Loop(SpaceRot, TimeRev, TimeShiftNum, TimeShiftDen, one_loop_coeffs):
    # Transforms coeffs defining a single loop and returns updated coeffs
    
    ncoeff = one_loop_coeffs.shape[1]
        
    cs = np.zeros((2))
    all_coeffs_new = np.zeros(one_loop_coeffs.shape)

    for k in range(ncoeff):
        
        dt = TimeShiftNum / TimeShiftDen
        cs[0] = m.cos( - twopi * k*dt)
        cs[1] = m.sin( - twopi * k*dt)  
            
        v = one_loop_coeffs[:,k,0] * cs[0] - TimeRev * one_loop_coeffs[:,k,1] * cs[1]
        w = one_loop_coeffs[:,k,0] * cs[1] + TimeRev * one_loop_coeffs[:,k,1] * cs[0]
            
        all_coeffs_new[:,k,0] = SpaceRot[:,:].dot(v)
        all_coeffs_new[:,k,1] = SpaceRot[:,:].dot(w)
        
    return all_coeffs_new

def Transform_Coeffs(SpaceRot, TimeRev, TimeShiftNum, TimeShiftDen, all_coeffs):
    # Transforms coeffs defining a path and returns updated coeffs
    
    nloop = all_coeffs.shape[0]
    ncoeff = all_coeffs.shape[2]
        
    cs = np.zeros((2))
    all_coeffs_new = np.zeros(all_coeffs.shape)

    for il in range(nloop):
        for k in range(ncoeff):
            
            dt = TimeShiftNum / TimeShiftDen
            cs[0] = m.cos( - twopi * k*dt)
            cs[1] = m.sin( - twopi * k*dt)  
                
            v = all_coeffs[il,:,k,0] * cs[0] - TimeRev * all_coeffs[il,:,k,1] * cs[1]
            w = all_coeffs[il,:,k,0] * cs[1] + TimeRev * all_coeffs[il,:,k,1] * cs[0]
                
            all_coeffs_new[il,:,k,0] = SpaceRot[:,:].dot(v)
            all_coeffs_new[il,:,k,1] = SpaceRot[:,:].dot(w)
        
    return all_coeffs_new

def Compose_Two_Paths(nTf,nbs,nbf,mass_mul,ncoeff,all_coeffs_slow,all_coeffs_fast_list,Rotate_fast_with_slow=False,mul_loops=None):
    # Composes a "slow" with a "fast" path

    nloop_slow = all_coeffs_slow.shape[0]
    nloop_fast = len(all_coeffs_fast_list)

    if (nloop_slow != nloop_fast):
        raise ValueError("Fast and slow have different number of loops")

    nloop = nloop_slow

    all_coeffs_composed_list = []

    for ils in range(nloop):

        all_coeffs_fast = all_coeffs_fast_list[ils]

        nloop_fast = all_coeffs_fast.shape[0]

        if (mul_loops[ils]):

            k_fac_slow = 1
            k_fac_fast = nTf[ils]
            
            phys_exp = 2*(1-n)

            rfac_slow = (k_fac_slow/(mass_mul[ils]*nbf[ils]))**(-1./phys_exp)
            rfac_fast = (k_fac_fast)**(-2./phys_exp)
        
        else:

            k_fac_slow = nbf[ils]
            k_fac_fast = nTf[ils]
            
            phys_exp = 2*(1-n)

            rfac_slow = (k_fac_slow/mass_mul[ils])**(-1./phys_exp)
            rfac_fast = (k_fac_fast/m.sqrt(mass_mul[ils]))**(-2./phys_exp)

        # print(k_fac_slow,k_fac_fast,rfac_slow,rfac_fast)

        ncoeff_slow = all_coeffs_slow.shape[2]
        ncoeff_fast = all_coeffs_fast.shape[2]
        
        all_coeffs_slow_mod = np.zeros((1         ,ndim,ncoeff,2),dtype=np.float64)
        all_coeffs_fast_mod = np.zeros((nloop_fast,ndim,ncoeff,2),dtype=np.float64)
        
        for idim in range(ndim):
            for k in range(min(ncoeff//k_fac_slow,ncoeff_slow)):
                
                all_coeffs_slow_mod[0,idim,k*k_fac_slow,:]  = rfac_slow * all_coeffs_slow[ils,idim,k,:]

        for ilf in range(nloop_fast):
            for idim in range(ndim):
                for k in range(1,min(ncoeff//k_fac_fast,ncoeff_fast)):
                    
                    all_coeffs_fast_mod[ilf,idim,k*k_fac_fast,:]  = rfac_fast * all_coeffs_fast[ilf,idim,k,:]
        
        if Rotate_fast_with_slow :
            
            nint = 2*ncoeff

            c_coeffs_slow = all_coeffs_slow_mod.view(dtype=np.complex128)[...,0]
            all_pos_slow = the_irfft(c_coeffs_slow,n=nint,axis=2)

            c_coeffs_fast = all_coeffs_fast_mod.view(dtype=np.complex128)[...,0]
            all_pos_fast = the_irfft(c_coeffs_fast,n=nint,axis=2)

            all_coeffs_slow_mod_speed = np.zeros((1,ndim,ncoeff,2),dtype=np.float64)

            for idim in range(ndim):
                for k in range(ncoeff):

                    all_coeffs_slow_mod_speed[0,idim,k,0] = k * all_coeffs_slow_mod[0,idim,k,1] 
                    all_coeffs_slow_mod_speed[0,idim,k,1] = -k * all_coeffs_slow_mod[0,idim,k,0] 
                    
            c_coeffs_slow_mod_speed = all_coeffs_slow_mod_speed.view(dtype=np.complex128)[...,0]
            all_pos_slow_mod_speed = the_irfft(c_coeffs_slow_mod_speed,n=nint,axis=2)
            
            all_pos_avg = np.zeros((nloop_fast,ndim,nint),dtype=np.float64)

            for ilf in range(nloop_fast):
                for iint in range(nint):
                    
                    v = all_pos_slow_mod_speed[0,:,iint]
                    v = v / np.linalg.norm(v)

                    SpRotMat = np.array( [[v[0] , -v[1]] , [v[1],v[0]]])
                    
                    all_pos_avg[ilf,:,iint] = all_pos_slow[0,:,iint] + SpRotMat.dot(all_pos_fast[ilf,:,iint])

            ## TODO Check norm of FFT !! Might be wrong now after fft change
            c_coeffs_avg = the_rfft(all_pos_avg,n=nint,axis=2)
            all_coeffs_composed = np.zeros((nloop_fast,ndim,ncoeff,2),dtype=np.float64)

            for ilf in range(nloop_fast):
                for idim in range(ndim):
                    for k in range(min(ncoeff,ncoeff_slow)):
                        all_coeffs_composed[ilf,idim,k,0] = c_coeffs_avg[ilf,idim,k].real
                        all_coeffs_composed[ilf,idim,k,1] = c_coeffs_avg[ilf,idim,k].imag
                        
                        
        else :
            
            all_coeffs_composed = np.zeros((nloop_fast,ndim,ncoeff,2),dtype=np.float64)

            for ilf in range(nloop_fast):

                all_coeffs_composed[ilf,:,:,:] = all_coeffs_fast_mod[ilf,:,:,:]  + all_coeffs_slow_mod[0,:,:,:] 

        all_coeffs_composed_list.append(all_coeffs_composed)

    all_coeffs = np.concatenate(all_coeffs_composed_list,axis=0)

    return all_coeffs

def Gen_init_avg(nTf,nbs,nbf,mass_mul,ncoeff,all_coeffs_slow_load,all_coeffs_fast_load=None,all_coeffs_fast_load_list=None,callfun=None,Rotate_fast_with_slow=False,Optimize_Init=True,Randomize_Fast_Init=True,mul_loops=None):

    if (all_coeffs_fast_load_list is None):
        if (all_coeffs_fast_load is None):
            raise ValueError("all_coeffs fast not provided")
        else:
            all_coeffs_fast_load_list = [all_coeffs_fast_load]

    nloop_slow = all_coeffs_slow_load.shape[0]
    nloop_fast = len(all_coeffs_fast_load_list)

    if (nloop_slow != nloop_fast):
        raise ValueError("There should be the same number of slow and fast loops")

    nloop = nloop_slow


    if Randomize_Fast_Init :

        init_SpaceRevscal = np.array([1. if (np.random.random() > 1./2.) else -1. for ils in range(nloop)])
        init_TimeRevscal = np.array([1. if (np.random.random() > 1./2.) else -1. for ils in range(nloop)])
        Act_Mul = 1. if (np.random.random() > 1./2.) else -1.
        init_x = np.array([ np.random.random() for iparam in range(2*nloop)])

    else:

        init_SpaceRevscal = np.array([1. for ils in range(nloop)])
        init_TimeRevscal = np.array([1. for ils in range(nloop)])
        Act_Mul = 1.
        init_x = np.zeros((2*nloop))

    def params_to_coeffs(x):

        all_coeffs_fast_list = []

        for ils in range(nloop):

            theta = twopi * x[2*ils]
            SpaceRevscal = init_SpaceRevscal[ils]
            SpaceRots = np.array( [[SpaceRevscal*np.cos(theta) , SpaceRevscal*np.sin(theta)] , [-np.sin(theta),np.cos(theta)]])
            TimeRevs = init_TimeRevscal[ils]
            TimeShiftNum = x[2*ils+1]
            TimeShiftDen = 1

            all_coeffs_fast_list.append(Transform_Coeffs(SpaceRots, TimeRevs, TimeShiftNum, TimeShiftDen, all_coeffs_fast_load_list[ils]))
            # all_coeffs_fast_list.append(np.copy(all_coeffs_fast_load_list[ils]))

        all_coeffs_avg = Compose_Two_Paths(nTf,nbs,nbf,mass_mul,ncoeff,all_coeffs_slow_load,all_coeffs_fast_list,Rotate_fast_with_slow,mul_loops)

        return all_coeffs_avg

    if Optimize_Init :

        def params_to_Action(x):

            all_coeffs_avg = params_to_coeffs(x)

            x_avg = Package_all_coeffs(all_coeffs_avg,callfun)
            Act, GAct = Compute_action(x_avg,callfun)
            
            return Act_Mul * Act

        maxiter = 1000
        tol = 1e-10

        opt_result = scipy.optimize.minimize(fun=params_to_Action,x0=init_x,method='CG',options={'disp':False,'maxiter':maxiter,'gtol':tol},tol=tol)

        x_opt = opt_result['x']

        all_coeffs_avg = params_to_coeffs(x_opt)

    else:
        all_coeffs_avg = params_to_coeffs(init_x)


    return all_coeffs_avg

def Make_Init_bounds_coeffs(nloop,ncoeff,coeff_ampl_o=1e-1,k_infl=1,k_max=200,coeff_ampl_min=1e-16):

    all_coeffs_min = np.zeros((nloop,ndim,ncoeff,2),dtype=np.float64)
    all_coeffs_max = np.zeros((nloop,ndim,ncoeff,2),dtype=np.float64)

    randlimfac = 0.1
    # randlimfac = 0.
    
    try:

        coeff_slope = m.log(coeff_ampl_o/coeff_ampl_min)/(k_max-k_infl)

        for il in range(nloop):
            for idim in range(ndim):
                for k in range(ncoeff):

                    if (k <= k_infl):
                        randampl = coeff_ampl_o
                    else:
                        randampl = coeff_ampl_o * m.exp(-coeff_slope*(k-k_infl))

                    all_coeffs_min[il,idim,k,0] = -randampl* (1+random.random()*randlimfac)
                    all_coeffs_min[il,idim,k,1] = -randampl* (1+random.random()*randlimfac)
                    all_coeffs_max[il,idim,k,0] =  randampl* (1+random.random()*randlimfac)
                    all_coeffs_max[il,idim,k,1] =  randampl* (1+random.random()*randlimfac)
        
    except:

        print("An error occured during initial random coefficient bounds initialization.")
        print("Please check your Random parameters for consistency.")
        print("")

    return all_coeffs_min,all_coeffs_max

def Param_to_Param_direct(x,callfun_source,callfun_target):

    args_source=callfun_source[0]
    args_target=callfun_target[0]

    y = args_source['param_to_coeff_list'][args_source["current_cvg_lvl"]].dot(x)
    all_coeffs = y.reshape(args_source['nloop'],ndim,args_source['ncoeff_list'][args_source["current_cvg_lvl"]],2)
    
    if (args_target['ncoeff_list'][args_target["current_cvg_lvl"]] < args_source['ncoeff_list'][args_source["current_cvg_lvl"]]):
        z = all_coeffs[:,:,0:args_target['ncoeff_list'][args_target["current_cvg_lvl"]],:].reshape(-1)
    else:
        z = np.zeros((args_target['nloop'],ndim,args_target['ncoeff_list'][args_target["current_cvg_lvl"]],2))
        z[:,:,0:args_source['ncoeff_list'][args_source["current_cvg_lvl"]],:] = all_coeffs
        z = z.reshape(-1)

    res = args_target['coeff_to_param_list'][args_target["current_cvg_lvl"]].dot(z)
    
    return res

def Param_to_Param_rev(Gx,callfun_source,callfun_target):

    args_source=callfun_source[0]
    args_target=callfun_target[0]

    Gy = args_source['coeff_to_param_T_list'][args_source["current_cvg_lvl"]].dot(Gx)
    all_coeffs = Gy.reshape(args_source['nloop'],ndim,args_source['ncoeff_list'][args_source["current_cvg_lvl"]],2)

    if (args_target['ncoeff_list'][args_target["current_cvg_lvl"]] < args_source['ncoeff_list'][args_source["current_cvg_lvl"]]):
        Gz = all_coeffs[:,:,0:args_target['ncoeff_list'][args_target["current_cvg_lvl"]],:].reshape(-1)
    else:
        Gz = np.zeros((args_target['nloop'],ndim,args_target['ncoeff_list'][args_target["current_cvg_lvl"]],2))
        Gz[:,:,0:args_source['ncoeff_list'][args_source["current_cvg_lvl"]],:] = all_coeffs
        Gz = Gz.reshape(-1)
    
    
    res = args_target['param_to_coeff_T_list'][args_target["current_cvg_lvl"]].dot(Gz)
    
    return res

def Compute_Auto_ODE_RHS(x,callfun):

    args = callfun[0]

    all_pos_vel = x.reshape(2,args['nbody'],ndim)
    
    rhs = np.zeros((2,args['nbody'],ndim))

    rhs[0,:,:] = all_pos_vel[1,:,:]
    rhs[1,:,:] = Compute_Forces_Cython(
        all_pos_vel[0,:,:],
        args['mass'],
        args['nbody'],
        )

    return rhs.reshape(2*args['nbody']*ndim)

Compute_ODE_RHS = lambda t,x,callfun : Compute_Auto_ODE_RHS(x,callfun)

def GetSymplecticODEDef(callfun):

    args = callfun[0]

    def fun(t,v):
        return v

    def gun(t,x):
        return Compute_Forces_Cython(
            x.reshape(args['nbody'],ndim),
            args['mass'],
            args['nbody'],
            ).reshape(-1)

    return fun,gun

def Compute_Auto_JacMat_ODE_RHS(x,callfun):

    args = callfun[0]

    nbody = args['nbody']

    all_pos_vel = x.reshape(2,nbody,ndim)
    
    drhs = np.zeros((2,nbody,ndim,2,nbody,ndim))

    for ib in range(nbody):
        for idim in range(ndim):
            drhs[0,ib,idim,1,ib,idim] = 1

    drhs[1,:,:,0,:,:] = Compute_JacMat_Forces_Cython(
        all_pos_vel[0,:,:],
        args['mass'],
        nbody,
        )

    return drhs.reshape(2*nbody*ndim,2*nbody*ndim)
        
Compute_JacMat_ODE_RHS = lambda t,x,callfun : Compute_Auto_JacMat_ODE_RHS(x,callfun)

def Compute_Auto_JacMul_ODE_RHS(x,dx,callfun):

    args = callfun[0]

    nbody = args['nbody']

    all_pos_vel = x.reshape(2,nbody,ndim)
    all_pos_vel_d = dx.reshape(2,nbody,ndim)
    
    drhs = np.zeros((2,nbody,ndim))

    drhs[0,:,:] = all_pos_vel_d[1,:,:]

    drhs[1,:,:] = Compute_JacMul_Forces_Cython(
        all_pos_vel[0,:,:],
        all_pos_vel_d[0,:,:],
        args['mass'],
        nbody,
        )

    return drhs.reshape(2*nbody*ndim)

Compute_JacMul_ODE_RHS = lambda t,x,dx,callfun : Compute_Auto_JacMul_ODE_RHS(x,dx,callfun)
    
def Compute_Auto_JacMul_ODE_RHS_LinOpt(x,callfun):

    args = callfun[0]

    nbody = args['nbody']

    return sp.linalg.LinearOperator((2*nbody*ndim,2*nbody*ndim),
        matvec =  (lambda dx,xl=x,callfunl=callfun : Compute_Auto_JacMul_ODE_RHS(xl,dx,callfunl)),
        rmatvec = (lambda dx,xl=x,callfunl=callfun : Compute_Auto_JacMul_ODE_RHS(xl,dx,callfunl)))

def GetTangentSystemDef(x,callfun,nint=None,method = 'SymplecticEuler'):

        args = callfun[0]
        nbody = args['nbody']
        mass = args['mass']
        ndof = nbody*ndim

        if nint is None:
            nint = args['nint_list'][args["current_cvg_lvl"]]

        if   method in ['SymplecticEuler','SymplecticEuler_XV','SymplecticEuler_VX']:
            pass
        elif method in ['SymplecticStormerVerlet','SymplecticStormerVerlet_XV','SymplecticStormerVerlet_VX']:
            nint = 2*nint
        elif method in ['SymplecticRuth3','SymplecticRuth3_XV','SymplecticRuth3_VX']:
            nint = 24*nint

        all_pos_vel = ComputeAllPosVel(x,callfun,nint=nint)

        def fun(t,v):
            return v

        def gun(t,x):
            i = round(t*nint) % nint

            cur_pos = np.ascontiguousarray(all_pos_vel[0,:,:,i])

            J = Compute_JacMat_Forces_Cython(cur_pos,mass,nbody).reshape(nbody*ndim,nbody*ndim)
            
            return J.dot(x.reshape(nbody*ndim,2*nbody*ndim)).reshape(-1)

        x0 = np.ascontiguousarray(np.concatenate((np.eye(ndof),np.zeros((ndof,ndof))),axis=1).reshape(-1))
        v0 = np.ascontiguousarray(np.concatenate((np.zeros((ndof,ndof)),np.eye(ndof)),axis=1).reshape(-1))

        return fun,gun,x0,v0

def HeuristicMinMax(callfun):

    args = callfun[0]
    nbody = args['nbody']
    nloop = args['nloop']
    loopnb = args['loopnb']
    Targets = args['Targets']
    SpaceRotsUn = args['SpaceRotsUn']
    all_pos = args['last_all_pos']

    xyminmaxl = np.zeros((2,2))
    xyminmax = np.zeros((2))
    xy = np.zeros((2))

    xmin = all_pos[0,0,0]
    xmax = all_pos[0,0,0]
    ymin = all_pos[0,1,0]
    ymax = all_pos[0,1,0]

    for il in range(nloop):

        xyminmaxl[0,0] = all_pos[il,0,:].min()
        xyminmaxl[1,0] = all_pos[il,0,:].max()
        xyminmaxl[0,1] = all_pos[il,1,:].min()
        xyminmaxl[1,1] = all_pos[il,1,:].max()

        for ib in range(loopnb[il]):

            if (args["RequiresLoopDispUn"][il,ib]):

                for i in range(2):

                    for j in range(2):

                        xyminmax[0] = xyminmaxl[i,0]
                        xyminmax[1] = xyminmaxl[j,1]

                        xy = np.dot(SpaceRotsUn[il,ib,:,:],xyminmax)

                        xmin = min(xmin,xy[0])
                        xmax = max(xmax,xy[0])
                        ymin = min(ymin,xy[1])
                        ymax = max(ymax,xy[1])

    return xmin,xmax,ymin,ymax

