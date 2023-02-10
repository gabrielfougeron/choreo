'''
Choreo_funs.py : Defines useful functions in the Choreographies2 project.
'''

import os
import itertools
import copy
import time
import pickle
import warnings
import functools
import json

import numpy as np
import math as m
import scipy.fft
import scipy.optimize
import scipy.linalg as la
import scipy.sparse as sp
import sparseqr
import networkx
import random
import inspect
import fractions

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import cnames
from matplotlib.collections import LineCollection
from matplotlib import animation

try:
    import ffmpeg
except:
    pass

from choreo.Choreo_cython_funs import ndim,twopi,nhash,n
from choreo.Choreo_cython_funs import Compute_action_Cython,Compute_action_hess_mul_Cython
from choreo.Choreo_cython_funs import Compute_hash_action_Cython,Compute_Newton_err_Cython
from choreo.Choreo_cython_funs import Assemble_Cstr_Matrix,diag_changevar
from choreo.Choreo_cython_funs import Compute_MinDist_Cython,Compute_Loop_Dist_btw_avg_Cython,Compute_square_dist,Compute_Loop_Size_Dist_Cython
from choreo.Choreo_cython_funs import Compute_Forces_Cython,Compute_JacMat_Forces_Cython,Compute_JacMul_Forces_Cython
from choreo.Choreo_cython_funs import Transform_Coeffs_Single_Loop,SparseScaleCoeffs,ComputeSpeedCoeffs
from choreo.Choreo_cython_funs import the_irfft,the_rfft
from choreo.Choreo_cython_funs import Compute_action_hess_mul_Tan_Cython_nosym

if ndim == 2:
    from choreo.Choreo_cython_funs_2D import Compute_action_Cython_2D as Compute_action_Cython ,Compute_action_hess_mul_Cython_2D as Compute_action_hess_mul_Cython
    from choreo.Choreo_cython_funs_2D import RotateFastWithSlow_2D

from choreo.Choreo_scipy_plus import *

def Pick_Named_Args_From_Dict(fun,the_dict,MissingArgsAreNone = True):
    
    list_of_args = inspect.getfullargspec(fun).args
    
    if MissingArgsAreNone:
        
        all_kwargs = {k:the_dict.get(k,None) for k in list_of_args}
        
    else:
        
        all_kwargs = {k:the_dict[k] for k in list_of_args}
    
    return all_kwargs

class ChoreoAction():
    r"""
    This class defines everything needed to compute the action.
    """
    def __init__(self, **kwargs):
        r"""
        Class constructor. Just shove everything in there.
        """

        for key, value in kwargs.items():
            setattr(self, key, value)

        for key in kwargs.keys():
            
            if key.endswith('_cvg_lvl_list'):
                self.DefGetCurrentListAttribute(key)

    def __str__(self):

        res = 'ChoreoAction object:\n'

        for key,val in self.__dict__.items():

            res += f'{key} : {val}\n'

        return res

    def GetCurrentListAttribute(self,key):

        return getattr(self, key)[ getattr(self,'current_cvg_lvl') ]

    def DefGetCurrentListAttribute(self,key):
        
        if not(isinstance(getattr(self, key, None), list)):
            raise ValueError(f"{key} is not a list.")

        fun_name = key.removesuffix('_cvg_lvl_list')

        if not(hasattr(ChoreoAction, fun_name)):

            setattr(ChoreoAction, fun_name, property(functools.partial(ChoreoAction.GetCurrentListAttribute,key=key)))

    def Package_all_coeffs(self,all_coeffs):
        r"""
        Transfers the Fourier coefficients of the generators to a single vector of parameters for search.
        The packaging process projects the trajectory onto the space of constraint satisfying trajectories.
        """

        return self.coeff_to_param.dot(all_coeffs.reshape(-1))
        
    def Unpackage_all_coeffs(self,x):
        r"""
        Computes the Fourier coefficients of the generator given the parameters.
        """
        
        y = self.param_to_coeff.dot(x)
        all_coeffs = y.reshape(self.nloop,ndim,self.ncoeff,2)
        
        return all_coeffs

    def RemoveSym(self,x):
        r"""
        Removes symmetries and returns coeffs for all bodies.
        """

        return RemoveSym_ann(
            self.Unpackage_all_coeffs(x),
            self.nbody,
            self.nloop,
            self.ncoeff,
            self.loopnb,
            self.Targets,
            self.SpaceRotsUn,
            self.TimeRevsUn,
            self.TimeShiftNumUn,
            self.TimeShiftDenUn
        )

    def ComputeAllPos(self,x,nint=None):
        r"""
        Returns the positions of all bodies.
        """

        if nint is None:
            nint = self.nint

        all_coeffs_nosym = self.RemoveSym(x).view(dtype=np.complex128)[...,0]
        all_pos_b = the_irfft(all_coeffs_nosym,norm="forward")

        return all_pos_b

    def ComputeAllLoopPos(self,x,nint=None):
        r"""
        Returns the positions of all loops, not bodies.
        """

        if nint is None:
            nint = self.nint

        all_coeffs_c = self.Unpackage_all_coeffs(x).view(dtype=np.complex128)[...,0]
        all_pos = the_irfft(all_coeffs_c,norm="forward")

        return all_pos

    def ComputeAllPosVel(self,x,nint=None):
        r"""
        Returns the positions and velocities of all bodies along the path.
        """

        if nint is None:
            nint = self.nint

        all_coeffs_nosym = self.RemoveSym(x).view(dtype=np.complex128)[...,0]
        all_pos_b = the_irfft(all_coeffs_nosym,norm="forward")

        ncoeff = all_coeffs_nosym.shape[2]
        for k in range(ncoeff):
            all_coeffs_nosym[:,:,k] *= twopi*1j*k

        all_vel_b = the_irfft(all_coeffs_nosym,norm="forward")

        return np.stack((all_pos_b,all_vel_b),axis=0)

    def Compute_xlim(self,x,extend=0.):

        all_pos_b = self.ComputeAllPos(x)

        xmin = np.amin(all_pos_b,axis=(0,2))
        xmax = np.amax(all_pos_b,axis=(0,2))

        xmin -= extend*(xmax-xmin)
        xmax += extend*(xmax-xmin)

        return np.stack((xmin,xmax),axis=1).reshape(-1)

    def Compute_init_pos_vel(self,x):
        r"""
        Returns the initial positions and velocities of all bodies.
        """
        # I litterally do not know of any more efficient way to compute the initial positions and velocities.

        all_pos_vel = self.ComputeAllPosVel(x)

        return np.ascontiguousarray(all_pos_vel[:,:,:,0])

    def Compute_action_onlygrad(self,x):
        r"""
        Wrapper function that returns ONLY the gradient of the action with respect to the parameters.
        """
        
        _,y = self.Compute_action(x)
        
        return y

    def Center_all_coeffs(self,all_coeffs):

        Center_all_coeffs(all_coeffs,self.nloop,self.mass,self.loopnb,self.Targets,self.SpaceRotsUn)
        
    def Compute_action_onlygrad_escape(self,x):

        rms_dist = Compute_Loop_Dist_btw_avg_Cython(
            self.nloop          ,
            self.ncoeff       ,
            self.nint         ,
            self.mass           ,
            self.loopnb         ,
            self.Targets        ,
            self.MassSum        ,
            self.SpaceRotsUn    ,
            self.TimeRevsUn     ,
            self.TimeShiftNumUn ,
            self.TimeShiftDenUn ,
            self.loopnbi        ,
            self.ProdMassSumAll ,
            self.SpaceRotsBin   ,
            self.TimeRevsBin    ,
            self.TimeShiftNumBin,
            self.TimeShiftDenBin,
            self.Unpackage_all_coeffs(x)
        )

        escape_pen = 1 + self.escape_fac * abs(rms_dist)**self.escape_pow
        
        # print("escape_pen = ",escape_pen)

        self.SavePosFFT(x)

        J,GradJ =  Compute_action_Cython(
            self.nloop          ,
            self.ncoeff       ,
            self.nint         ,
            self.mass           ,
            self.loopnb         ,
            self.Targets        ,
            self.MassSum        ,
            self.SpaceRotsUn    ,
            self.TimeRevsUn     ,
            self.TimeShiftNumUn ,
            self.TimeShiftDenUn ,
            self.loopnbi        ,
            self.ProdMassSumAll ,
            self.SpaceRotsBin   ,
            self.TimeRevsBin    ,
            self.TimeShiftNumBin,
            self.TimeShiftDenBin,
            self.last_all_coeffs,
            self.last_all_pos
        )

        GJ = GradJ.reshape(-1)
        GJparam = (self.param_to_coeff_T.dot(GJ)) * escape_pen
        
        return GJparam

    def SavePosFFT(self,x):

        if self.Do_Pos_FFT:
            
            self.last_all_coeffs = self.Unpackage_all_coeffs(x)
            
            c_coeffs = self.last_all_coeffs.view(dtype=np.complex128)[...,0]
            self.last_all_pos = the_irfft(c_coeffs,norm="forward")
        
    def Compute_action_hess_mul(self,x,dx):
        r"""
        Returns the Hessian of the action (computed wrt the parameters) times a test vector of parameter deviations.
        """

        self.SavePosFFT(x)

        HessJdx = Compute_action_hess_mul_Cython(
            self.nloop                      ,
            self.ncoeff                   ,
            self.nint                     ,
            self.mass                       ,
            self.loopnb                     ,
            self.Targets                    ,
            self.MassSum                    ,
            self.SpaceRotsUn                ,
            self.TimeRevsUn                 ,
            self.TimeShiftNumUn             ,
            self.TimeShiftDenUn             ,
            self.loopnbi                    ,
            self.ProdMassSumAll             ,
            self.SpaceRotsBin               ,
            self.TimeRevsBin                ,
            self.TimeShiftNumBin            ,
            self.TimeShiftDenBin            ,
            self.last_all_coeffs            ,
            self.Unpackage_all_coeffs(dx)   ,
            self.last_all_pos               
        )

        HJdx = HessJdx.reshape(-1)
        z = self.param_to_coeff_T.dot(HJdx)
        
        return z
            
    def Compute_action_hess_LinOpt(self,x):
        r"""
        Returns the Hessian of the action wrt parameters at a given point as a Scipy LinearOperator.
        """

        return sp.linalg.LinearOperator((self.coeff_to_param.shape[0],self.coeff_to_param.shape[0]),
            matvec =  (lambda dx, xl=x, selfl=self : selfl.Compute_action_hess_mul(xl,dx)),
            rmatvec = (lambda dx, xl=x, selfl=self : selfl.Compute_action_hess_mul(xl,dx)))

    def Compute_action(self,x):
        r"""
        Computes the action and its gradient with respect to the parameters at a given value of the parameters.
        """
    
        self.SavePosFFT(x)

        J,GradJ =  Compute_action_Cython(
            self.nloop          ,
            self.ncoeff       ,
            self.nint         ,
            self.mass           ,
            self.loopnb         ,
            self.Targets        ,
            self.MassSum        ,
            self.SpaceRotsUn    ,
            self.TimeRevsUn     ,
            self.TimeShiftNumUn ,
            self.TimeShiftDenUn ,
            self.loopnbi        ,
            self.ProdMassSumAll ,
            self.SpaceRotsBin   ,
            self.TimeRevsBin    ,
            self.TimeShiftNumBin,
            self.TimeShiftDenBin,
            self.last_all_coeffs,
            self.last_all_pos
        )

        GJ = GradJ.reshape(-1)
        y = self.param_to_coeff_T.dot(GJ)
        
        return J,y

    def Compute_hash_action(self,x):
        r"""
        Returns an invariant hash of the trajectories.
        Useful for duplicate detection
        """

        Hash_Action =  Compute_hash_action_Cython(
            self.nloop                      ,
            self.ncoeff                   ,
            self.nint                     ,
            self.mass                       ,
            self.loopnb                     ,
            self.Targets                    ,
            self.MassSum                    ,
            self.SpaceRotsUn                ,
            self.TimeRevsUn                 ,
            self.TimeShiftNumUn             ,
            self.TimeShiftDenUn             ,
            self.loopnbi                    ,
            self.ProdMassSumAll             ,
            self.SpaceRotsBin               ,
            self.TimeRevsBin                ,
            self.TimeShiftNumBin            ,
            self.TimeShiftDenBin            ,
            self.Unpackage_all_coeffs(x)
        )

        return Hash_Action
                
    def Compute_Newton_err(self,x):
        r"""
        Computes the Newton error at a certain value of parameters
        WARNING : DOUBLING NUMBER OF INTEGRATION POINTS
        """

        all_Newt_err =  Compute_Newton_err_Cython(
            self.nbody                  ,
            self.nloop                  ,
            self.ncoeff               ,
            self.nint                 ,
            self.mass                   ,
            self.loopnb                 ,
            self.Targets                ,
            self.SpaceRotsUn            ,
            self.TimeRevsUn             ,
            self.TimeShiftNumUn         ,
            self.TimeShiftDenUn         ,
            self.Unpackage_all_coeffs(x)
        )

        return all_Newt_err
                
    def Detect_Escape(self,x):
        r"""
        Returns True if the trajectories are so far that they are likely to never interact again
        """

        res = Compute_Loop_Size_Dist_Cython(
            self.nloop                  ,
            self.ncoeff               ,
            self.nint                 ,
            self.mass                   ,
            self.loopnb                 ,
            self.Targets                ,
            self.MassSum                ,
            self.SpaceRotsUn            ,
            self.TimeRevsUn             ,
            self.TimeShiftNumUn         ,
            self.TimeShiftDenUn         ,
            self.loopnbi                ,
            self.ProdMassSumAll         ,
            self.SpaceRotsBin           ,
            self.TimeRevsBin            ,
            self.TimeShiftNumBin        ,
            self.TimeShiftDenBin        ,
            self.Unpackage_all_coeffs(x)
        )
        
        # return (max_loop_dist > (4.5 * self.nbody * max_loop_size))
        return (res[1] > (4.5 * self.nbody * res[0])),res
            
    def Compute_MinDist(self,x):
        r"""
        Returns the minimum inter-body distance along a set of trajectories
        """
        
        MinDist =  Compute_MinDist_Cython(
            self.nloop                  ,
            self.ncoeff               ,
            self.nint                 ,
            self.mass                   ,
            self.loopnb                 ,
            self.Targets                ,
            self.MassSum                ,
            self.SpaceRotsUn            ,
            self.TimeRevsUn             ,
            self.TimeShiftNumUn         ,
            self.TimeShiftDenUn         ,
            self.loopnbi                ,
            self.ProdMassSumAll         ,
            self.SpaceRotsBin           ,
            self.TimeRevsBin            ,
            self.TimeShiftNumBin        ,
            self.TimeShiftDenBin        ,
            self.Unpackage_all_coeffs(x) 
        )
        
        return MinDist

    def Compute_MaxPathLength(self,x):
        r"""
        Computes the maximum path length for speed sync
        """

        nint = self.nint

        self.SavePosFFT(x)

        dx = self.last_all_pos.copy()
        dx[:,:,0:(nint-1)] -= self.last_all_pos[:,:,1:nint]
        dx[:,:,nint-1] -= self.last_all_pos[:,:,0]
        
        max_path_length = np.linalg.norm(dx,axis=1).sum(axis=1).max(axis=0)

        return max_path_length

    def Compute_Auto_ODE_RHS(self,x):

        all_pos_vel = x.reshape(2,self.nbody,ndim)
        
        rhs = np.zeros((2,self.nbody,ndim))

        rhs[0,:,:] = all_pos_vel[1,:,:]
        rhs[1,:,:] = Compute_Forces_Cython(
            all_pos_vel[0,:,:]  ,
            self.mass           ,
            self.nbody          
        )

        return rhs.reshape(2*self.nbody*ndim)

    def Compute_ODE_RHS(self,t,x):
        return self.Compute_Auto_ODE_RHS(x)

    def GetSymplecticODEDef(self):

        def fun(t,v):
            return v

        def gun(t,x):
            return Compute_Forces_Cython(
                x.reshape(self.nbody,ndim)  ,
                self.mass                   ,
                self.nbody                  ,
            ).reshape(-1)

        return fun,gun

    def Compute_Auto_JacMat_ODE_RHS(self,x):

        all_pos_vel = x.reshape(2,self.nbody,ndim)
        
        drhs = np.zeros((2,self.nbody,ndim,2,self.nbody,ndim))

        for ib in range(v):
            for idim in range(ndim):
                drhs[0,ib,idim,1,ib,idim] = 1

        drhs[1,:,:,0,:,:] = Compute_JacMat_Forces_Cython(
            all_pos_vel[0,:,:]  ,
            self.mass           ,
            self.nbody          ,
        )

        return drhs.reshape(2*self.nbody*ndim,2*self.nbody*ndim)

    def Compute_JacMat_ODE_RHS(self,t,x):
        return self.Compute_Auto_JacMat_ODE_RHS(x)
            
    def Compute_Auto_JacMul_ODE_RHS(self,x,dx):

        all_pos_vel   =  x.reshape(2,self.nbody,ndim)
        all_pos_vel_d = dx.reshape(2,self.nbody,ndim)
        
        drhs = np.zeros((2,self.nbody,ndim))

        drhs[0,:,:] = all_pos_vel_d[1,:,:]

        drhs[1,:,:] = Compute_JacMul_Forces_Cython(
            all_pos_vel[0,:,:]  ,
            all_pos_vel_d[0,:,:],
            self.mass           ,
            self.nbody          ,
        )

        return drhs.reshape(2*self.nbody*ndim)

    def Compute_JacMul_ODE_RHS(self,t,x,dx):
        return self.Compute_Auto_JacMul_ODE_RHS(x,dx)

    def Compute_Auto_JacMul_ODE_RHS_LinOpt(self,x):

        return sp.linalg.LinearOperator((2*self.nbody*ndim,2*self.nbody*ndim),
            matvec =  (lambda dx, xl=x, selfl=self : selfl.Compute_Auto_JacMul_ODE_RHS(xl,dx)),
            rmatvec = (lambda dx, xl=x, selfl=self : selfl.Compute_Auto_JacMul_ODE_RHS(xl,dx)))

    def GetTangentSystemDef(self,x,nint=None,method = 'SymplecticEuler'):

            if nint is None:
                nint = self.nint

            ndof = self.nbody*ndim

            if nint is None:
                nint = self.nint

            if   method in ['SymplecticEuler','SymplecticEuler_XV','SymplecticEuler_VX']:
                pass
            elif method in ['SymplecticStormerVerlet','SymplecticStormerVerlet_XV','SymplecticStormerVerlet_VX']:
                nint = 2*nint
            elif method in ['SymplecticRuth3','SymplecticRuth3_XV','SymplecticRuth3_VX']:
                nint = 24*nint

            all_pos_vel = self.ComputeAllPosVel(x,nint=nint)

            def fun(t,v):
                return v

            def gun(t,x):
                i = round(t*nint) % nint

                cur_pos = np.ascontiguousarray(all_pos_vel[0,:,:,i])

                J = Compute_JacMat_Forces_Cython(cur_pos,self.mass,self.nbody).reshape(self.nbody*ndim,self.nbody*ndim)
                
                return J.dot(x.reshape(self.nbody*ndim,2*self.nbody*ndim)).reshape(-1)

            x0 = np.ascontiguousarray(np.concatenate((np.eye(ndof),np.zeros((ndof,ndof))),axis=1).reshape(-1))
            v0 = np.ascontiguousarray(np.concatenate((np.zeros((ndof,ndof)),np.eye(ndof)),axis=1).reshape(-1))

            return fun,gun,x0,v0

    def HeuristicMinMax(self):
        r"""
        Computes an approximate quickly computed min max for canvas adjusting
        """

        xyminmaxl = np.zeros((2,2))
        xyminmax = np.zeros((2))
        xy = np.zeros((2))

        xmin = self.last_all_pos[0,0,0]
        xmax = self.last_all_pos[0,0,0]
        ymin = self.last_all_pos[0,1,0]
        ymax = self.last_all_pos[0,1,0]

        for il in range(self.nloop):

            xyminmaxl[0,0] = self.last_all_pos[il,0,:].min()
            xyminmaxl[1,0] = self.last_all_pos[il,0,:].max()
            xyminmaxl[0,1] = self.last_all_pos[il,1,:].min()
            xyminmaxl[1,1] = self.last_all_pos[il,1,:].max()

            for ib in range(self.loopnb[il]):

                if (self.RequiresLoopDispUn[il,ib]):

                    for i in range(2):

                        for j in range(2):

                            xyminmax[0] = xyminmaxl[i,0]
                            xyminmax[1] = xyminmaxl[j,1]

                            xy = np.dot(self.SpaceRotsUn[il,ib,:,:],xyminmax)

                            xmin = min(xmin,xy[0])
                            xmax = max(xmax,xy[0])
                            ymin = min(ymin,xy[1])
                            ymax = max(ymax,xy[1])

        return xmin,xmax,ymin,ymax

    def Compose_Two_Paths(self,Info_dict_slow,Info_dict_fast_list,il_slow_source,ibl_slow_source,il_fast_source,ibl_fast_source,nT_slow,nT_fast,all_coeffs_slow,all_coeffs_fast_list,Rotate_fast_with_slow=False):
        r"""
        Composes a **slow** with a **fast** path
        """

        ncoeff = self.ncoeff
        all_coeffs = np.zeros((self.nloop,ndim,ncoeff,2),dtype=np.float64)

        for il in range(self.nloop):

            ib = self.Targets[il,0]
            il_slow = il_slow_source[ib]
            ibl_slow = ibl_slow_source[ib]
            il_fast = il_fast_source[ib]
            ibl_fast = ibl_fast_source[ib]

            mass_fast_tot = 0.
            
            for il_fast_p in range(Info_dict_fast_list[il_slow]['nloop']):

                for ibl_fast_p in range(Info_dict_fast_list[il_slow]["loopnb"][il_fast]):

                    ib_fast_p = Info_dict_fast_list[il_slow]["Targets"][il_fast_p][ibl_fast_p]
                    mass_fast_tot += Info_dict_fast_list[il_slow]["mass"][ib_fast_p]

            mass_fac = mass_fast_tot / Info_dict_slow["mass"][Info_dict_slow["Targets"][il_slow][ibl_slow]]

            ########################################################

            k_fac_slow = nT_slow
            k_fac_fast = nT_fast[il_slow]
            
            phys_exp = 1/(n-1)

            rfac_slow = (k_fac_slow) ** phys_exp
            rfac_fast = (k_fac_fast*m.sqrt(mass_fac)) ** phys_exp

            ########################################################

            SpaceRot = np.array(Info_dict_fast_list[il_slow]["SpaceRotsUn"][il_fast][ibl_fast],dtype=np.float64)
            TimeRev = float(Info_dict_fast_list[il_slow]["TimeRevsUn"][il_fast][ibl_fast])
            TimeShiftNum = float(Info_dict_fast_list[il_slow]["TimeShiftNumUn"][il_fast][ibl_fast])
            TimeShiftDen = float(Info_dict_fast_list[il_slow]["TimeShiftDenUn"][il_fast][ibl_fast])

            ncoeff_slow = all_coeffs_slow.shape[2]

            all_coeffs_fast = Transform_Coeffs_Single_Loop(SpaceRot, TimeRev, TimeShiftNum, TimeShiftDen, all_coeffs_fast_list[il_slow][il_fast,:,:,:],all_coeffs_fast_list[il_slow].shape[2])
            
            ncoeff_fast = all_coeffs_fast.shape[1]
            
            all_coeffs_slow_mod = SparseScaleCoeffs(all_coeffs_slow[il_slow,:,:,:],ncoeff,ncoeff_slow,k_fac_slow,rfac_slow)
            all_coeffs_fast_mod = SparseScaleCoeffs(all_coeffs_fast               ,ncoeff,ncoeff_fast,k_fac_fast,rfac_fast)

            if Rotate_fast_with_slow :
                
                nint = 2*(ncoeff-1)

                c_coeffs_slow = all_coeffs_slow_mod.view(dtype=np.complex128)[...,0]
                all_pos_slow = the_irfft(c_coeffs_slow,n=nint,axis=1)

                c_coeffs_fast = all_coeffs_fast_mod.view(dtype=np.complex128)[...,0]
                all_pos_fast = the_irfft(c_coeffs_fast,n=nint,axis=1)

                all_coeffs_slow_mod_speed = ComputeSpeedCoeffs(all_coeffs_slow_mod,ncoeff)
                c_coeffs_slow_mod_speed = all_coeffs_slow_mod_speed.view(dtype=np.complex128)[...,0]
                all_pos_slow_mod_speed = the_irfft(c_coeffs_slow_mod_speed,n=nint,axis=1)

                all_pos_avg = RotateFastWithSlow_2D(all_pos_slow,all_pos_slow_mod_speed,all_pos_fast,nint)

                c_coeffs_avg = the_rfft(all_pos_avg,axis=1)

                kmax = min(ncoeff,ncoeff_slow)

                all_coeffs[il,:,0:kmax,0] = c_coeffs_avg[:,0:kmax].real
                all_coeffs[il,:,0:kmax,1] = c_coeffs_avg[:,0:kmax].imag    

            else :

                all_coeffs[il,:,:,:] = all_coeffs_fast_mod + all_coeffs_slow_mod

        return all_coeffs

    def plot_Newton_Error(self,x,filename,fig_size=(8,5),color_list = None):
        
        if color_list is None:
            color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']

        Newt_err = self.Compute_Newton_err(x)
        
        Newt_err = np.linalg.norm(Newt_err,axis=(1))
        
        fig = plt.figure()
        fig.set_size_inches(fig_size)
        ax = plt.gca()
        
        ncol = len(color_list)

        cb = []
        for ib in range(self.nbody):
            cb.append(color_list[ib%ncol])
        
        # for ib in range(self.nbody):
        for ib in range(1):
            ax.plot(Newt_err[ib,:],c=cb[ib])
            
        ax.set_yscale('log')
            
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def Write_Descriptor(self,x,filename,Action=None,Gradaction=None,Newt_err_norm=None,dxmin=None,Hash_Action=None,max_path_length=None,extend=0.03):
        r"""
        Dumps a json file describing the current trajectories
        """

        nint = self.nint

        if ((Action is None) or (Gradaction is None) ):
            Action,Gradaction_vect = self.Compute_action(x)
            Gradaction = np.linalg.norm(Gradaction_vect)

        if Newt_err_norm is None :

            Newt_err = self.Compute_Newton_err(x)
            Newt_err_norm = np.linalg.norm(Newt_err)/(nint*self.nbody)

        if dxmin is None:
            
            dxmin = self.Compute_MinDist(x)

        if Hash_Action is None:
            
            Hash_Action = self.Compute_hash_action(x)

        if max_path_length is None:

            max_path_length = self.Compute_MaxPathLength(x)
        
        c_coeffs = self.Unpackage_all_coeffs(x).view(dtype=np.complex128)[...,0]

        all_pos = the_irfft(c_coeffs,norm="forward")        
        all_pos_b = np.zeros((self.nbody,ndim,nint),dtype=np.float64)
        
        for il in range(self.nloop):
            for ib in range(self.loopnb[il]):
                for iint in range(nint):
                    # exact time is irrelevant
                    all_pos_b[self.Targets[il,ib],:,iint] = np.dot(self.SpaceRotsUn[il,ib,:,:],all_pos[il,:,iint])

        xmin = all_pos_b[:,0,:].min()
        xmax = all_pos_b[:,0,:].max()
        ymin = all_pos_b[:,1,:].min()
        ymax = all_pos_b[:,1,:].max()

        xinf = xmin - extend*(xmax-xmin)
        xsup = xmax + extend*(xmax-xmin)
        
        yinf = ymin - extend*(ymax-ymin)
        ysup = ymax + extend*(ymax-ymin)
        
        hside = max(xsup-xinf,ysup-yinf)/2

        xmid = (xinf+xsup)/2
        ymid = (yinf+ysup)/2

        xinf = xmid - hside
        xsup = xmid + hside

        yinf = ymid - hside
        ysup = ymid + hside

        Info_dict = {}

        max_path_length

        Info_dict["nbody"] = self.nbody
        Info_dict["n_Fourier"] = self.ncoeff
        Info_dict["n_int"] = nint

        Info_dict["mass"] = self.mass.tolist()
        Info_dict["nloop"] = self.nloop

        Info_dict["Action"] = Action
        Info_dict["Grad_Action"] = Gradaction
        Info_dict["Newton_Error"] = Newt_err_norm
        Info_dict["Min_Distance"] = dxmin
        Info_dict["Max_PathLength"] = max_path_length

        Info_dict["Hash"] = Hash_Action.tolist()

        Info_dict["xinf"] = xinf
        Info_dict["xsup"] = xsup
        Info_dict["yinf"] = yinf
        Info_dict["ysup"] = ysup

        Info_dict["loopnb"] = self.loopnb.tolist()
        Info_dict["Targets"] = self.Targets.tolist()
        Info_dict["SpaceRotsUn"] = self.SpaceRotsUn.tolist()
        Info_dict["TimeRevsUn"] = self.TimeRevsUn.tolist()
        Info_dict["TimeShiftNumUn"] = self.TimeShiftNumUn.tolist()
        Info_dict["TimeShiftDenUn"] = self.TimeShiftDenUn.tolist()
        Info_dict["RequiresLoopDispUn"] = self.RequiresLoopDispUn.tolist()

        with open(filename, "w") as jsonFile:
            jsonString = json.dumps(Info_dict, indent=4, sort_keys=False)
            jsonFile.write(jsonString)

    def Check_Duplicates(self,x,hash_dict,store_folder,duplicate_eps,Hash_Action=None):
        r"""
        Checks whether there is a duplicate of a given trajecory in the provided folder
        """

        if Hash_Action is None:
            Hash_Action = self.Compute_hash_action(x)

        file_path_list = SelectFiles_Action(store_folder,hash_dict,Hash_Action,duplicate_eps)

        if (len(file_path_list) == 0):
            
            Found_duplicate = False
            file_path = ''
        
        else:
            Found_duplicate = True
            file_path = file_path_list[0]
        
        return Found_duplicate,file_path

    if ndim == 2:

        def Gen_init_avg_2D(self,nT_slow,nT_fast,Info_dict_slow,all_coeffs_slow,Info_dict_fast_list,all_coeffs_fast_list,il_slow_source,ibl_slow_source,il_fast_source,ibl_fast_source,Rotate_fast_with_slow,Optimize_Init,Randomize_Fast_Init):

            nloop_slow = len(all_coeffs_fast_list)

            if Randomize_Fast_Init :

                init_SpaceRevscal = np.array([1. if (np.random.random() > 1./2.) else -1. for ils in range(nloop_slow)],dtype=np.float64)
                init_TimeRevscal = np.array([1. if (np.random.random() > 1./2.) else -1. for ils in range(nloop_slow)],dtype=np.float64)
                Act_Mul = 1. if (np.random.random() > 1./2.) else -1.
                init_x = np.array([ np.random.random() for iparam in range(2*nloop_slow)],dtype=np.float64)

            else:

                init_SpaceRevscal = np.array([1. for ils in range(nloop_slow)],dtype=np.float64)
                init_TimeRevscal = np.array([1. for ils in range(nloop_slow)],dtype=np.float64)
                Act_Mul = 1.
                init_x = np.zeros((2*nloop_slow),dtype=np.float64)

            def params_to_coeffs(x):

                all_coeffs_fast_list_mod = []

                for ils in range(nloop_slow):

                    theta = twopi * x[2*ils]
                    SpaceRevscal = init_SpaceRevscal[ils]
                    SpaceRots = np.array( [[SpaceRevscal*np.cos(theta) , SpaceRevscal*np.sin(theta)] , [-np.sin(theta),np.cos(theta)]],dtype=np.float64)
                    TimeRevs = init_TimeRevscal[ils]
                    TimeShiftNum = x[2*ils+1]
                    TimeShiftDen = 1

                    all_coeffs_fast_list_mod.append(Transform_Coeffs(SpaceRots, TimeRevs, TimeShiftNum, TimeShiftDen, all_coeffs_fast_list[ils]))

                all_coeffs_avg = self.Compose_Two_Paths(Info_dict_slow,Info_dict_fast_list,il_slow_source,ibl_slow_source,il_fast_source,ibl_fast_source,nT_slow,nT_fast,all_coeffs_slow,all_coeffs_fast_list_mod,Rotate_fast_with_slow)

                return all_coeffs_avg

            if Optimize_Init :

                def params_to_Action(x):

                    all_coeffs_avg = params_to_coeffs(x)

                    x_avg = self.Package_all_coeffs(all_coeffs_avg)
                    Act, GAct = self.Compute_action(x_avg)
                    
                    return Act_Mul * Act

                maxiter = 100
                tol = 1e-10

                opt_result = scipy.optimize.minimize(fun=params_to_Action,x0=init_x,method='CG',options={'disp':False,'maxiter':maxiter,'gtol':tol},tol=tol)

                x_opt = opt_result['x']

                all_coeffs_avg = params_to_coeffs(x_opt)

            else:
                all_coeffs_avg = params_to_coeffs(init_x)

            return all_coeffs_avg

        def plot_all_2D(self,x,nint_plot,filename,fig_size=(10,10),dpi=100,color=None,color_list = None,xlim=None,extend=0.03):
            r"""
            Plots 2D trajectories and saves image in file
            """

            if color_list is None:
                color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']

            if isinstance(color,list):
                
                for the_color in color :
                    
                    file_bas,file_ext = os.path.splitext(filename)
                    
                    the_filename = file_bas+'_'+the_color+file_ext
                    
                    self.plot_all_2D(x=x,nint_plot=nint_plot,filename=the_filename,fig_size=fig_size,dpi=dpi,color=the_color,color_list=color_list)
            
            elif (color is None) or (color == "body") or (color == "loop") or (color == "loop_id") or (color == "none"):
                
                self.plot_all_2D_cpb(x,nint_plot,filename,fig_size=fig_size,dpi=dpi,color=color,color_list=color_list,xlim=xlim,extend=extend)
                
            elif (color == "velocity"):
                
                self.plot_all_2D_cpv(x,nint_plot,filename,fig_size=fig_size,dpi=dpi,xlim=xlim,extend=extend)
                
            elif (color == "all"):
                
                self.plot_all_2D(x=x,nint_plot=nint_plot,filename=filename,fig_size=fig_size,dpi=dpi,color=["body","velocity"],color_list=color_list,xlim=xlim,extend=extend)

            else:
                
                raise ValueError("Unknown color scheme")

        def plot_all_2D_cpb(self,x,nint_plot,filename,fig_size=(10,10),dpi=100,color=None,color_list=None,xlim=None,extend=0.03):
            r"""
            Plots 2D trajectories with one color per body and saves image in file
            """
            
            if color_list is None:
                color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']

            all_coeffs = self.Unpackage_all_coeffs(x)
            
            c_coeffs = all_coeffs.view(dtype=np.complex128)[...,0]
            
            all_pos = np.zeros((self.nloop,ndim,nint_plot+1),dtype=np.float64)
            all_pos[:,:,0:nint_plot] = the_irfft(c_coeffs,n=nint_plot,norm="forward")
            all_pos[:,:,nint_plot] = all_pos[:,:,0]
            
            n_loop_plot = np.count_nonzero((self.RequiresLoopDispUn))
            i_loop_plot = 0

            all_pos_b = np.zeros((n_loop_plot,ndim,nint_plot+1),dtype=np.float64)

            for il in range(self.nloop):
                for ib in range(self.loopnb[il]):
                    if (self.RequiresLoopDispUn[il,ib]) :
                        for iint in range(nint_plot+1):
                            # exact time is irrelevant
                            all_pos_b[i_loop_plot,:,iint] = np.dot(self.SpaceRotsUn[il,ib,:,:],all_pos[il,:,iint])

                        i_loop_plot +=1
            
            ncol = len(color_list)
            
            cb = ['b' for ib in range(n_loop_plot)]
            i_loop_plot = 0

            if (color is None) or (color == "none"):
                for il in range(self.nloop):
                    for ib in range(self.loopnb[il]):
                        if (self.RequiresLoopDispUn[il,ib]) :

                            cb[i_loop_plot] = color_list[0]

                            i_loop_plot +=1

            elif (color == "body"):
                for il in range(self.nloop):
                    for ib in range(self.loopnb[il]):
                        if (self.RequiresLoopDispUn[il,ib]) :

                            cb[i_loop_plot] = color_list[self.Targets[il,ib]%ncol]

                            i_loop_plot +=1

            elif (color == "loop"):
                for il in range(self.nloop):
                    for ib in range(self.loopnb[il]):
                        if (self.RequiresLoopDispUn[il,ib]) :

                            cb[i_loop_plot] = color_list[il%ncol]

                            i_loop_plot +=1

            elif (color == "loop_id"):
                for il in range(self.nloop):
                    for ib in range(self.loopnb[il]):
                        if (self.RequiresLoopDispUn[il,ib]) :

                            cb[i_loop_plot] = color_list[ib%ncol]

                            i_loop_plot +=1

            else:
                raise ValueError(f'Unknown color scheme "{color}"')

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
            
            hside = max(xsup-xinf,ysup-yinf)/2

            xmid = (xinf+xsup)/2
            ymid = (yinf+ysup)/2

            xinf = xmid - hside
            xsup = xmid + hside

            yinf = ymid - hside
            ysup = ymid + hside

            # Plot-related
            fig = plt.figure()
            fig.set_size_inches(fig_size)
            fig.set_dpi(dpi)
            ax = plt.gca()

            lines = sum([ax.plot([], [],'-',color=cb[ib] ,antialiased=True,zorder=-ib)  for ib in range(n_loop_plot)], [])
            points = sum([ax.plot([], [],'ko', antialiased=True)for ib in range(n_loop_plot)], [])

            ax.axis('off')
            ax.set_xlim([xinf, xsup])
            ax.set_ylim([yinf, ysup ])
            ax.set_aspect('equal', adjustable='box')
            plt.tight_layout()
            
            for i_loop_plot in range(n_loop_plot):

                lines[i_loop_plot].set_data(all_pos_b[i_loop_plot,0,:], all_pos_b[i_loop_plot,1,:])

            plt.savefig(filename)
            
            plt.close()

        def plot_all_2D_cpv(self,x,nint_plot,filename,fig_size=(10,10),dpi=100,xlim=None,extend=0.03):
            r"""
            Plots 2D trajectories colored according to velocity and saves image in file
            """
            
            all_coeffs = self.Unpackage_all_coeffs(x)            
            c_coeffs = all_coeffs.view(dtype=np.complex128)[...,0]
            
            all_pos = np.zeros((self.nloop,ndim,nint_plot+1),dtype=np.float64)
            all_pos[:,:,0:nint_plot] = the_irfft(c_coeffs,n=nint_plot,norm="forward")
            all_pos[:,:,nint_plot] = all_pos[:,:,0]
            
            all_coeffs_v = np.zeros(all_coeffs.shape)
            
            for k in range(self.ncoeff):
                all_coeffs_v[:,:,k,0] = -k * all_coeffs[:,:,k,1]
                all_coeffs_v[:,:,k,1] =  k * all_coeffs[:,:,k,0]
            
            c_coeffs_v = all_coeffs_v.view(dtype=np.complex128)[...,0]
            
            all_vel = np.zeros((self.nloop,nint_plot+1),dtype=np.float64)
            all_vel[:,0:nint_plot] = np.linalg.norm(the_irfft(c_coeffs_v,n=nint_plot,norm="forward"),axis=1)
            all_vel[:,nint_plot] = all_vel[:,0]
            
            all_pos_b = np.zeros((self.nbody,ndim,nint_plot+1),dtype=np.float64)
            
            for il in range(self.nloop):
                for ib in range(self.loopnb[il]):
                    for iint in range(nint_plot+1):
                        # exact time is irrelevant
                        all_pos_b[self.Targets[il,ib],:,iint] = np.dot(self.SpaceRotsUn[il,ib,:,:],all_pos[il,:,iint])
            
            all_vel_b = np.zeros((self.nbody,nint_plot+1),dtype=np.float64)
            
            for il in range(nloself.nloopop):
                for ib in range(self.loopnb[il]):
                    for iint in range(nint_plot+1):
                        # exact time is irrelevant
                        all_vel_b[self.Targets[il,ib],iint] = all_vel[il,iint]
            
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
            
            hside = max(xsup-xinf,ysup-yinf)/2

            xmid = (xinf+xsup)/2
            ymid = (yinf+ysup)/2

            xinf = xmid - hside
            xsup = xmid + hside

            yinf = ymid - hside
            ysup = ymid + hside

            # Plot-related
            fig = plt.figure()
            fig.set_size_inches(fig_size)
            fig.set_dpi(dpi)
            ax = plt.gca()

            # cmap = None
            cmap = 'turbo'
            # cmap = 'rainbow'
            
            norm = plt.Normalize(0,all_vel_b.max())
            
            for ib in range(self.nbody-1,-1,-1):
                        
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

        def plot_all_2D_anim(self,x,nint_plot,filename,nperiod=1,Plot_trace=True,fig_size=(5,5),dnint=1,all_pos_trace=None,all_pos_points=None,xlim=None,extend=0.03,color_list=None,color=None):
            r"""
            Creates a video of the bodies moving along their trajectories, and saves the file
            """

            if color_list is None:
                color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']

            all_coeffs = self.Unpackage_all_coeffs(x)

            maxloopnb = self.loopnb.max()
            
            ncol = len(color_list)

            if (all_pos_trace is None):

                n_loop_plot = np.count_nonzero((self.RequiresLoopDispUn))
                i_loop_plot = 0

                cb = ['b' for ib in range(n_loop_plot)]

                i_loop_plot = 0

                if (color is None) or (color == "none"):
                    for il in range(self.nloop):
                        for ib in range(self.loopnb[il]):
                            if (self.RequiresLoopDispUn[il,ib]) :

                                cb[i_loop_plot] = color_list[0]

                                i_loop_plot +=1

                elif (color == "body"):
                    for il in range(self.nloop):
                        for ib in range(self.loopnb[il]):
                            if (self.RequiresLoopDispUn[il,ib]) :

                                cb[i_loop_plot] = color_list[self.Targets[il,ib]%ncol]

                                i_loop_plot +=1

                elif (color == "loop"):
                    for il in range(self.nloop):
                        for ib in range(self.loopnb[il]):
                            if (self.RequiresLoopDispUn[il,ib]) :

                                cb[i_loop_plot] = color_list[il%ncol]

                                i_loop_plot +=1

                elif (color == "loop_id"):
                    for il in range(self.nloop):
                        for ib in range(self.loopnb[il]):
                            if (self.RequiresLoopDispUn[il,ib]) :

                                cb[i_loop_plot] = color_list[ib%ncol]

                                i_loop_plot +=1

                else:
                    raise ValueError(f'Unknown color scheme "{color}"')

            else:
                
                cb = ['b' for ib in range(len(all_pos_trace))]

                if (color == "body"):
                    for ib in range(len(all_pos_trace)):
                        cb[ib] = color_list[ib%ncol]

                else:
                    for ib in range(len(all_pos_trace)):
                        cb[ib] = color_list[0]

            nint_plot_img = nint_plot*dnint
            nint_plot_vid = nint_plot

            if (all_pos_trace is None) or (all_pos_points is None):

                all_pos_b = np.zeros((self.nbody,ndim,nint_plot_img+1),dtype=np.float64)
                all_pos_b[:,:,:nint_plot_img] = self.ComputeAllPos(x,nint=nint_plot_img)
                all_pos_b[:,:,nint_plot_img] = all_pos_b[:,:,0]

            if (all_pos_trace is None):

                i_loop_plot = 0

                all_pos_trace = np.zeros((n_loop_plot,ndim,nint_plot_img+1),dtype=np.float64)

                for il in range(self.nloop):
                    for ib in range(self.loopnb[il]):
                        if (self.RequiresLoopDispUn[il,ib]) :
                            for iint in range(nint_plot+1):
                                # exact time is irrelevant
                                all_pos_trace[i_loop_plot,:,:] = all_pos_b[self.Targets[il,ib],:,:]

                            i_loop_plot +=1

            if (all_pos_points is None):
                all_pos_points = all_pos_b

            size_all_pos_points = all_pos_points.shape[2] - 1

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
            
            hside = max(xsup-xinf,ysup-yinf)/2

            xmid = (xinf+xsup)/2
            ymid = (yinf+ysup)/2

            xinf = xmid - hside
            xsup = xmid + hside

            yinf = ymid - hside
            ysup = ymid + hside


            # Plot-related
            fig = plt.figure()
            fig.set_size_inches(fig_size)
            ax = plt.gca()
            lines = sum([ax.plot([], [],'-',color=cb[ib], antialiased=True,zorder=-ib)  for ib in range(len(all_pos_trace))], [])
            points = sum([ax.plot([], [],'ko', antialiased=True)for ib in range(len(all_pos_points))], [])
            
            ax.axis('off')
            ax.set_xlim([xinf, xsup])
            ax.set_ylim([yinf, ysup ])
            ax.set_aspect('equal', adjustable='box')
            plt.tight_layout()
            
            # TODO: Understand why this is needed / how to rationalize this use. Is it even legal python ?

            iint = [0]
            
            def init():
                
                if (Plot_trace):
                    for ib in range(len(all_pos_trace)):
                        lines[ib].set_data(all_pos_trace[ib,0,:], all_pos_trace[ib,1,:])
                
                return lines + points

            def update(i):
                
                for ib in range(len(all_pos_points)):
                    points[ib].set_data(all_pos_points[ib,0,iint[0]], all_pos_points[ib,1,iint[0]])
                    
                iint[0] = ((iint[0]+dnint) % size_all_pos_points)

                return lines + points
            
            anim = animation.FuncAnimation(fig, update, frames=int(nperiod*nint_plot_vid),init_func=init, blit=True)
                                
            # Save as mp4. This requires mplayer or ffmpeg to be installed
            # anim.save(filename, fps=60, codec='hevc')
            anim.save(filename, fps=60, codec='h264')
            # anim.save(filename, fps=60, codec='webm')
            # anim.save(filename, fps=60,extra_args=['-vcodec ', 'h264_amf'])
            # anim.save(filename, fps=60,extra_args=['-hwaccel ', 'cuda'])
            
            plt.close()
            
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

    def __str__(self):

        out  = ""
        out += f"Loop Target: {self.LoopTarget}\n"
        out += f"Loop Source: {self.LoopSource}\n"
        out += f"SpaceRot: {self.SpaceRot}\n"
        out += f"TimeRev: {self.TimeRev}\n"
        out += f"TimeShift: {self.TimeShift}"

        return out
        
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

        B.ComposeLight(A) returns the composition B o A, i.e. applies A then B, ignoring that target A might be different from source B.
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
            
    def IsSameLight(self,other):
        r"""
        Returns True if the two transformations are almost identical, ignoring source and target
        """   

        RoundTrip = (self.Inverse()).ComposeLight(other)
        RoundTrip.LoopSource = 0
        RoundTrip.LoopTarget = 0

        return RoundTrip.IsIdentity()
        
    def IsSame(self,other):
        r"""
        Returns True if the two transformations are almost identical.
        """   
        return ((self.Inverse()).Compose(other)).IsIdentity()

def setup_changevar(nbody,nint_init,mass,n_reconverge_it_max=6,MomCons=True,n_grad_change=1.,Sym_list=[],CrashOnIdentity=True):
    r"""
    This function constructs a ChoreoAction
    It detects loops and constraints based on symmetries.
    It defines parameters according to given constraints and diagonal change of variable.
    It computes useful objects to optimize the computation of the action :
     - Exhaustive list of unary transformation for generator to body.
     - Exhaustive list of binary transformations from generator within each loop.
    """
    
    Identity_detected = False

    SymGraph = networkx.Graph()
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
            
    Cycles = list(networkx.cycle_basis(SymGraph))    
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
    
    ConnectedComponents = list(networkx.connected_components(SymGraph))    

    nloop = len(ConnectedComponents)

    maxlooplen = 0
    for il in range(nloop):
        looplen = len(ConnectedComponents[il])
        if (looplen > maxlooplen):
            maxlooplen = looplen

    loopgen = np.zeros((nloop),dtype=np.intp)
    loopnb = np.zeros((nloop),dtype=np.intp)
    loopnbi = np.zeros((nloop),dtype=np.intp)
    
    loop_gen_to_target = []
    
    Targets = np.zeros((nloop,maxlooplen),dtype=np.intp)
    MassSum = np.zeros((nloop),dtype=np.float64)
    ProdMassSumAll_list = []
    UniqueSymsAll_list = []
    
    SpaceRotsUn = np.zeros((nloop,maxlooplen,ndim,ndim),dtype=np.float64)
    TimeRevsUn = np.zeros((nloop,maxlooplen),dtype=np.intp)
    TimeShiftNumUn = np.zeros((nloop,maxlooplen),dtype=np.intp)
    TimeShiftDenUn = np.zeros((nloop,maxlooplen),dtype=np.intp)

    for il in range(len(ConnectedComponents)):
        
        loopgen[il] = ConnectedComponents[il].pop()

        paths_to_gen = networkx.shortest_path(SymGraph, target=loopgen[il])
        
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

            if (Sym.TimeShift.denominator > 0):

                TimeShiftNumUn[il,ib] = Sym.TimeShift.numerator % Sym.TimeShift.denominator
                TimeShiftDenUn[il,ib] = Sym.TimeShift.denominator

            else:

                TimeShiftNumUn[il,ib] = -Sym.TimeShift.numerator % (-Sym.TimeShift.denominator)
                TimeShiftDenUn[il,ib] = -Sym.TimeShift.denominator

            
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

                Sym = (gen_to_target[ibp]).Compose(gen_to_target[ib].Inverse())

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

                    IsUnique = not(Sym.IsSameLight(UniqueSyms[isym]))

                    if not(IsUnique):
                        break

                    SymInv = Sym.Inverse()
                    IsUnique = not(SymInv.IsSameLight(UniqueSyms[isym]))

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
    TimeRevsBin = np.zeros((nloop,maxloopnbi),dtype=np.intp)
    TimeShiftNumBin = np.zeros((nloop,maxloopnbi),dtype=np.intp)
    TimeShiftDenBin = np.zeros((nloop,maxloopnbi),dtype=np.intp)

    for il in range(nloop):
        for ibi in range(loopnbi[il]):
            
            ProdMassSumAll[il,ibi] = ProdMassSumAll_list[il][ibi]

            SpaceRotsBin[il,ibi,:,:] = UniqueSymsAll_list[il][ibi].SpaceRot
            TimeRevsBin[il,ibi] = UniqueSymsAll_list[il][ibi].TimeRev

            if (UniqueSymsAll_list[il][ibi].TimeShift.denominator > 0):
                TimeShiftNumBin[il,ibi] = UniqueSymsAll_list[il][ibi].TimeShift.numerator % UniqueSymsAll_list[il][ibi].TimeShift.denominator
                TimeShiftDenBin[il,ibi] = UniqueSymsAll_list[il][ibi].TimeShift.denominator
            else:
                TimeShiftNumBin[il,ibi] = (- UniqueSymsAll_list[il][ibi].TimeShift.numerator) % (- UniqueSymsAll_list[il][ibi].TimeShift.denominator)
                TimeShiftDenBin[il,ibi] = - UniqueSymsAll_list[il][ibi].TimeShift.denominator

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
    loopncstr = np.zeros((nloop),dtype=np.intp)
    
    for il in range(nloop):
        loopncstr[il] = len(SymGraph.nodes[loopgen[il]]["Constraint_list"])
    
    maxloopncstr = loopncstr.max()
    
    SpaceRotsCstr = np.zeros((nloop,maxloopncstr,ndim,ndim),dtype=np.float64)
    TimeRevsCstr = np.zeros((nloop,maxloopncstr),dtype=np.intp)
    TimeShiftNumCstr = np.zeros((nloop,maxloopncstr),dtype=np.intp)
    TimeShiftDenCstr = np.zeros((nloop,maxloopncstr),dtype=np.intp)
    
    for il in range(nloop):
        for i in range(loopncstr[il]):
            
            SpaceRotsCstr[il,i,:,:] = SymGraph.nodes[loopgen[il]]["Constraint_list"][i].SpaceRot
            TimeRevsCstr[il,i]      = SymGraph.nodes[loopgen[il]]["Constraint_list"][i].TimeRev
            TimeShiftNumCstr[il,i]  = SymGraph.nodes[loopgen[il]]["Constraint_list"][i].TimeShift.numerator
            TimeShiftDenCstr[il,i]  = SymGraph.nodes[loopgen[il]]["Constraint_list"][i].TimeShift.denominator

    # Now detect parameters and build change of variables

    MassPowSum = np.zeros((nloop),dtype=np.float64)

    ncoeff_cvg_lvl_list = []
    nint_cvg_lvl_list = []
    param_to_coeff_cvg_lvl_list = []
    coeff_to_param_cvg_lvl_list = []

    param_to_coeff_T_cvg_lvl_list = []
    coeff_to_param_T_cvg_lvl_list = []

    for i in range(n_reconverge_it_max+1):

        assert (nint_init % 2) == 0

        nint_cvg_lvl_list.append(nint_init * (2**i))
        ncoeff_cvg_lvl_list.append(nint_cvg_lvl_list[i] // 2 + 1)

        cstrmat_sp = Assemble_Cstr_Matrix(
            nloop               ,
            ncoeff_cvg_lvl_list[i]      ,
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

        param_to_coeff_cvg_lvl_list.append(null_space_sparseqr(cstrmat_sp))
        coeff_to_param_cvg_lvl_list.append(param_to_coeff_cvg_lvl_list[i].transpose(copy=True))

        print('aaa',param_to_coeff_cvg_lvl_list[i].shape)



        diag_changevar(
            param_to_coeff_cvg_lvl_list[i].nnz,
            ncoeff_cvg_lvl_list[i],
            -n_grad_change,
            param_to_coeff_cvg_lvl_list[i].row,
            param_to_coeff_cvg_lvl_list[i].data,
            MassSum
        )
        
        diag_changevar(
            coeff_to_param_cvg_lvl_list[i].nnz,
            ncoeff_cvg_lvl_list[i],
            n_grad_change,
            coeff_to_param_cvg_lvl_list[i].col,
            coeff_to_param_cvg_lvl_list[i].data,
            MassSum
        )

        param_to_coeff_T_cvg_lvl_list.append(param_to_coeff_cvg_lvl_list[i].transpose(copy=True))
        coeff_to_param_T_cvg_lvl_list.append(coeff_to_param_cvg_lvl_list[i].transpose(copy=True))

    kwargs = {
        "nbody"                         :   nbody                           ,
        "nloop"                         :   nloop                           ,
        "mass"                          :   mass                            ,
        "loopnb"                        :   loopnb                          ,
        "loopgen"                       :   loopgen                         ,
        "Targets"                       :   Targets                         ,
        "MassSum"                       :   MassSum                         ,
        "SpaceRotsUn"                   :   SpaceRotsUn                     ,
        "TimeRevsUn"                    :   TimeRevsUn                      ,
        "TimeShiftNumUn"                :   TimeShiftNumUn                  ,
        "TimeShiftDenUn"                :   TimeShiftDenUn                  ,
        "RequiresLoopDispUn"            :   RequiresLoopDispUn              ,
        "loopnbi"                       :   loopnbi                         ,
        "ProdMassSumAll"                :   ProdMassSumAll                  ,
        "SpaceRotsBin"                  :   SpaceRotsBin                    ,
        "TimeRevsBin"                   :   TimeRevsBin                     ,
        "TimeShiftNumBin"               :   TimeShiftNumBin                 ,
        "TimeShiftDenBin"               :   TimeShiftDenBin                 ,
        "ncoeff_cvg_lvl_list"           :   ncoeff_cvg_lvl_list             ,
        "nint_cvg_lvl_list"             :   nint_cvg_lvl_list               ,
        "param_to_coeff_cvg_lvl_list"   :   param_to_coeff_cvg_lvl_list     ,
        "coeff_to_param_cvg_lvl_list"   :   coeff_to_param_cvg_lvl_list     ,
        "param_to_coeff_T_cvg_lvl_list" :   param_to_coeff_T_cvg_lvl_list   ,
        "coeff_to_param_T_cvg_lvl_list" :   coeff_to_param_T_cvg_lvl_list   ,
        "current_cvg_lvl"               :   0                               ,
        "last_all_coeffs"               :   None                            ,
        "last_all_pos"                  :   None                            ,
        "Do_Pos_FFT"                    :   True                            ,
    }

    return ChoreoAction(**kwargs)

class UniformRandom():
    def __init__(self, d):
        self.d = d
        self.rdn = np.random.RandomState(np.int64(time.time_ns()) % np.int64(2**32))

    def random(self):
        return self.rdn.random_sample((self.d))

def null_space_sparseqr(AT):
    # Returns a basis of the null space of a matrix A.
    # AT must be in COO format
    # The nullspace of the TRANSPOSE of AT will be returned

    # tolerance = 1e-5
    tolerance = None

    Q, R, E, rank = sparseqr.qr( AT, tolerance=tolerance )

    nrow = AT.shape[0]
    
    if (nrow <= rank):
        
        return sp.coo_matrix(([],([],[])),shape=(nrow,0))
    
    else:

        mask = []
        iker = 0
        while (iker < Q.nnz):
            if (Q.col[iker] >= rank):
                mask.append(iker)
            iker += 1
            
        return sp.coo_matrix((Q.data[mask],(Q.row[mask],Q.col[mask]-rank)),shape=(nrow,nrow-rank))
     
def AllPosToAllCoeffs(all_pos,ncoeffs):

    nloop = all_pos.shape[0]

    c_coeffs = the_rfft(all_pos,axis=2,norm="forward")
    all_coeffs = np.zeros((nloop,ndim,ncoeffs,2),dtype=np.float64)
    all_coeffs[:,:,:,0] = c_coeffs.real
    all_coeffs[:,:,:,1] = c_coeffs.imag

    return all_coeffs

def Transform_Coeffs(SpaceRot, TimeRev, TimeShiftNum, TimeShiftDen, all_coeffs):
    # Transforms coeffs defining a path and returns updated coeffs
    
    nloop = all_coeffs.shape[0]
    ncoeff = all_coeffs.shape[2]
    all_coeffs_new = np.zeros(all_coeffs.shape)

    for il in range(nloop):

            all_coeffs_new[il,:,:,:] = Transform_Coeffs_Single_Loop(SpaceRot, float(TimeRev), float(TimeShiftNum), float(TimeShiftDen), all_coeffs[il,:,:,:],ncoeff)
        
    return all_coeffs_new

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

def Compute_bar(all_coeffs,nloop,mass,loopnb,Targets,SpaceRotsUn):

    xbar = np.zeros((ndim))
    tot_mass = 0.

    for il in range(nloop):
        for ib in range(loopnb[il]):

            ibody = Targets[il,ib]

            tot_mass += mass[ibody]
            xbar += mass[ibody] * np.dot(SpaceRotsUn[il,ib,:,:],all_coeffs[il,:,0,0] )

    xbar /= tot_mass

    return xbar

def Center_all_coeffs(all_coeffs,nloop,mass,loopnb,Targets,SpaceRotsUn):

    xbar = Compute_bar(all_coeffs,nloop,mass,loopnb,Targets,SpaceRotsUn)

    for il in range(nloop):

        all_coeffs[il,:,0,0] -= xbar

def RemoveSym_ann(all_coeffs,nbody,nloop,ncoeff,loopnb,Targets,SpaceRotsUn,TimeRevsUn,TimeShiftNumUn,TimeShiftDenUn):
    # Removes symmetries and gives coeffs for all bodies

    all_coeffs_nosym = np.zeros((nbody,ndim,ncoeff,2),dtype=np.float64)

    for il in range(nloop):
        for ib in range(loopnb[il]):

            ibody = Targets[il,ib]

            SpaceRot = SpaceRotsUn[il,ib,:,:]
            TimeRev = TimeRevsUn[il,ib]
            TimeShiftNum = float(TimeShiftNumUn[il,ib])
            TimeShiftDen = float(TimeShiftDenUn[il,ib])

            all_coeffs_nosym[ibody,:,:,:] = Transform_Coeffs_Single_Loop(SpaceRot, TimeRev, TimeShiftNum, TimeShiftDen, all_coeffs[il,:,:],ncoeff)

    return all_coeffs_nosym

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
    for img_name in png_files:
        f.write('file \''+os.path.abspath(img_name)+'\'\n')
        f.write('duration 0.0333333 \n')
    
    if ReverseEnd:
        for img_name in reversed(png_files):
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

    if nvid == 1:
        
        os.rename(input_list[0], output_filename)

    else:
        
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

def ReadHashFromFile(filename):

    with open(filename,'r') as jsonFile:
        Info_dict = json.load(jsonFile)

    the_hash = Info_dict.get("Hash")

    if the_hash is None:
        return None

    else:
        return np.array(the_hash)

def SelectFiles_Action(store_folder,hash_dict,Action_Hash_val=np.zeros((nhash)),rtol=1e-5):
    # Creates a list of possible duplicates based on value of the action and hashes

    file_path_list = []
    for file_path in os.listdir(store_folder):
        file_path = os.path.join(store_folder, file_path)
        file_root, file_ext = os.path.splitext(os.path.basename(file_path))
        
        if (file_ext == '.json' ):
            
            This_Action_Hash = hash_dict.get(file_root)
            
            if (This_Action_Hash is None) :

                This_Action_Hash = ReadHashFromFile(file_path) 

                if not(This_Action_Hash is None):

                    hash_dict[file_root] = This_Action_Hash

            if not(This_Action_Hash is None):

                IsCandidate = True
                for ihash in range(nhash):

                    IsCandidate = (IsCandidate and ((abs(This_Action_Hash[ihash]-Action_Hash_val[ihash])) < ((abs(This_Action_Hash[ihash])+abs(Action_Hash_val[ihash]))*rtol)))

                if IsCandidate:
                    file_path_list.append(store_folder+'/'+file_root)
                        
    return file_path_list

def Param_to_Param_direct(x,ActionSyst_source,ActionSyst_target):

    all_coeffs_source = ActionSyst_source.Unpackage_all_coeffs(x)

    ncoeffs_source = ActionSyst_source.ncoeff
    ncoeffs_target = ActionSyst_target.ncoeff
    
    if (ncoeffs_target < ncoeffs_source):
        z = all_coeffs_source[:,:,0:ncoeffs_target,:].reshape(-1)
    else:
        z = np.zeros((ActionSyst_target.nloop,ndim,ncoeffs_target,2))
        z[:,:,0:ncoeffs_source,:] = all_coeffs_source
        z = z.reshape(-1)

    res = ActionSyst_target.coeff_to_param.dot(z)
    
    return res

def Param_to_Param_rev(Gx,ActionSyst_source,ActionSyst_target):

    ncoeffs_source = ActionSyst_source.ncoeff
    ncoeffs_target = ActionSyst_target.ncoeff

    Gy = ActionSyst_source.coeff_to_param_T.dot(Gx)
    all_coeffs = Gy.reshape(ActionSyst_source.nloop,ndim,ncoeffs_source,2)

    if (ncoeffs_target < ncoeffs_source):
        Gz = all_coeffs[:,:,0:ncoeffs_target,:].reshape(-1)
    else:
        Gz = np.zeros((ActionSyst_target.nloop,ndim,ncoeffs_target,2))
        Gz[:,:,0:ncoeffs_source,:] = all_coeffs
        Gz = Gz.reshape(-1)
    
    res = ActionSyst_target.param_to_coeff_T.dot(Gz)
    
    return res

def TangentLagrangeResidual(
        x,
        nbody,
        ncoeff,
        nint,
        mass,
        all_coeffs,
        all_pos
    ):

    x_1D = x.reshape(-1)
    
    ibeg = 0
    iend = (nbody*ndim*nbody*ndim*ncoeff*2)
    all_coeffs_d = x_1D[ibeg:iend].reshape((nbody,ndim,nbody,ndim,ncoeff,2))

    ibeg = iend
    iend = iend + (2*nbody*ndim*2*nbody*ndim)
    LagrangeMulInit = x_1D[ibeg:iend].reshape((2,nbody,ndim,2,nbody,ndim))


        
    Action_hess_dx, LagrangeMulInit_der = Compute_action_hess_mul_Tan_Cython_nosym(
        nbody           ,
        ncoeff          ,
        nint            ,
        mass            ,
        all_coeffs      ,
        all_coeffs_d    ,
        all_pos         ,
        LagrangeMulInit ,
    )

    return np.concatenate((Action_hess_dx.reshape(-1),LagrangeMulInit_der.reshape(-1)))