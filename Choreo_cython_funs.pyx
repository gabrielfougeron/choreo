#cython: language_level=3, boundscheck=False, wraparound = False

'''
Choreo_cython_funs.pyx : Defines useful compiled functions in the Choreographies2 project.

The functions in this file are (as much as possible) written is Cython.
They will be cythonized (i.e. processed by Cython into a C code, which will be compiled ) in setup.py.

Hence, in this file, performance is favored against readability or ease of use.

This file also defines global constants in both C and Python format like the nuber of space dimensions (ndim), the potential law, ect ...

'''


import os
import numpy as np
cimport numpy as np
cimport cython

import scipy.sparse as sp

from libc.math cimport pow as cpow
from libc.math cimport fabs as cfabs
from libc.math cimport cos as ccos
from libc.math cimport sin as csin
from libc.math cimport sqrt as csqrt
from libc.math cimport isnan as cisnan
from libc.math cimport isinf as cisinf
import time

cdef long cndim = 2 # Number of space dimensions

cdef double cn = -0.5  #coeff of x^2 in the potential power law
cdef double cnm1 = cn-1  #coeff of x^2 in the potential power law
cdef double cnm2 = cn-2  #coeff of x^2 in the potential power law

cdef double ctwopi = 2* np.pi
cdef double cfourpi = 4 * np.pi
cdef double cfourpisq = ctwopi*ctwopi

cdef double cnnm1 = cn*(cn-1)

cdef double cmn = -cn
cdef double cmnnm1 = -cnnm1

# cdef  np.ndarray[double, ndim=1, mode="c"] hash_exps = np.array([0.3,0.4,0.6],dtype=np.float64)
# cdef long cnhash = hash_exps.size
# Unfortunately, Cython does not allow that. Let's do it manually then

cdef double hash_exp0 = 0.3
cdef double hash_exp1 = 0.4
cdef double hash_exp2 = 0.6

cdef long cnhash = 3

# Python definition of the very same variables

ndim = cndim

n = cn
nm1 = cnm1
nm2 = cnm2

twopi = ctwopi
fourpi = cfourpi
fourpisq = cfourpisq

nnm1 = cnnm1

mn = cmn
mnnm1 = cmnnm1

nhash = cnhash

cdef inline (double, double, double) CCpt_interbody_pot(double xsq):  # xsq is the square of the distance between two bodies !
    # Cython dedinition of the potential law
    
    cdef double a = cpow(xsq,cnm2)
    cdef double b = xsq*a
    
    cdef double pot = -xsq*b
    cdef double potp = cmn*b
    cdef double potpp = cmnnm1*a
    
    return pot,potp,potpp
    
def Cpt_interbody_pot(double xsq): 
    # Python definition of the potential law
    
    return CCpt_interbody_pot(xsq)
     
def CCpt_hash_pot(double xsq):  # xsq is the square of the distance between two bodies !
    # C definition of the hashing potential. Allows easy detection of duplicates 
    
    cdef np.ndarray[double, ndim=1, mode="c"] hash_pots = np.zeros((cnhash),dtype=np.float64)

    hash_pots[0] = cpow(xsq,hash_exp0)
    hash_pots[1] = cpow(xsq,hash_exp1)
    hash_pots[2] = cpow(xsq,hash_exp2)
    
    return hash_pots
    
def Compute_action_Cython(
    long nloop,
    long ncoeff,
    long nint,
    np.ndarray[double, ndim=1, mode="c"] mass not None ,
    np.ndarray[long  , ndim=1, mode="c"] loopnb not None ,
    np.ndarray[long  , ndim=2, mode="c"] Targets not None ,
    np.ndarray[double, ndim=1, mode="c"] MassSum not None ,
    np.ndarray[double, ndim=4, mode="c"] SpaceRotsUn not None ,
    np.ndarray[long  , ndim=2, mode="c"] TimeRevsUn not None ,
    np.ndarray[long  , ndim=2, mode="c"] TimeShiftNumUn not None ,
    np.ndarray[long  , ndim=2, mode="c"] TimeShiftDenUn not None ,
    np.ndarray[long  , ndim=1, mode="c"] loopnbi not None ,
    np.ndarray[double, ndim=2, mode="c"] ProdMassSumAll not None ,
    np.ndarray[double, ndim=4, mode="c"] SpaceRotsBin not None ,
    np.ndarray[long  , ndim=2, mode="c"] TimeRevsBin not None ,
    np.ndarray[long  , ndim=2, mode="c"] TimeShiftNumBin not None ,
    np.ndarray[long  , ndim=2, mode="c"] TimeShiftDenBin not None ,
    np.ndarray[double, ndim=4, mode="c"] all_coeffs  not None
    ):
    # This function is probably the most important one.
    # Computes the action and its gradient with respect to the Fourier coefficients of the generator in each loop.
    
    cdef long il,ilp,i
    cdef long idim,idimp
    cdef long ibi
    cdef long ib,ibp
    cdef long iint
    cdef long div
    cdef long k,kp,k2
    cdef double pot,potp,potpp
    cdef double prod_mass,a,b,dx2,prod_fac
    cdef np.ndarray[double, ndim=1, mode="c"]  dx = np.zeros((cndim),dtype=np.float64)
    cdef np.ndarray[double, ndim=1, mode="c"]  df = np.zeros((cndim),dtype=np.float64)
        
    cdef long maxloopnb = loopnb.max()
    cdef long maxloopnbi = loopnbi.max()
    
    cdef double Kin_en = 0

    cdef np.ndarray[double, ndim=4, mode="c"] Action_grad = np.zeros((nloop,cndim,ncoeff,2),np.float64)

    for il in range(nloop):
        
        prod_fac = MassSum[il]*cfourpisq
        
        for idim in range(cndim):
            for k in range(1,ncoeff):
                
                k2 = k*k
                a = prod_fac*k2
                b=2*a

                Kin_en += a *((all_coeffs[il,idim,k,0]*all_coeffs[il,idim,k,0]) + (all_coeffs[il,idim,k,1]*all_coeffs[il,idim,k,1]))
                
                Action_grad[il,idim,k,0] += b*all_coeffs[il,idim,k,0]
                Action_grad[il,idim,k,1] += b*all_coeffs[il,idim,k,1]
        
    cdef double Pot_en = 0.

    c_coeffs = all_coeffs.view(dtype=np.complex128)[...,0]
    
    cdef np.ndarray[double, ndim=3, mode="c"] all_pos = np.fft.irfft(c_coeffs,n=nint,axis=2)*nint

    cdef np.ndarray[long, ndim=2, mode="c"]  all_shiftsUn = np.zeros((nloop,maxloopnb),dtype=np.int_)
    cdef np.ndarray[long, ndim=2, mode="c"]  all_shiftsBin = np.zeros((nloop,maxloopnbi),dtype=np.int_)
    
    for il in range(nloop):
        for ib in range(loopnb[il]):
                
            if not(((-TimeRevsUn[il,ib]*nint*TimeShiftNumUn[il,ib]) % TimeShiftDenUn[il,ib]) == 0):
                print("WARNING : remainder in integer division")
                
            all_shiftsUn[il,ib] = ((-TimeRevsUn[il,ib]*nint*TimeShiftNumUn[il,ib]) // TimeShiftDenUn[il,ib] ) % nint
        
        for ibi in range(loopnbi[il]):

            if not(((-TimeRevsBin[il,ibi]*nint*TimeShiftNumBin[il,ibi]) % TimeShiftDenBin[il,ibi]) == 0):
                print("WARNING : remainder in integer division")
                
            all_shiftsBin[il,ibi] = ((-TimeRevsBin[il,ibi]*nint*TimeShiftNumBin[il,ibi]) // TimeShiftDenBin[il,ibi]) % nint
    
    cdef np.ndarray[double, ndim=3, mode="c"] grad_pot_all = np.zeros((nloop,cndim,nint),dtype=np.float64)

    for iint in range(nint):

        # Different loops
        for il in range(nloop):
            for ilp in range(il+1,nloop):

                for ib in range(loopnb[il]):
                    for ibp in range(loopnb[ilp]):
                        
                        prod_mass = mass[Targets[il,ib]]*mass[Targets[ilp,ibp]]

                        for idim in range(cndim):
                            dx[idim] = SpaceRotsUn[il,ib,idim,0]*all_pos[il,0,all_shiftsUn[il,ib]] - SpaceRotsUn[ilp,ibp,idim,0]*all_pos[ilp,0,all_shiftsUn[ilp,ibp]]
                            for jdim in range(1,cndim):
                                dx[idim] += SpaceRotsUn[il,ib,idim,jdim]*all_pos[il,jdim,all_shiftsUn[il,ib]] - SpaceRotsUn[ilp,ibp,idim,jdim]*all_pos[ilp,jdim,all_shiftsUn[ilp,ibp]]

                        dx2 = dx[0]*dx[0]
                        for idim in range(1,cndim):
                            dx2 += dx[idim]*dx[idim]
                            
                        pot,potp,potpp = CCpt_interbody_pot(dx2)
                        
                        Pot_en += pot*prod_mass

                        a = (2*prod_mass*potp)

                        for idim in range(cndim):
                            dx[idim] = a*dx[idim]

                        for idim in range(cndim):
                            
                            b = SpaceRotsUn[il,ib,0,idim]*dx[0]
                            for jdim in range(1,cndim):
                                b+=SpaceRotsUn[il,ib,jdim,idim]*dx[jdim]
                            
                            grad_pot_all[il ,idim,all_shiftsUn[il ,ib ]] += b
                            
                            b = SpaceRotsUn[ilp,ibp,0,idim]*dx[0]
                            for jdim in range(1,cndim):
                                b+=SpaceRotsUn[ilp,ibp,jdim,idim]*dx[jdim]
                            
                            grad_pot_all[ilp,idim,all_shiftsUn[ilp,ibp]] -= b

        # Same loop + symmetry
        for il in range(nloop):

            for ibi in range(loopnbi[il]):
                
                for idim in range(cndim):
                    dx[idim] = SpaceRotsBin[il,ibi,idim,0]*all_pos[il,0,all_shiftsBin[il,ibi]]
                    for jdim in range(1,cndim):
                        dx[idim] += SpaceRotsBin[il,ibi,idim,jdim]*all_pos[il,jdim,all_shiftsBin[il,ibi]]
                    
                    dx[idim] -= all_pos[il,idim,iint]
                    
                dx2 = dx[0]*dx[0]
                for idim in range(1,cndim):
                    dx2 += dx[idim]*dx[idim]

                pot,potp,potpp = CCpt_interbody_pot(dx2)
                
                Pot_en += pot*ProdMassSumAll[il,ibi]
                
                a = (2*ProdMassSumAll[il,ibi]*potp)

                for idim in range(cndim):
                    dx[idim] = a*dx[idim]

                for idim in range(cndim):
                    
                    b = SpaceRotsBin[il,ibi,0,idim]*dx[0]
                    for jdim in range(1,cndim):
                        b+=SpaceRotsBin[il,ibi,jdim,idim]*dx[jdim]
                    
                    grad_pot_all[il ,idim,all_shiftsBin[il,ibi]] += b
                    
                    grad_pot_all[il ,idim,iint] -= dx[idim]

        # Increments time at the end
        for il in range(nloop):
            for ib in range(loopnb[il]):
                all_shiftsUn[il,ib] = (all_shiftsUn[il,ib]+TimeRevsUn[il,ib]) % nint
                
            for ibi in range(loopnbi[il]):
                all_shiftsBin[il,ibi] = (all_shiftsBin[il,ibi]+TimeRevsBin[il,ibi]) % nint

    cdef np.ndarray[doublecomplex , ndim=3, mode="c"]  grad_pot_fft = np.fft.ihfft(grad_pot_all,nint)


    for il in range(nloop):
        for idim in range(cndim):
            
            Action_grad[il,idim,0,0] -= grad_pot_fft[il,idim,0].real
            
            for k in range(1,ncoeff):
            
                Action_grad[il,idim,k,0] -= 2*grad_pot_fft[il,idim,k].real
                Action_grad[il,idim,k,1] += 2*grad_pot_fft[il,idim,k].imag

    Pot_en = Pot_en / nint
    
    Action = Kin_en-Pot_en
    
#~     if cisnan(Action):
#~         print("Action is NaN.")
#~     if cisinf(Action):
#~         print("Action is Infinity. Likely explaination : two body positions might have been identical")
    
    return Action,Action_grad
    
def Compute_hash_action_Cython(
    long nloop,
    long ncoeff,
    long nint,
    np.ndarray[double, ndim=1, mode="c"] mass not None ,
    np.ndarray[long  , ndim=1, mode="c"] loopnb not None ,
    np.ndarray[long  , ndim=2, mode="c"] Targets not None ,
    np.ndarray[double, ndim=1, mode="c"] MassSum not None ,
    np.ndarray[double, ndim=4, mode="c"] SpaceRotsUn not None ,
    np.ndarray[long  , ndim=2, mode="c"] TimeRevsUn not None ,
    np.ndarray[long  , ndim=2, mode="c"] TimeShiftNumUn not None ,
    np.ndarray[long  , ndim=2, mode="c"] TimeShiftDenUn not None ,
    np.ndarray[long  , ndim=1, mode="c"] loopnbi not None ,
    np.ndarray[double, ndim=2, mode="c"] ProdMassSumAll not None ,
    np.ndarray[double, ndim=4, mode="c"] SpaceRotsBin not None ,
    np.ndarray[long  , ndim=2, mode="c"] TimeRevsBin not None ,
    np.ndarray[long  , ndim=2, mode="c"] TimeShiftNumBin not None ,
    np.ndarray[long  , ndim=2, mode="c"] TimeShiftDenBin not None ,
    np.ndarray[double, ndim=4, mode="c"] all_coeffs  not None
    ):
    # Computes the hash of a set of trajectories.
    # The hash is meant to provide a likely unique short identification for duplicate detection.
    # It is hence engineered to be invariant wrt permutation of bodies, time shifts / reversals and space isometries.

    cdef long il,ilp,i
    cdef long idim,idimp
    cdef long ibi
    cdef long ib,ibp
    cdef long iint
    cdef long div
    cdef long k,kp,k2
    cdef double pot,potp,potpp
    cdef double prod_mass,a,b,dx2,prod_fac
    cdef np.ndarray[double, ndim=1, mode="c"]  dx = np.zeros((cndim),dtype=np.float64)
        
    cdef long maxloopnb = loopnb.max()
    cdef long maxloopnbi = loopnbi.max()
    
    cdef long ihash
    
    cdef np.ndarray[double, ndim=1, mode="c"]  Hash_En = np.zeros((cnhash),dtype=np.float64)
    cdef np.ndarray[double, ndim=1, mode="c"]  Hash_pot = np.zeros((cnhash),dtype=np.float64)

    c_coeffs = all_coeffs.view(dtype=np.complex128)[...,0]
    
    cdef np.ndarray[double, ndim=3, mode="c"] all_pos = np.fft.irfft(c_coeffs,n=nint,axis=2)*nint

    cdef np.ndarray[long, ndim=2, mode="c"]  all_shiftsUn = np.zeros((nloop,maxloopnb),dtype=np.int_)
    cdef np.ndarray[long, ndim=2, mode="c"]  all_shiftsBin = np.zeros((nloop,maxloopnbi),dtype=np.int_)
    
    for il in range(nloop):
        for ib in range(loopnb[il]):
                
            if not(((-TimeRevsUn[il,ib]*nint*TimeShiftNumUn[il,ib]) % TimeShiftDenUn[il,ib]) == 0):
                print("WARNING : remainder in integer division")
                
            all_shiftsUn[il,ib] = ((-TimeRevsUn[il,ib]*nint*TimeShiftNumUn[il,ib]) // TimeShiftDenUn[il,ib] ) % nint
        
        for ibi in range(loopnbi[il]):

            if not(((-TimeRevsBin[il,ibi]*nint*TimeShiftNumBin[il,ibi]) % TimeShiftDenBin[il,ibi]) == 0):
                print("WARNING : remainder in integer division")
                
            all_shiftsBin[il,ibi] = ((-TimeRevsBin[il,ibi]*nint*TimeShiftNumBin[il,ibi]) // TimeShiftDenBin[il,ibi]) % nint
    
    cdef np.ndarray[double, ndim=3, mode="c"] grad_pot_all = np.zeros((nloop,cndim,nint),dtype=np.float64)

    for iint in range(nint):

        # Different loops
        for il in range(nloop):
            for ilp in range(il+1,nloop):

                for ib in range(loopnb[il]):
                    for ibp in range(loopnb[ilp]):
                        
                        prod_mass = mass[Targets[il,ib]]*mass[Targets[ilp,ibp]]

                        for idim in range(cndim):
                            dx[idim] = SpaceRotsUn[il,ib,idim,0]*all_pos[il,0,all_shiftsUn[il,ib]] - SpaceRotsUn[ilp,ibp,idim,0]*all_pos[ilp,0,all_shiftsUn[ilp,ibp]]
                            for jdim in range(1,cndim):
                                dx[idim] += SpaceRotsUn[il,ib,idim,jdim]*all_pos[il,jdim,all_shiftsUn[il,ib]] - SpaceRotsUn[ilp,ibp,idim,jdim]*all_pos[ilp,jdim,all_shiftsUn[ilp,ibp]]

                        dx2 = dx[0]*dx[0]
                        for idim in range(1,cndim):
                            dx2 += dx[idim]*dx[idim]
                            
                        Hash_pot = CCpt_hash_pot(dx2)
                        
                        for ihash in range(cnhash):
                            Hash_En[ihash] += Hash_pot[ihash] * prod_mass

        # Same loop + symmetry
        for il in range(nloop):

            for ibi in range(loopnbi[il]):
                
                for idim in range(cndim):
                    dx[idim] = SpaceRotsBin[il,ibi,idim,0]*all_pos[il,0,all_shiftsBin[il,ibi]]
                    for jdim in range(1,cndim):
                        dx[idim] += SpaceRotsBin[il,ibi,idim,jdim]*all_pos[il,jdim,all_shiftsBin[il,ibi]]
                    
                    dx[idim] -= all_pos[il,idim,iint]
                    
                dx2 = dx[0]*dx[0]
                for idim in range(1,cndim):
                    dx2 += dx[idim]*dx[idim]

                Hash_pot = CCpt_hash_pot(dx2)
                
                for ihash in range(cnhash):
                    Hash_En[ihash] += Hash_pot[ihash] * ProdMassSumAll[il,ibi]


        # Increments time at the end
        for il in range(nloop):
            for ib in range(loopnb[il]):
                all_shiftsUn[il,ib] = (all_shiftsUn[il,ib]+TimeRevsUn[il,ib]) % nint
                
            for ibi in range(loopnbi[il]):
                all_shiftsBin[il,ibi] = (all_shiftsBin[il,ibi]+TimeRevsBin[il,ibi]) % nint

    for ihash in range(cnhash):
        Hash_En[ihash] /= nint

    return Hash_En
    
def Compute_MinDist_Cython(
    long nloop,
    long ncoeff,
    long nint,
    np.ndarray[double, ndim=1, mode="c"] mass not None ,
    np.ndarray[long  , ndim=1, mode="c"] loopnb not None ,
    np.ndarray[long  , ndim=2, mode="c"] Targets not None ,
    np.ndarray[double, ndim=1, mode="c"] MassSum not None ,
    np.ndarray[double, ndim=4, mode="c"] SpaceRotsUn not None ,
    np.ndarray[long  , ndim=2, mode="c"] TimeRevsUn not None ,
    np.ndarray[long  , ndim=2, mode="c"] TimeShiftNumUn not None ,
    np.ndarray[long  , ndim=2, mode="c"] TimeShiftDenUn not None ,
    np.ndarray[long  , ndim=1, mode="c"] loopnbi not None ,
    np.ndarray[double, ndim=2, mode="c"] ProdMassSumAll not None ,
    np.ndarray[double, ndim=4, mode="c"] SpaceRotsBin not None ,
    np.ndarray[long  , ndim=2, mode="c"] TimeRevsBin not None ,
    np.ndarray[long  , ndim=2, mode="c"] TimeShiftNumBin not None ,
    np.ndarray[long  , ndim=2, mode="c"] TimeShiftDenBin not None ,
    np.ndarray[double, ndim=4, mode="c"]  all_coeffs  not None
    ):
    # Computes the minimum inter-body distance along the trajectory.
    # A useful tool for collision detection.

    cdef long il,ilp,i
    cdef long idim,idimp
    cdef long ibi
    cdef long ib,ibp
    cdef long iint
    cdef long div
    cdef long k,kp,k2
    cdef double pot,potp,potpp
    cdef double prod_mass,a,b,dx2,prod_fac
    cdef np.ndarray[double, ndim=1, mode="c"]  dx = np.zeros((cndim),dtype=np.float64)
#~     cdef np.ndarray[double, ndim=1, mode="c"]  df = np.zeros((cndim),dtype=np.float64)
        
    cdef long maxloopnb = loopnb.max()
    cdef long maxloopnbi = loopnbi.max()
    
    cdef double dx2min = 1e100

    c_coeffs = all_coeffs.view(dtype=np.complex128)[...,0]
    
    cdef np.ndarray[double, ndim=3, mode="c"] all_pos = np.fft.irfft(c_coeffs,n=nint,axis=2)*nint

    cdef np.ndarray[long, ndim=2, mode="c"]  all_shiftsUn = np.zeros((nloop,maxloopnb),dtype=np.int_)
    cdef np.ndarray[long, ndim=2, mode="c"]  all_shiftsBin = np.zeros((nloop,maxloopnbi),dtype=np.int_)
    
    for il in range(nloop):
        for ib in range(loopnb[il]):
                
            if not(((-TimeRevsUn[il,ib]*nint*TimeShiftNumUn[il,ib]) % TimeShiftDenUn[il,ib]) == 0):
                print("WARNING : remainder in integer division")
                
            all_shiftsUn[il,ib] = ((-TimeRevsUn[il,ib]*nint*TimeShiftNumUn[il,ib]) // TimeShiftDenUn[il,ib] ) % nint
        
        for ibi in range(loopnbi[il]):

            if not(((-TimeRevsBin[il,ibi]*nint*TimeShiftNumBin[il,ibi]) % TimeShiftDenBin[il,ibi]) == 0):
                print("WARNING : remainder in integer division")
                
            all_shiftsBin[il,ibi] = ((-TimeRevsBin[il,ibi]*nint*TimeShiftNumBin[il,ibi]) // TimeShiftDenBin[il,ibi]) % nint

    for iint in range(nint):

        # Different loops
        for il in range(nloop):
            for ilp in range(il+1,nloop):

                for ib in range(loopnb[il]):
                    for ibp in range(loopnb[ilp]):
                        
                        for idim in range(cndim):
                            dx[idim] = SpaceRotsUn[il,ib,idim,0]*all_pos[il,0,all_shiftsUn[il,ib]] - SpaceRotsUn[ilp,ibp,idim,0]*all_pos[ilp,0,all_shiftsUn[ilp,ibp]]
                            for jdim in range(1,cndim):
                                dx[idim] += SpaceRotsUn[il,ib,idim,jdim]*all_pos[il,jdim,all_shiftsUn[il,ib]] - SpaceRotsUn[ilp,ibp,idim,jdim]*all_pos[ilp,jdim,all_shiftsUn[ilp,ibp]]

                        dx2 = dx[0]*dx[0]
                        for idim in range(1,cndim):
                            dx2 += dx[idim]*dx[idim]
                            
                        if (dx2 < dx2min):
                            dx2min = dx2
                            
        # Same loop + symmetry
        for il in range(nloop):

            for ibi in range(loopnbi[il]):
                
                for idim in range(cndim):
                    dx[idim] = SpaceRotsBin[il,ibi,idim,0]*all_pos[il,0,all_shiftsBin[il,ibi]]
                    for jdim in range(1,cndim):
                        dx[idim] += SpaceRotsBin[il,ibi,idim,jdim]*all_pos[il,jdim,all_shiftsBin[il,ibi]]
                    
                    dx[idim] -= all_pos[il,idim,iint]
                    
                dx2 = dx[0]*dx[0]
                for idim in range(1,cndim):
                    dx2 += dx[idim]*dx[idim]
                    
                if (dx2 < dx2min):
                    dx2min = dx2

        # Increments time at the end
        for il in range(nloop):
            for ib in range(loopnb[il]):
                all_shiftsUn[il,ib] = (all_shiftsUn[il,ib]+TimeRevsUn[il,ib]) % nint
                
            for ibi in range(loopnbi[il]):
                all_shiftsBin[il,ibi] = (all_shiftsBin[il,ibi]+TimeRevsBin[il,ibi]) % nint

    return csqrt(dx2min)
   
def Compute_Loop_Dist_Cython(
    long nloop,
    long ncoeff,
    long nint,
    np.ndarray[double, ndim=1, mode="c"] mass not None ,
    np.ndarray[long  , ndim=1, mode="c"] loopnb not None ,
    np.ndarray[long  , ndim=2, mode="c"] Targets not None ,
    np.ndarray[double, ndim=1, mode="c"] MassSum not None ,
    np.ndarray[double, ndim=4, mode="c"] SpaceRotsUn not None ,
    np.ndarray[long  , ndim=2, mode="c"] TimeRevsUn not None ,
    np.ndarray[long  , ndim=2, mode="c"] TimeShiftNumUn not None ,
    np.ndarray[long  , ndim=2, mode="c"] TimeShiftDenUn not None ,
    np.ndarray[long  , ndim=1, mode="c"] loopnbi not None ,
    np.ndarray[double, ndim=2, mode="c"] ProdMassSumAll not None ,
    np.ndarray[double, ndim=4, mode="c"] SpaceRotsBin not None ,
    np.ndarray[long  , ndim=2, mode="c"] TimeRevsBin not None ,
    np.ndarray[long  , ndim=2, mode="c"] TimeShiftNumBin not None ,
    np.ndarray[long  , ndim=2, mode="c"] TimeShiftDenBin not None ,
    np.ndarray[double, ndim=4, mode="c"]  all_coeffs  not None
    ):
        
    cdef long il,ilp,i
    cdef long idim,idimp
    cdef long ibi
    cdef long ib,ibp
    cdef long iint
    cdef long div
    cdef long k,kp,k2
    cdef double sum_loop_dist2
    cdef double dx2
    cdef np.ndarray[double, ndim=1, mode="c"]  dx = np.zeros((cndim),dtype=np.float64)

    sum_loop_dist2 = 0.
    for il in range(nloop-1):
        for ilp in range(il,nloop):
            
            for ib in range(loopnb[il]):
                for ibp in range(loopnb[ilp]):
                    
                    for idim in range(cndim):
                        dx[idim] = SpaceRotsUn[il,ib,idim,0]*all_coeffs[il,0,0,0] - SpaceRotsUn[ilp,ibp,idim,0]*all_coeffs[ilp,0,0,0]
                    
                        for jdim in range(1,cndim):
                            dx[idim] += SpaceRotsUn[il,ib,idim,jdim]*all_coeffs[il,jdim,0,0] - SpaceRotsUn[ilp,ibp,idim,jdim]*all_coeffs[ilp,jdim,0,0]
                            
                    dx2 = dx[0]*dx[0]
                    for idim in range(1,cndim):
                        dx2 += dx[idim]*dx[idim]

                    sum_loop_dist2 += dx2


    for il in range(nloop):
        for ibi in range(loopnbi[il]):
                
            for idim in range(cndim):
                
                dx[idim] = SpaceRotsBin[il,ibi,idim,0]*all_coeffs[il,0,0,0] - all_coeffs[il,idim,0,0]
                for jdim in range(1,cndim):
                    
                    dx[idim] += SpaceRotsBin[il,ibi,idim,jdim]*all_coeffs[il,jdim,0,0]
                    
                
                dx2 = dx[0]*dx[0]
                for idim in range(1,cndim):
                    dx2 += dx[idim]*dx[idim]

                sum_loop_dist2 += dx2

    return csqrt(sum_loop_dist2)
   
   
   
def Compute_Loop_Dist_Cython_test(
    long nloop,
    long ncoeff,
    long nint,
    np.ndarray[double, ndim=1, mode="c"] mass not None ,
    np.ndarray[long  , ndim=1, mode="c"] loopnb not None ,
    np.ndarray[long  , ndim=2, mode="c"] Targets not None ,
    np.ndarray[double, ndim=1, mode="c"] MassSum not None ,
    np.ndarray[double, ndim=4, mode="c"] SpaceRotsUn not None ,
    np.ndarray[long  , ndim=2, mode="c"] TimeRevsUn not None ,
    np.ndarray[long  , ndim=2, mode="c"] TimeShiftNumUn not None ,
    np.ndarray[long  , ndim=2, mode="c"] TimeShiftDenUn not None ,
    np.ndarray[long  , ndim=1, mode="c"] loopnbi not None ,
    np.ndarray[double, ndim=2, mode="c"] ProdMassSumAll not None ,
    np.ndarray[double, ndim=4, mode="c"] SpaceRotsBin not None ,
    np.ndarray[long  , ndim=2, mode="c"] TimeRevsBin not None ,
    np.ndarray[long  , ndim=2, mode="c"] TimeShiftNumBin not None ,
    np.ndarray[long  , ndim=2, mode="c"] TimeShiftDenBin not None ,
    np.ndarray[double, ndim=4, mode="c"]  all_coeffs  not None
    ):

    cdef long il,ilp,i
    cdef long idim,idimp
    cdef long ibi
    cdef long ib,ibp
    cdef long iint
    cdef long div
    cdef long k,kp,k2
    cdef double sum_loop_dist2
    cdef double dx2
    cdef np.ndarray[double, ndim=1, mode="c"]  dx = np.zeros((cndim),dtype=np.float64)

    sum_loop_dist2 = 0.
    for il in range(nloop-1):
        
        dx2 = all_coeffs[il,0,0,0]*all_coeffs[il,0,0,0]
        for idim in range(1,cndim):
            dx2 += all_coeffs[il,idim,0,0]*all_coeffs[il,idim,0,0]

        sum_loop_dist2 += dx2


    return csqrt(sum_loop_dist2)
   
   
def Compute_Loop_Size_Dist_Cython(
    long nloop,
    long ncoeff,
    long nint,
    np.ndarray[double, ndim=1, mode="c"] mass not None ,
    np.ndarray[long  , ndim=1, mode="c"] loopnb not None ,
    np.ndarray[long  , ndim=2, mode="c"] Targets not None ,
    np.ndarray[double, ndim=1, mode="c"] MassSum not None ,
    np.ndarray[double, ndim=4, mode="c"] SpaceRotsUn not None ,
    np.ndarray[long  , ndim=2, mode="c"] TimeRevsUn not None ,
    np.ndarray[long  , ndim=2, mode="c"] TimeShiftNumUn not None ,
    np.ndarray[long  , ndim=2, mode="c"] TimeShiftDenUn not None ,
    np.ndarray[long  , ndim=1, mode="c"] loopnbi not None ,
    np.ndarray[double, ndim=2, mode="c"] ProdMassSumAll not None ,
    np.ndarray[double, ndim=4, mode="c"] SpaceRotsBin not None ,
    np.ndarray[long  , ndim=2, mode="c"] TimeRevsBin not None ,
    np.ndarray[long  , ndim=2, mode="c"] TimeShiftNumBin not None ,
    np.ndarray[long  , ndim=2, mode="c"] TimeShiftDenBin not None ,
    np.ndarray[double, ndim=4, mode="c"]  all_coeffs  not None
    ):

    cdef long il,ilp,i
    cdef long idim,idimp
    cdef long ibi
    cdef long ib,ibp
    cdef long iint
    cdef long div
    cdef long k,kp,k2
    cdef double loop_size,max_loop_size
    cdef double loop_dist,max_loop_dist
    cdef double dx2
    cdef np.ndarray[double, ndim=1, mode="c"]  dx = np.zeros((cndim),dtype=np.float64)
    
    cdef np.ndarray[double, ndim=1, mode="c"]  res = np.zeros((2),dtype=np.float64)


    max_loop_size = 0.
    for il in range(nloop):
        
        loop_size = 0
        
        for idim in range(cndim):
            for k in range(1,ncoeff):
                
                loop_size += all_coeffs[il,idim,k,0]*all_coeffs[il,idim,k,0]+all_coeffs[il,idim,k,1]*all_coeffs[il,idim,k,1]
        
        if (loop_size > max_loop_size):
            max_loop_size = loop_size
            
    res[0] = csqrt(max_loop_size)
    
    max_loop_dist = 0.
    for il in range(nloop-1):
        for ilp in range(il,nloop):
            
            for ib in range(loopnb[il]):
                for ibp in range(loopnb[ilp]):
                    
                    for idim in range(cndim):
                        dx[idim] = SpaceRotsUn[il,ib,idim,0]*all_coeffs[il,0,0,0] - SpaceRotsUn[ilp,ibp,idim,0]*all_coeffs[ilp,0,0,0]
                    
                        for jdim in range(1,cndim):
                            dx[idim] += SpaceRotsUn[il,ib,idim,jdim]*all_coeffs[il,jdim,0,0] - SpaceRotsUn[ilp,ibp,idim,jdim]*all_coeffs[ilp,jdim,0,0]
                            
                    dx2 = dx[0]*dx[0]
                    for idim in range(1,cndim):
                        dx2 += dx[idim]*dx[idim]

                    if (dx2 > max_loop_dist):
                        max_loop_dist = dx2

    for il in range(nloop):
        for ibi in range(loopnbi[il]):
                
            for idim in range(cndim):
                
                dx[idim] = SpaceRotsBin[il,ibi,idim,0]*all_coeffs[il,0,0,0] - all_coeffs[il,idim,0,0]
                for jdim in range(1,cndim):
                    
                    dx[idim] += SpaceRotsBin[il,ibi,idim,jdim]*all_coeffs[il,jdim,0,0]
                    
                
                dx2 = dx[0]*dx[0]
                for idim in range(1,cndim):
                    dx2 += dx[idim]*dx[idim]

                if (dx2 > max_loop_dist):
                    max_loop_dist = dx2

    res[1] = csqrt(max_loop_dist)
    
    return res
   
def Compute_action_hess_mul_Cython(
    long nloop,
    long ncoeff,
    long nint,
    np.ndarray[double, ndim=1, mode="c"] mass not None              ,
    np.ndarray[long  , ndim=1, mode="c"] loopnb not None            ,
    np.ndarray[long  , ndim=2, mode="c"] Targets not None           ,
    np.ndarray[double, ndim=1, mode="c"] MassSum not None           ,
    np.ndarray[double, ndim=4, mode="c"] SpaceRotsUn not None       ,
    np.ndarray[long  , ndim=2, mode="c"] TimeRevsUn not None        ,
    np.ndarray[long  , ndim=2, mode="c"] TimeShiftNumUn not None    ,
    np.ndarray[long  , ndim=2, mode="c"] TimeShiftDenUn not None    ,
    np.ndarray[long  , ndim=1, mode="c"] loopnbi not None           ,
    np.ndarray[double, ndim=2, mode="c"] ProdMassSumAll not None    ,
    np.ndarray[double, ndim=4, mode="c"] SpaceRotsBin not None      ,
    np.ndarray[long  , ndim=2, mode="c"] TimeRevsBin not None       ,
    np.ndarray[long  , ndim=2, mode="c"] TimeShiftNumBin not None   ,
    np.ndarray[long  , ndim=2, mode="c"] TimeShiftDenBin not None   ,
    np.ndarray[double, ndim=4, mode="c"] all_coeffs  not None       ,
    np.ndarray[double, ndim=4, mode="c"] all_coeffs_d  not None
    ):
    # Computes the matrix vector product H*dx where H is the Hessian of the action.
    # Useful to guide the root finding / optimisation process and to better understand the topography of the action (critical points / Morse theory).

    cdef long il,ilp,i
    cdef long idim,idimp
    cdef long ibi
    cdef long ib,ibp
    cdef long iint
    cdef long div
    cdef long k,kp,k2
    cdef double pot,potp,potpp
    cdef double prod_mass,a,b,c,dx2,prod_fac,dxtddx
    cdef np.ndarray[double, ndim=1, mode="c"]  dx = np.zeros((cndim),dtype=np.float64)
    cdef np.ndarray[double, ndim=1, mode="c"]  ddx = np.zeros((cndim),dtype=np.float64)
    cdef np.ndarray[double, ndim=1, mode="c"]  ddf = np.zeros((cndim),dtype=np.float64)
        
    cdef long maxloopnb = loopnb.max()
    cdef long maxloopnbi = loopnbi.max()
    
    cdef double Kin_en = 0

    cdef np.ndarray[double, ndim=4, mode="c"] Action_hess_dx = np.zeros((nloop,cndim,ncoeff,2),np.float64)

    for il in range(nloop):
        
        prod_fac = MassSum[il]*cfourpisq
        
        for idim in range(cndim):
            for k in range(1,ncoeff):
                
                k2 = k*k
                a = prod_fac*k2
                b=2*a
                
                Action_hess_dx[il,idim,k,0] += b*all_coeffs_d[il,idim,k,0]
                Action_hess_dx[il,idim,k,1] += b*all_coeffs_d[il,idim,k,1]
                
    c_coeffs = all_coeffs.view(dtype=np.complex128)[...,0]
    cdef np.ndarray[double, ndim=3, mode="c"] all_pos = np.fft.irfft(c_coeffs,n=nint,axis=2)*nint
    
    c_coeffs_d = all_coeffs_d.view(dtype=np.complex128)[...,0]
    cdef np.ndarray[double, ndim=3, mode="c"]  all_pos_d = np.fft.irfft(c_coeffs_d,n=nint,axis=2)*nint

    cdef np.ndarray[long, ndim=2, mode="c"]  all_shiftsUn = np.zeros((nloop,maxloopnb),dtype=np.int_)
    cdef np.ndarray[long, ndim=2, mode="c"]  all_shiftsBin = np.zeros((nloop,maxloopnbi),dtype=np.int_)
    
    for il in range(nloop):
        for ib in range(loopnb[il]):
                
            if not(((-TimeRevsUn[il,ib]*nint*TimeShiftNumUn[il,ib]) % TimeShiftDenUn[il,ib]) == 0):
                print("WARNING : remainder in integer division")
                
            all_shiftsUn[il,ib] = ((-TimeRevsUn[il,ib]*nint*TimeShiftNumUn[il,ib]) // TimeShiftDenUn[il,ib] ) % nint
        
        for ibi in range(loopnbi[il]):

            if not(((-TimeRevsBin[il,ibi]*nint*TimeShiftNumBin[il,ibi]) % TimeShiftDenBin[il,ibi]) == 0):
                print("WARNING : remainder in integer division")
                
            all_shiftsBin[il,ibi] = ((-TimeRevsBin[il,ibi]*nint*TimeShiftNumBin[il,ibi]) // TimeShiftDenBin[il,ibi]) % nint
    
    cdef np.ndarray[double, ndim=3, mode="c"] hess_pot_all_d = np.zeros((nloop,cndim,nint),dtype=np.float64)

    for iint in range(nint):

        # Different loops
        for il in range(nloop):
            for ilp in range(il+1,nloop):

                for ib in range(loopnb[il]):
                    for ibp in range(loopnb[ilp]):
                        
                        prod_mass = mass[Targets[il,ib]]*mass[Targets[ilp,ibp]]

                        for idim in range(cndim):
                            dx[idim] = SpaceRotsUn[il,ib,idim,0]*all_pos[il,0,all_shiftsUn[il,ib]] - SpaceRotsUn[ilp,ibp,idim,0]*all_pos[ilp,0,all_shiftsUn[ilp,ibp]]
                            ddx[idim] = SpaceRotsUn[il,ib,idim,0]*all_pos_d[il,0,all_shiftsUn[il,ib]] - SpaceRotsUn[ilp,ibp,idim,0]*all_pos_d[ilp,0,all_shiftsUn[ilp,ibp]]
                            for jdim in range(1,cndim):
                                dx[idim] += SpaceRotsUn[il,ib,idim,jdim]*all_pos[il,jdim,all_shiftsUn[il,ib]] - SpaceRotsUn[ilp,ibp,idim,jdim]*all_pos[ilp,jdim,all_shiftsUn[ilp,ibp]]
                                ddx[idim] += SpaceRotsUn[il,ib,idim,jdim]*all_pos_d[il,jdim,all_shiftsUn[il,ib]] - SpaceRotsUn[ilp,ibp,idim,jdim]*all_pos_d[ilp,jdim,all_shiftsUn[ilp,ibp]]

                        dx2 = dx[0]*dx[0]
                        dxtddx = dx[0]*ddx[0]
                        for idim in range(1,cndim):
                            dx2 += dx[idim]*dx[idim]
                            dxtddx += dx[idim]*ddx[idim]
                            
                        pot,potp,potpp = CCpt_interbody_pot(dx2)

                        a = (2*prod_mass*potp)
                        b = (4*prod_mass*potpp*dxtddx)
                        
                        for idim in range(cndim):
                            ddf[idim] = b*dx[idim]+a*ddx[idim]
                            
                        for idim in range(cndim):
                            
                            c = SpaceRotsUn[il,ib,0,idim]*ddf[0]
                            for jdim in range(1,cndim):
                                c+=SpaceRotsUn[il,ib,jdim,idim]*ddf[jdim]
                            
                            hess_pot_all_d[il ,idim,all_shiftsUn[il ,ib ]] += c
                            
                            c = SpaceRotsUn[ilp,ibp,0,idim]*ddf[0]
                            for jdim in range(1,cndim):
                                c+=SpaceRotsUn[ilp,ibp,jdim,idim]*ddf[jdim]
                            
                            hess_pot_all_d[ilp,idim,all_shiftsUn[ilp,ibp]] -= c

        # Same loop + symmetry
        for il in range(nloop):

            for ibi in range(loopnbi[il]):
                
                for idim in range(cndim):
                    dx[idim]  = SpaceRotsBin[il,ibi,idim,0]*all_pos[il,0,all_shiftsBin[il,ibi]]
                    ddx[idim] = SpaceRotsBin[il,ibi,idim,0]*all_pos_d[il,0,all_shiftsBin[il,ibi]]
                    for jdim in range(1,cndim):
                        dx[idim]  += SpaceRotsBin[il,ibi,idim,jdim]*all_pos[il,jdim,all_shiftsBin[il,ibi]]
                        ddx[idim] += SpaceRotsBin[il,ibi,idim,jdim]*all_pos_d[il,jdim,all_shiftsBin[il,ibi]]
                    
                    dx[idim]  -= all_pos[il,idim,iint]
                    ddx[idim] -= all_pos_d[il,idim,iint]
                    
                dx2 = dx[0]*dx[0]
                dxtddx = dx[0]*ddx[0]
                for idim in range(1,cndim):
                    dx2 += dx[idim]*dx[idim]
                    dxtddx += dx[idim]*ddx[idim]

                pot,potp,potpp = CCpt_interbody_pot(dx2)
                
                a = (2*ProdMassSumAll[il,ibi]*potp)
                b = (4*ProdMassSumAll[il,ibi]*potpp*dxtddx)
                
        
                for idim in range(cndim):
                    ddf[idim] = b*dx[idim]+a*ddx[idim]

                for idim in range(cndim):
                    
                    c = SpaceRotsBin[il,ibi,0,idim]*ddf[0]
                    for jdim in range(1,cndim):
                        c+=SpaceRotsBin[il,ibi,jdim,idim]*ddf[jdim]
                    
                    hess_pot_all_d[il ,idim,all_shiftsBin[il,ibi]] += c
                    
                    hess_pot_all_d[il ,idim,iint] -= ddf[idim]
                    
        # Increments time at the end
        for il in range(nloop):
            for ib in range(loopnb[il]):
                all_shiftsUn[il,ib] = (all_shiftsUn[il,ib]+TimeRevsUn[il,ib]) % nint
                
            for ibi in range(loopnbi[il]):
                all_shiftsBin[il,ibi] = (all_shiftsBin[il,ibi]+TimeRevsBin[il,ibi]) % nint

    cdef np.ndarray[doublecomplex , ndim=3, mode="c"]  hess_dx_pot_fft = np.fft.ihfft(hess_pot_all_d,nint)


    for il in range(nloop):
        for idim in range(cndim):
            
            Action_hess_dx[il,idim,0,0] -= hess_dx_pot_fft[il,idim,0].real
            
            for k in range(1,ncoeff):
            
                Action_hess_dx[il,idim,k,0] -= 2*hess_dx_pot_fft[il,idim,k].real
                Action_hess_dx[il,idim,k,1] += 2*hess_dx_pot_fft[il,idim,k].imag

    return Action_hess_dx
    
def Compute_Newton_err_Cython(
    long nbody,
    long nloop,
    long ncoeff,
    long nint,
    np.ndarray[double, ndim=1, mode="c"] mass not None ,
    np.ndarray[long  , ndim=1, mode="c"] loopnb not None ,
    np.ndarray[long  , ndim=2, mode="c"] Targets not None ,
    np.ndarray[double, ndim=4, mode="c"] SpaceRotsUn not None ,
    np.ndarray[long  , ndim=2, mode="c"] TimeRevsUn not None ,
    np.ndarray[long  , ndim=2, mode="c"] TimeShiftNumUn not None ,
    np.ndarray[long  , ndim=2, mode="c"] TimeShiftDenUn not None ,
    np.ndarray[double, ndim=4, mode="c"] all_coeffs  not None
    ):
    # Computes the "Newton error", i.e. the deviation wrt to the fundamental theorem of Newtonian dynamics m_i * a_i - \sum_j f_ij = 0
    # If the Newton error is zero, then the trajectory is physical.
    # Under some symmetry hypotheses, this is the Fourier transform of the gradient of the action.
    # Computing it explicitely is a useful safeguard.

    cdef long il,ilp,i
    cdef long idim,idimp
    cdef long ibi
    cdef long ib,ibp
    cdef long iint
    cdef long div
    cdef long k,kp,k2
    cdef double pot,potp,potpp
    cdef double prod_mass,a,b,dx2,prod_fac
    cdef np.ndarray[double, ndim=1, mode="c"]  dx = np.zeros((cndim),dtype=np.float64)

    cdef long maxloopnb = loopnb.max()
    
    cdef np.ndarray[double, ndim=4, mode="c"] acc_coeff = np.zeros((nloop,cndim,ncoeff,2),dtype=np.float64)

    for il in range(nloop):
        for idim in range(cndim):
            for k in range(ncoeff):
                
                k2 = k*k
                acc_coeff[il,idim,k,0] = k2*cfourpisq*all_coeffs[il,idim,k,0]
                acc_coeff[il,idim,k,1] = k2*cfourpisq*all_coeffs[il,idim,k,1]
                
    c_acc_coeffs = acc_coeff.view(dtype=np.complex128)[...,0]
    cdef np.ndarray[double, ndim=3, mode="c"] all_acc = np.fft.irfft(c_acc_coeffs,n=nint,axis=2)*nint
    
    cdef np.ndarray[double, ndim=3, mode="c"] all_Newt_err = np.zeros((nbody,cndim,nint),np.float64)
    
    c_coeffs = all_coeffs.view(dtype=np.complex128)[...,0]
    
    cdef np.ndarray[double, ndim=3, mode="c"] all_pos = np.fft.irfft(c_coeffs,n=nint,axis=2)*nint

    cdef np.ndarray[long, ndim=2, mode="c"]  all_shiftsUn = np.zeros((nloop,maxloopnb),dtype=np.int_)
    
    for il in range(nloop):
        for ib in range(loopnb[il]):
                
            if not(((-TimeRevsUn[il,ib]*nint*TimeShiftNumUn[il,ib]) % TimeShiftDenUn[il,ib]) == 0):
                print("WARNING : remainder in integer division")
                
            all_shiftsUn[il,ib] = ((-TimeRevsUn[il,ib]*nint*TimeShiftNumUn[il,ib]) // TimeShiftDenUn[il,ib] ) % nint
        
    for iint in range(nint):

        for il in range(nloop):
            for ib in range(loopnb[il]):
                for idim in range(cndim):
                    
                    b = SpaceRotsUn[il,ib,idim,0]*all_acc[il,0,all_shiftsUn[il,ib]]
                    for jdim in range(1,cndim):  
                        b += SpaceRotsUn[il,ib,idim,jdim]*all_acc[il,jdim,all_shiftsUn[il,ib]]                  

                    all_Newt_err[Targets[il,ib],idim,iint] -= mass[Targets[il,ib]]*b

        # Different loops
        for il in range(nloop):
            for ilp in range(il+1,nloop):

                for ib in range(loopnb[il]):
                    for ibp in range(loopnb[ilp]):
                        
                        prod_mass = mass[Targets[il,ib]]*mass[Targets[ilp,ibp]]

                        for idim in range(cndim):
                            dx[idim] = SpaceRotsUn[il,ib,idim,0]*all_pos[il,0,all_shiftsUn[il,ib]] - SpaceRotsUn[ilp,ibp,idim,0]*all_pos[ilp,0,all_shiftsUn[ilp,ibp]]
                            for jdim in range(1,cndim):
                                dx[idim] += SpaceRotsUn[il,ib,idim,jdim]*all_pos[il,jdim,all_shiftsUn[il,ib]] - SpaceRotsUn[ilp,ibp,idim,jdim]*all_pos[ilp,jdim,all_shiftsUn[ilp,ibp]]

                        dx2 = dx[0]*dx[0]
                        for idim in range(1,cndim):
                            dx2 += dx[idim]*dx[idim]
                            
                        pot,potp,potpp = CCpt_interbody_pot(dx2)
                        
                        a = (2*prod_mass*potp)

                        for idim in range(cndim):
                                
                            b = a*dx[idim]
                            all_Newt_err[Targets[il ,ib ],idim,iint] += b
                            all_Newt_err[Targets[ilp,ibp],idim,iint] -= b

        # Same Loop
        for il in range(nloop):

            for ib in range(loopnb[il]):
                for ibp in range(ib+1,loopnb[il]):
                    
                    prod_mass = mass[Targets[il,ib]]*mass[Targets[il,ibp]]

                    for idim in range(cndim):
                        dx[idim] = SpaceRotsUn[il,ib,idim,0]*all_pos[il,0,all_shiftsUn[il,ib]] - SpaceRotsUn[il,ibp,idim,0]*all_pos[il,0,all_shiftsUn[il,ibp]]
                        for jdim in range(1,cndim):
                            dx[idim] += SpaceRotsUn[il,ib,idim,jdim]*all_pos[il,jdim,all_shiftsUn[il,ib]] - SpaceRotsUn[il,ibp,idim,jdim]*all_pos[il,jdim,all_shiftsUn[il,ibp]]
                    
                    dx2 = dx[0]*dx[0]
                    for idim in range(1,cndim):
                        dx2 += dx[idim]*dx[idim]
                        
                    pot,potp,potpp = CCpt_interbody_pot(dx2)
                    
                    a = (2*prod_mass*potp)

                    for idim in range(cndim):
                        
                        b = a*dx[idim]
                        all_Newt_err[Targets[il,ib] ,idim,iint] += b
                        all_Newt_err[Targets[il,ibp],idim,iint] -= b

        # Increments time at the end
        for il in range(nloop):
            for ib in range(loopnb[il]):
                all_shiftsUn[il,ib] = (all_shiftsUn[il,ib]+TimeRevsUn[il,ib]) % nint
                
    return all_Newt_err

def Assemble_Cstr_Matrix(
    long nloop,
    long ncoeff,
    bint MomCons,
    np.ndarray[double, ndim=1, mode="c"] mass not None ,
    np.ndarray[long  , ndim=1, mode="c"] loopnb not None ,
    np.ndarray[long  , ndim=2, mode="c"] Targets not None ,
    np.ndarray[double, ndim=1, mode="c"] MassSum not None ,
    np.ndarray[double, ndim=4, mode="c"] SpaceRotsUn not None ,
    np.ndarray[long  , ndim=2, mode="c"] TimeRevsUn not None ,
    np.ndarray[long  , ndim=2, mode="c"] TimeShiftNumUn not None ,
    np.ndarray[long  , ndim=2, mode="c"] TimeShiftDenUn not None ,
    np.ndarray[long  , ndim=1, mode="c"] loopncstr not None ,
    np.ndarray[double, ndim=4, mode="c"] SpaceRotsCstr not None ,
    np.ndarray[long  , ndim=2, mode="c"] TimeRevsCstr not None ,
    np.ndarray[long  , ndim=2, mode="c"] TimeShiftNumCstr not None ,
    np.ndarray[long  , ndim=2, mode="c"] TimeShiftDenCstr not None
    ):
    # Assembles the matrix of constraints used to select constraint satisfying parameters

#~     cdef double eps_zero = 1e-14
    cdef double eps_zero = 1e-10
    
    # il,idim,k,ift => ift + 2*(k + ncoeff*(idim + ndim*il))

    cdef long nnz = 0
    cdef long il,idim,jdim,ib,k,i
    cdef long ilcstr
    
    cdef double val,dt
    cdef double masstot=0
    cdef double invmasstot
    cdef np.ndarray[double, ndim=1, mode="c"] cs = np.zeros((2),dtype=np.float64)
    
    # Removes imaginary part of c_0
    for il in range(nloop):
        for idim in range(cndim):
             
            nnz +=1
    
    # Zero momentum constraint
    if MomCons :
        
        for il in range(nloop):
            for ib in range(loopnb[il]):
                
                masstot += mass[Targets[il,ib]]
                
        invmasstot = cpow(masstot,-1)
        
        for k in range(ncoeff):
            for idim in range(cndim):
                                      
                for il in range(nloop):
                    for ib in range(loopnb[il]):
                        
                        dt = TimeShiftNumUn[il,ib] / TimeShiftDenUn[il,ib]
                        cs[0] = ccos( - ctwopi * k*dt)
                        cs[1] = csin( - ctwopi * k*dt)  
                        
                        for jdim in range(cndim):

# ~                             val = SpaceRotsUn[il,ib,idim,jdim]*cs[0]*mass[Targets[il,ib]]
                            val = SpaceRotsUn[il,ib,idim,jdim]*cs[0]*mass[Targets[il,ib]]*invmasstot

                            if (cfabs(val) > eps_zero):
                            
                                nnz +=1

# ~                             val = -TimeRevsUn[il,ib]*SpaceRotsUn[il,ib,idim,jdim]*cs[1]*mass[Targets[il,ib]]
                            val = -TimeRevsUn[il,ib]*SpaceRotsUn[il,ib,idim,jdim]*cs[1]*mass[Targets[il,ib]]*invmasstot

                            if (cfabs(val) > eps_zero):
                            
                                nnz +=1
                    
                for il in range(nloop):
                    for ib in range(loopnb[il]):
                        
                        dt = TimeShiftNumUn[il,ib] / TimeShiftDenUn[il,ib]
                        cs[0] = ccos( - ctwopi * k*dt)
                        cs[1] = csin( - ctwopi * k*dt)  
                        
                        for jdim in range(cndim):
                                
# ~                             val = SpaceRotsUn[il,ib,idim,jdim]*cs[1]*mass[Targets[il,ib]]
                            val = SpaceRotsUn[il,ib,idim,jdim]*cs[1]*mass[Targets[il,ib]]*invmasstot

                            if (cfabs(val) > eps_zero):
                            
                                nnz +=1

# ~                             val = TimeRevsUn[il,ib]*SpaceRotsUn[il,ib,idim,jdim]*cs[0]*mass[Targets[il,ib]]
                            val = TimeRevsUn[il,ib]*SpaceRotsUn[il,ib,idim,jdim]*cs[0]*mass[Targets[il,ib]]*invmasstot

                            if (cfabs(val) > eps_zero):
                            
                                nnz +=1
                                
    # Symmetry constraints on loops
    for il in range(nloop):
        
        for ilcstr in range(loopncstr[il]):
            
            for k in range(ncoeff):
                
                dt = TimeShiftNumCstr[il,ilcstr]/TimeShiftDenCstr[il,ilcstr]
                
                if (TimeRevsCstr[il,ilcstr] == 1):

                    cs[0] = ccos( - ctwopi * k*dt)
                    cs[1] = csin( - ctwopi * k*dt)                        
                        
                    for idim in range(cndim):
                            
                        for jdim in range(cndim):

                            val = SpaceRotsCstr[il,ilcstr,idim,jdim]*cs[0]
                            
                            if (idim == jdim):
                                val -=1.

                            if (cfabs(val) > eps_zero):
                            
                                nnz +=1

                            val = - SpaceRotsCstr[il,ilcstr,idim,jdim]*cs[1]
                            
                            if (cfabs(val) > eps_zero):
                            
                                nnz +=1
                            
                        for jdim in range(cndim):

                            val = SpaceRotsCstr[il,ilcstr,idim,jdim]*cs[0]
                            
                            if (idim == jdim):
                                val -=1.

                            if (cfabs(val) > eps_zero):
                            
                                nnz +=1

                            val = SpaceRotsCstr[il,ilcstr,idim,jdim]*cs[1]
                            
                            if (cfabs(val) > eps_zero):
                            
                                nnz +=1
                                             
                elif (TimeRevsCstr[il,ilcstr] == -1):

                    cs[0] = ccos( ctwopi * k*dt)
                    cs[1] = csin( ctwopi * k*dt)
                    
                    for idim in range(cndim):
                            
                        for jdim in range(cndim):

                            val = SpaceRotsCstr[il,ilcstr,idim,jdim]*cs[0]
                            
                            if (idim == jdim):
                                val -=1.
                            
                            if (cfabs(val) > eps_zero):
                            
                                nnz +=1

                            val = SpaceRotsCstr[il,ilcstr,idim,jdim]*cs[1]
                            
                            if (cfabs(val) > eps_zero):
                            
                                nnz +=1         
                            
                        for jdim in range(cndim):

                            val = - SpaceRotsCstr[il,ilcstr,idim,jdim]*cs[0]
                            
                            if (idim == jdim):
                                val -=1.

                            if (cfabs(val) > eps_zero):
                            
                                nnz +=1

                            val = SpaceRotsCstr[il,ilcstr,idim,jdim]*cs[1]
                            
                            if (cfabs(val) > eps_zero):
                            
                                nnz +=1
                                       
                else:
                    print(TimeRevsCstr[il,ilcstr])
                    raise ValueError("Invalid TimeRev")
    
    cdef np.ndarray[long  , ndim=1, mode="c"] cstr_row  = np.zeros((nnz),dtype=np.int_   )
    cdef np.ndarray[long  , ndim=1, mode="c"] cstr_col  = np.zeros((nnz),dtype=np.int_   )
    cdef np.ndarray[double, ndim=1, mode="c"] cstr_data = np.zeros((nnz),dtype=np.float64)

    cdef long icstr = 0
    nnz = 0

    # Removes imaginary part of c_0
    for il in range(nloop):
        for idim in range(cndim):
            
            i = 1 + 2*(0 + ncoeff*(idim + cndim*il))  
            
            cstr_row[nnz] = i
            cstr_col[nnz] = icstr
            cstr_data[nnz] = 1. 
              
            nnz +=1
            icstr +=1 
    
    # Zero momentum constraint
    if MomCons :
        
        for k in range(ncoeff):
            for idim in range(cndim):
                                      
                for il in range(nloop):
                    for ib in range(loopnb[il]):
                        
                        dt = TimeShiftNumUn[il,ib] / TimeShiftDenUn[il,ib]
                        cs[0] = ccos( - ctwopi * k*dt)
                        cs[1] = csin( - ctwopi * k*dt)  
                        
                        for jdim in range(cndim):
                                
                            i =  0 + 2*(k + ncoeff*(jdim + cndim*il))

# ~                             val = SpaceRotsUn[il,ib,idim,jdim]*cs[0]*mass[Targets[il,ib]]
                            val = SpaceRotsUn[il,ib,idim,jdim]*cs[0]*mass[Targets[il,ib]]*invmasstot

                            if (cfabs(val) > eps_zero):
                            
                                cstr_row[nnz] = i
                                cstr_col[nnz] = icstr
                                cstr_data[nnz] = val 
                            
                                nnz +=1
                                
                            i =  1 + 2*(k + ncoeff*(jdim + cndim*il))

# ~                             val = -TimeRevsUn[il,ib]*SpaceRotsUn[il,ib,idim,jdim]*cs[1]*mass[Targets[il,ib]]
                            val = -TimeRevsUn[il,ib]*SpaceRotsUn[il,ib,idim,jdim]*cs[1]*mass[Targets[il,ib]]*invmasstot

                            if (cfabs(val) > eps_zero):
                            
                                cstr_row[nnz] = i
                                cstr_col[nnz] = icstr
                                cstr_data[nnz] = val 
                            
                                nnz +=1
                                
                icstr +=1
                    
                for il in range(nloop):
                    for ib in range(loopnb[il]):
                        
                        dt = TimeShiftNumUn[il,ib] / TimeShiftDenUn[il,ib]
                        cs[0] = ccos( - ctwopi * k*dt)
                        cs[1] = csin( - ctwopi * k*dt)  
                        
                        for jdim in range(cndim):
                                
                            i =  0 + 2*(k + ncoeff*(jdim + cndim*il))

# ~                             val = SpaceRotsUn[il,ib,idim,jdim]*cs[1]*mass[Targets[il,ib]]
                            val = SpaceRotsUn[il,ib,idim,jdim]*cs[1]*mass[Targets[il,ib]]*invmasstot

                            if (cfabs(val) > eps_zero):
                                                
                                cstr_row[nnz] = i
                                cstr_col[nnz] = icstr
                                cstr_data[nnz] = val
                            
                                nnz +=1
                                
                            i =  1 + 2*(k + ncoeff*(jdim + cndim*il))

# ~                             val = TimeRevsUn[il,ib]*SpaceRotsUn[il,ib,idim,jdim]*cs[0]*mass[Targets[il,ib]]
                            val = TimeRevsUn[il,ib]*SpaceRotsUn[il,ib,idim,jdim]*cs[0]*mass[Targets[il,ib]]*invmasstot

                            if (cfabs(val) > eps_zero):
                                                
                                cstr_row[nnz] = i
                                cstr_col[nnz] = icstr
                                cstr_data[nnz] = val
                            
                                nnz +=1
                                
                icstr +=1
             
    # Symmetry constraints on loops
    for il in range(nloop):
        
        for ilcstr in range(loopncstr[il]):
            
            for k in range(ncoeff):
                
                dt = TimeShiftNumCstr[il,ilcstr]/TimeShiftDenCstr[il,ilcstr]
                
                if (TimeRevsCstr[il,ilcstr] == 1):

                    cs[0] = ccos( - ctwopi * k*dt)
                    cs[1] = csin( - ctwopi * k*dt)                        
                        
                    for idim in range(cndim):
                            
                        for jdim in range(cndim):
                                
                            i =  0 + 2*(k + ncoeff*(jdim + cndim*il))

                            val = SpaceRotsCstr[il,ilcstr,idim,jdim]*cs[0]
                            
                            if (idim == jdim):
                                val -=1.

                            if (cfabs(val) > eps_zero):
                                                
                                cstr_row[nnz] = i
                                cstr_col[nnz] = icstr
                                cstr_data[nnz] = val
                            
                                nnz +=1
                                
                            i =  1 + 2*(k + ncoeff*(jdim + cndim*il))

                            val = - SpaceRotsCstr[il,ilcstr,idim,jdim]*cs[1]
                            
                            if (cfabs(val) > eps_zero):
                            
                                cstr_row[nnz] = i
                                cstr_col[nnz] = icstr
                                cstr_data[nnz] = val
                            
                                nnz +=1
                                
                        icstr+=1
                            
                        for jdim in range(cndim):
                                
                            i =  1 + 2*(k + ncoeff*(jdim + cndim*il))

                            val = SpaceRotsCstr[il,ilcstr,idim,jdim]*cs[0]
                            
                            if (idim == jdim):
                                val -=1.

                            if (cfabs(val) > eps_zero):
                                                
                                cstr_row[nnz] = i
                                cstr_col[nnz] = icstr
                                cstr_data[nnz] = val
                            
                                nnz +=1
                                
                            i =  0 + 2*(k + ncoeff*(jdim + cndim*il))

                            val = SpaceRotsCstr[il,ilcstr,idim,jdim]*cs[1]
                            
                            if (cfabs(val) > eps_zero):
                                                
                                cstr_row[nnz] = i
                                cstr_col[nnz] = icstr
                                cstr_data[nnz] = val
                            
                                nnz +=1

                        icstr+=1
                                             
                elif (TimeRevsCstr[il,ilcstr] == -1):

                    cs[0] = ccos( ctwopi * k*dt)
                    cs[1] = csin( ctwopi * k*dt)
                    
                    for idim in range(cndim):
                            
                        for jdim in range(cndim):
                                
                            i =  0 + 2*(k + ncoeff*(jdim + cndim*il))

                            val = SpaceRotsCstr[il,ilcstr,idim,jdim]*cs[0]
                            
                            if (idim == jdim):
                                val -=1.
                            
                            if (cfabs(val) > eps_zero):
                            
                                cstr_row[nnz] = i
                                cstr_col[nnz] = icstr
                                cstr_data[nnz] = val
                            
                                nnz +=1
                                
                            i =  1 + 2*(k + ncoeff*(jdim + cndim*il))

                            val = SpaceRotsCstr[il,ilcstr,idim,jdim]*cs[1]
                            
                            if (cfabs(val) > eps_zero):
                                                
                                cstr_row[nnz] = i
                                cstr_col[nnz] = icstr
                                cstr_data[nnz] = val
                            
                                nnz +=1         
                                
                        icstr+=1
                            
                        for jdim in range(cndim):
                                
                            i =  1 + 2*(k + ncoeff*(jdim + cndim*il))

                            val = - SpaceRotsCstr[il,ilcstr,idim,jdim]*cs[0]
                            
                            if (idim == jdim):
                                val -=1.

                            if (cfabs(val) > eps_zero):
                                                    
                                cstr_row[nnz] = i
                                cstr_col[nnz] = icstr
                                cstr_data[nnz] = val
                            
                                nnz +=1
                                
                            i =  0 + 2*(k + ncoeff*(jdim + cndim*il))

                            val = SpaceRotsCstr[il,ilcstr,idim,jdim]*cs[1]
                            
                            if (cfabs(val) > eps_zero):
                                                    
                                cstr_row[nnz] = i
                                cstr_col[nnz] = icstr
                                cstr_data[nnz] = val 
                            
                                nnz +=1

                        icstr+=1
                                       
                else:
                    print(TimeRevsCstr[il,ilcstr])
                    raise ValueError("Invalid TimeRev")

    cdef long n_idx = nloop*cndim*ncoeff*2

    return  sp.coo_matrix((cstr_data,(cstr_row,cstr_col)),shape=(n_idx,icstr), dtype=np.float64)
    
@cython.cdivision(True)
def diag_changevar(
    long nnz,
    long ncoeff,
    double n_grad_change,
    int [:] idxarray not None,
    double [:] data not None,
    double [:] MassSum not None ,
    ):
    
    cdef long idx, res, ift, k , il, idim
    cdef double kfac,kd
        
    for idx in range(nnz):

        ift = idxarray[idx]%2
        res = idxarray[idx]/2
    
        k = res % ncoeff
        res = res / ncoeff
                
        idim = res % cndim
        il = res / cndim

        if (k >=2):
            kd = k*cpow(MassSum[il],0.5)
# ~             kd = k*MassSum[il]
# ~             kd = k
            kfac = cpow(kd,n_grad_change)
        else:
            kfac = 1.
        
        data[idx] *= kfac
    
def Compute_square_dist(
    np.ndarray[double, ndim=1, mode="c"] x not None ,
    np.ndarray[double, ndim=1, mode="c"] y not None ,
    long s
    ):
        
    cdef double diff
    cdef double res = 0.
    cdef long i
    
    for i in range(s):
        
        diff = x[i]-y[i]
        res+=diff*diff
    
    return res
