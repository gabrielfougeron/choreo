'''
Choreo_cython_funs.pyx : Defines useful compiled functions in the choreo project.

The functions in this file are (as much as possible) written is Cython.
They will be cythonized (i.e. processed by Cython into a C code, which will be compiled ) in setup.py.

Hence, in this file, performance is favored against readability or ease of use.

This file also defines global constants in both C and Python format like the potential law, etc ...


'''

import os
import numpy as np
cimport numpy as np
np.import_array()

cimport cython

try:
    from choreo.Choreo_cython_funs_parallel import Compute_action_Cython_time_loop, Compute_action_hess_mul_Cython_time_loop
except:
    from choreo.Choreo_cython_funs_serial import Compute_action_Cython_time_loop, Compute_action_hess_mul_Cython_time_loop



import scipy.sparse

from libc.math cimport pow as cpow
from libc.math cimport fabs as cfabs
from libc.math cimport cos as ccos
from libc.math cimport sin as csin
from libc.math cimport sqrt as csqrt
from libc.math cimport isnan as cisnan
from libc.math cimport isinf as cisinf


try:

    import mkl_fft._numpy_fft

    the_rfft  = mkl_fft._numpy_fft.rfft
    the_irfft = mkl_fft._numpy_fft.irfft

except:

    try:

        import scipy.fft

        the_rfft = scipy.fft.rfft
        the_irfft = scipy.fft.irfft

    except:

        the_rfft = np.fft.rfft
        the_irfft = np.fft.irfft


cdef double cn = -0.5  #coeff of x^2 in the potential power law
cdef double cnm1 = cn-1
cdef double cnm2 = cn-2

cdef double ctwopi = 2* np.pi
cdef double ctwopisqrt2 = ctwopi*csqrt(2.)
cdef double cfourpi = 4 * np.pi
cdef double cfourpisq = ctwopi*ctwopi

cdef double cnnm1 = cn*(cn-1)

cdef double cmn = -cn
cdef double cmnnm1 = -cnnm1

# cdef  np.ndarray[double, ndim=1, mode="c"] hash_exps = np.array([0.3,0.4,0.6],dtype=np.float64)
# cdef long cnhash = hash_exps.size
# Unfortunately, Cython does not allow that. Let's do it manually then

cdef double hash_exp0 = -0.5
cdef double hash_exp1 = -0.2
cdef double hash_exp2 = -0.4
cdef double hash_exp3 = -0.6
cdef double hash_exp4 = -0.8
# ~ cdef double hash_exp5 = -0.49
# ~ cdef double hash_exp6 = -0.501
# ~ cdef double hash_exp7 = -0.499
# ~ cdef double hash_exp8 = -0.5001
# ~ cdef double hash_exp9 = -0.4999
# ~ cdef double hash_exp10 = -0.50001
# ~ cdef double hash_exp11 = -0.49999
# ~ cdef double hash_exp12 = -0.5

# ~ cdef long cnhash = 3
cdef long cnhash = 5
# cdef long cnhash = 13

# Python definition of the very same variables

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

@cython.profile(False)
@cython.linetrace(False)
cdef inline (double, double, double) CCpt_interbody_pot(double xsq):  # xsq is the square of the distance between two bodies !
    # Cython definition of the potential law
    
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
    
    cdef double[::1] hash_pots = np.zeros((cnhash),dtype=np.float64)

    hash_pots[0] = -cpow(xsq,hash_exp0)
    hash_pots[1] = -cpow(xsq,hash_exp1)
    hash_pots[2] = -cpow(xsq,hash_exp2)
    hash_pots[3] = -cpow(xsq,hash_exp3)
    hash_pots[4] = -cpow(xsq,hash_exp4)
    # hash_pots[5] = -cpow(xsq,hash_exp5)
    # hash_pots[6] = -cpow(xsq,hash_exp6)
    # hash_pots[7] = -cpow(xsq,hash_exp7)
    # hash_pots[8] = -cpow(xsq,hash_exp8)
    # hash_pots[9] = -cpow(xsq,hash_exp9)
    # hash_pots[10] = -cpow(xsq,hash_exp10)
    # hash_pots[11] = -cpow(xsq,hash_exp11)
    # hash_pots[12] = -cpow(xsq,hash_exp12)
    
    return hash_pots
    

    
@cython.cdivision(True)
def Compute_action_Cython(
    long nloop                          ,
    long ncoeff                         ,
    long nint                           ,
    double[::1]       mass              ,
    long[::1]         loopnb            ,
    long[:,::1]       Targets           ,
    double[::1]       MassSum           ,
    double[:,:,:,::1] SpaceRotsUn       ,
    long[:,::1]       TimeRevsUn        ,
    long[:,::1]       TimeShiftNumUn    ,
    long[:,::1]       TimeShiftDenUn    ,
    long[::1]         loopnbi           ,
    double[:,::1]     ProdMassSumAll    ,
    double[:,:,:,::1] SpaceRotsBin      ,
    long[:,::1]       TimeRevsBin       ,
    long[:,::1]       TimeShiftNumBin   ,
    long[:,::1]       TimeShiftDenBin   ,
    double[:,:,:,::1] all_coeffs        ,
    double[:,:,::1]   all_pos 
):
    # This function is probably the most important one.
    # Computes the action and its gradient with respect to the Fourier coefficients of the generator in each loop.
    
    cdef Py_ssize_t geodim = all_pos.shape[1]

    cdef Py_ssize_t il,ilp,i
    cdef Py_ssize_t idim,jdim
    cdef Py_ssize_t ibi
    cdef Py_ssize_t ib,ibp
    cdef Py_ssize_t iint
    cdef Py_ssize_t k
    cdef long k2
    cdef double prod_mass,a,b,dx2,prod_fac

    cdef double Pot_en
    cdef double[:,:,::1] grad_pot_all

    Pot_en, grad_pot_all = Compute_action_Cython_time_loop(
        nloop             ,
        nint              ,
        mass              ,
        loopnb            ,
        Targets           ,
        SpaceRotsUn       ,
        TimeRevsUn        ,
        TimeShiftNumUn    ,
        TimeShiftDenUn    ,
        loopnbi           ,
        ProdMassSumAll    ,
        SpaceRotsBin      ,
        TimeRevsBin       ,
        TimeShiftNumBin   ,
        TimeShiftDenBin   ,
        all_pos           
    )

    cdef double complex[:,:,::1]  grad_pot_fft = the_rfft(grad_pot_all,norm="forward")  #
    cdef double Kin_en = 0  #
    cdef np.ndarray[double, ndim=4, mode="c"] Action_grad_np = np.empty((nloop,geodim,ncoeff,2),np.float64)
    cdef double[:,:,:,::1] Action_grad = Action_grad_np #
    for il in range(nloop):
        
        prod_fac = MassSum[il]*cfourpisq
        
        for idim in range(geodim):   #
            Action_grad[il,idim,0,0] = -grad_pot_fft[il,idim,0].real
            Action_grad[il,idim,0,1] = 0    #
            for k in range(1,ncoeff):
                
                k2 = k*k
                a = prod_fac*k2
                b=2*a   #
                Kin_en += a *((all_coeffs[il,idim,k,0]*all_coeffs[il,idim,k,0]) + (all_coeffs[il,idim,k,1]*all_coeffs[il,idim,k,1]))
                
                Action_grad[il,idim,k,0] = b*all_coeffs[il,idim,k,0] - 2*grad_pot_fft[il,idim,k].real
                Action_grad[il,idim,k,1] = b*all_coeffs[il,idim,k,1] - 2*grad_pot_fft[il,idim,k].imag
            
    Action = Kin_en-Pot_en
    
    return Action,Action_grad_np
    

@cython.cdivision(True)
def Compute_action_hess_mul_Cython(
    long nloop                          ,
    long ncoeff                         ,
    long nint                           ,
    double[::1]       mass              ,
    long[::1]         loopnb            ,
    long[:,::1]       Targets           ,
    double[::1]       MassSum           ,
    double[:,:,:,::1] SpaceRotsUn       ,
    long[:,::1]       TimeRevsUn        ,
    long[:,::1]       TimeShiftNumUn    ,
    long[:,::1]       TimeShiftDenUn    ,
    long[::1]         loopnbi           ,
    double[:,::1]     ProdMassSumAll    ,
    double[:,:,:,::1] SpaceRotsBin      ,
    long[:,::1]       TimeRevsBin       ,
    long[:,::1]       TimeShiftNumBin   ,
    long[:,::1]       TimeShiftDenBin   ,
    double[:,:,:,::1] all_coeffs        ,
    np.ndarray[double, ndim=4, mode="c"]  all_coeffs_d  , # required
    double[:,:,::1]   all_pos 
):
    # Computes the matrix vector product H*dx where H is the Hessian of the action.
    # Useful to guide the root finding / optimisation process and to better understand the topography of the action (critical points / Morse theory).

    cdef long geodim = all_coeffs.shape[1]

    cdef Py_ssize_t il,ilp,i
    cdef Py_ssize_t idim,jdim
    cdef Py_ssize_t ibi
    cdef Py_ssize_t ib,ibp
    cdef Py_ssize_t iint
    cdef Py_ssize_t k
    cdef long k2

    cdef double prod_mass,a,b,c,dx2,prod_fac,dxtddx

    c_coeffs_d = all_coeffs_d.view(dtype=np.complex128)[...,0]
    cdef double[:,:,::1]  all_pos_d = the_irfft(c_coeffs_d,n=nint,axis=2,norm="forward")

    cdef double[:,:,::1] hess_pot_all_d

    hess_pot_all_d = Compute_action_hess_mul_Cython_time_loop(
        nloop             ,
        nint              ,
        mass              ,
        loopnb            ,
        Targets           ,
        SpaceRotsUn       ,
        TimeRevsUn        ,
        TimeShiftNumUn    ,
        TimeShiftDenUn    ,
        loopnbi           ,
        ProdMassSumAll    ,
        SpaceRotsBin      ,
        TimeRevsBin       ,
        TimeShiftNumBin   ,
        TimeShiftDenBin   ,
        all_pos           ,
        all_pos_d         
    )


    cdef double complex[:,:,::1]  hess_dx_pot_fft = the_rfft(hess_pot_all_d,norm="forward")

    cdef np.ndarray[double, ndim=4, mode="c"] Action_hess_dx_np = np.empty((nloop,geodim,ncoeff,2),np.float64)
    cdef double[:,:,:,::1] Action_hess_dx = Action_hess_dx_np

    for il in range(nloop):
        
        prod_fac = MassSum[il]*cfourpisq
        
        for idim in range(geodim):
            
            Action_hess_dx[il,idim,0,0] = -hess_dx_pot_fft[il,idim,0].real
            Action_hess_dx[il,idim,0,1] = 0 

            for k in range(1,ncoeff):
                
                k2 = k*k
                a = 2*prod_fac*k2
                
                Action_hess_dx[il,idim,k,0] = a*all_coeffs_d[il,idim,k,0] - 2*hess_dx_pot_fft[il,idim,k].real
                Action_hess_dx[il,idim,k,1] = a*all_coeffs_d[il,idim,k,1] - 2*hess_dx_pot_fft[il,idim,k].imag


    return Action_hess_dx_np
    



def Compute_hash_action_Cython(
    long                geodim          ,
    long                nloop           ,
    long                ncoeff          ,
    long                nint            ,
    double[::1]         mass            ,
    long[::1]           loopnb          ,
    long[:,::1]         Targets         ,
    double[::1]         MassSum         ,
    double[:,:,:,::1]   SpaceRotsUn     ,
    long[:,::1]         TimeRevsUn      ,
    long[:,::1]         TimeShiftNumUn  ,
    long[:,::1]         TimeShiftDenUn  ,
    long[::1]           loopnbi         ,
    double[:,::1]       ProdMassSumAll  ,
    double[:,:,:,::1]   SpaceRotsBin    ,
    long[:,::1]         TimeRevsBin     ,
    long[:,::1]         TimeShiftNumBin ,
    long[:,::1]         TimeShiftDenBin ,
    np.ndarray[double, ndim=4, mode="c"]   all_coeffs  
):
    # Computes the hash of a set of trajectories.
    # The hash is meant to provide a likely unique short identification for duplicate detection.
    # It is hence engineered to be invariant wrt permutation of bodies, time shifts / reversals and space isometries.

    cdef long il,ilp,i
    cdef long idim,idimp
    cdef long ibi
    cdef long ib,ibp
    cdef long iint
    cdef long k,k2
    cdef long ddiv,rem
    cdef double pot,potp,potpp
    cdef double prod_mass,a,b,dx2,prod_fac
    cdef double[::1] dx = np.zeros((geodim),dtype=np.float64)
        
    cdef long maxloopnb = 0
    cdef long maxloopnbi = 0

    for il in range(nloop):
        if (maxloopnb < loopnb[il]):
            maxloopnb = loopnb[il]
        if (maxloopnbi < loopnbi[il]):
            maxloopnbi = loopnbi[il]
    
    cdef long ihash
    
    cdef np.ndarray[double, ndim=1, mode="c"] Hash_En = np.zeros((cnhash),dtype=np.float64)
    cdef double[::1] Hash_pot = np.zeros((cnhash),dtype=np.float64)

    cdef double Kin_en = 0

    for il in range(nloop):
        
        prod_fac = MassSum[il]*cfourpisq
        
        for idim in range(geodim):
            for k in range(1,ncoeff):
                
                k2 = k*k
                a = prod_fac*k2

                Kin_en += a *((all_coeffs[il,idim,k,0]*all_coeffs[il,idim,k,0]) + (all_coeffs[il,idim,k,1]*all_coeffs[il,idim,k,1]))

    c_coeffs = all_coeffs.view(dtype=np.complex128)[...,0]
    cdef np.ndarray[double, ndim=3, mode="c"] all_pos = the_irfft(c_coeffs,n=nint,axis=2,norm="forward")

    cdef np.ndarray[long, ndim=2, mode="c"]  all_shiftsUn = np.zeros((nloop,maxloopnb),dtype=np.int_)
    cdef np.ndarray[long, ndim=2, mode="c"]  all_shiftsBin = np.zeros((nloop,maxloopnbi),dtype=np.int_)
    
    for il in range(nloop):
        for ib in range(loopnb[il]):

            k = (TimeRevsUn[il,ib]*nint*TimeShiftNumUn[il,ib])

            ddiv = - k // TimeShiftDenUn[il,ib]
            rem = k + ddiv * TimeShiftDenUn[il,ib]

            all_shiftsUn[il,ib] = (((ddiv) % nint) + nint) % nint
        
        for ibi in range(loopnbi[il]):

            k = (TimeRevsBin[il,ibi]*nint*TimeShiftNumBin[il,ibi])

            ddiv = - k // TimeShiftDenBin[il,ibi]
            rem = k + ddiv * TimeShiftDenBin[il,ibi]

            all_shiftsBin[il,ibi] = (((ddiv) % nint) + nint) % nint
    
    cdef np.ndarray[double, ndim=3, mode="c"] grad_pot_all = np.zeros((nloop,geodim,nint),dtype=np.float64)

    for iint in range(nint):

        # Different loops
        for il in range(nloop):
            for ilp in range(il+1,nloop):

                for ib in range(loopnb[il]):
                    for ibp in range(loopnb[ilp]):
                        
                        prod_mass = mass[Targets[il,ib]]*mass[Targets[ilp,ibp]]

                        for idim in range(geodim):
                            dx[idim] = SpaceRotsUn[il,ib,idim,0]*all_pos[il,0,all_shiftsUn[il,ib]] - SpaceRotsUn[ilp,ibp,idim,0]*all_pos[ilp,0,all_shiftsUn[ilp,ibp]]
                            for jdim in range(1,geodim):
                                dx[idim] += SpaceRotsUn[il,ib,idim,jdim]*all_pos[il,jdim,all_shiftsUn[il,ib]] - SpaceRotsUn[ilp,ibp,idim,jdim]*all_pos[ilp,jdim,all_shiftsUn[ilp,ibp]]

                        dx2 = dx[0]*dx[0]
                        for idim in range(1,geodim):
                            dx2 += dx[idim]*dx[idim]
                            
                        Hash_pot = CCpt_hash_pot(dx2)
                        
                        for ihash in range(cnhash):
                            Hash_En[ihash] += Hash_pot[ihash] * prod_mass

        # Same loop + symmetry
        for il in range(nloop):

            for ibi in range(loopnbi[il]):
                
                for idim in range(geodim):
                    dx[idim] = SpaceRotsBin[il,ibi,idim,0]*all_pos[il,0,all_shiftsBin[il,ibi]]
                    for jdim in range(1,geodim):
                        dx[idim] += SpaceRotsBin[il,ibi,idim,jdim]*all_pos[il,jdim,all_shiftsBin[il,ibi]]
                    
                    dx[idim] -= all_pos[il,idim,iint]
                    
                dx2 = dx[0]*dx[0]
                for idim in range(1,geodim):
                    dx2 += dx[idim]*dx[idim]

                Hash_pot = CCpt_hash_pot(dx2)
                
                for ihash in range(cnhash):
                    Hash_En[ihash] += Hash_pot[ihash] * ProdMassSumAll[il,ibi]


        # Increments time at the end
        for il in range(nloop):
            for ib in range(loopnb[il]):
                all_shiftsUn[il,ib] = (((all_shiftsUn[il,ib]+TimeRevsUn[il,ib]) % nint ) + nint) % nint
                
            for ibi in range(loopnbi[il]):
                all_shiftsBin[il,ibi] = (((all_shiftsBin[il,ibi]+TimeRevsBin[il,ibi]) % nint ) + nint) % nint

    for ihash in range(cnhash):
        Hash_En[ihash] = Kin_en - Hash_En[ihash]/nint

    return Hash_En
    
def Compute_MinDist_Cython(
    long                nloop           ,
    long                ncoeff          ,
    long                nint            ,
    double[::1]         mass            ,
    long[::1]           loopnb          ,
    long[:,::1]         Targets         ,
    double[::1]         MassSum         ,
    double[:,:,:,::1]   SpaceRotsUn     ,
    long[:,::1]         TimeRevsUn      ,
    long[:,::1]         TimeShiftNumUn  ,
    long[:,::1]         TimeShiftDenUn  ,
    long[::1]           loopnbi         ,
    double[:,::1]       ProdMassSumAll  ,
    double[:,:,:,::1]   SpaceRotsBin    ,
    long[:,::1]         TimeRevsBin     ,
    long[:,::1]         TimeShiftNumBin ,
    long[:,::1]         TimeShiftDenBin ,
    np.ndarray[double, ndim=4, mode="c"] all_coeffs   
):
    # Computes the minimum inter-body distance along the trajectory.
    # A useful tool for collision detection.

    cdef long geodim = SpaceRotsUn.shape[2]

    cdef long il,ilp,i
    cdef long idim,idimp
    cdef long ibi
    cdef long ib,ibp
    cdef long iint
    cdef long k,k2
    cdef long ddiv,rem
    cdef double pot,potp,potpp
    cdef double prod_mass,a,b,dx2,prod_fac
    cdef np.ndarray[double, ndim=1, mode="c"]  dx = np.zeros((geodim),dtype=np.float64)
        
    cdef long maxloopnb = 0
    cdef long maxloopnbi = 0

    for il in range(nloop):
        if (maxloopnb < loopnb[il]):
            maxloopnb = loopnb[il]
        if (maxloopnbi < loopnbi[il]):
            maxloopnbi = loopnbi[il]
    
    cdef double dx2min = 1e100

    c_coeffs = all_coeffs.view(dtype=np.complex128)[...,0]

    cdef np.ndarray[double, ndim=3, mode="c"] all_pos = the_irfft(c_coeffs,n=nint,axis=2,norm="forward")

    cdef np.ndarray[long, ndim=2, mode="c"]  all_shiftsUn = np.zeros((nloop,maxloopnb),dtype=np.int_)
    cdef np.ndarray[long, ndim=2, mode="c"]  all_shiftsBin = np.zeros((nloop,maxloopnbi),dtype=np.int_)
    
    for il in range(nloop):
        for ib in range(loopnb[il]):

            k = (TimeRevsUn[il,ib]*nint*TimeShiftNumUn[il,ib])

            ddiv = - k // TimeShiftDenUn[il,ib]
            rem = k + ddiv * TimeShiftDenUn[il,ib]

            all_shiftsUn[il,ib] = (((ddiv) % nint) + nint) % nint
        
        for ibi in range(loopnbi[il]):

            k = (TimeRevsBin[il,ibi]*nint*TimeShiftNumBin[il,ibi])

            ddiv = - k // TimeShiftDenBin[il,ibi]
            rem = k + ddiv * TimeShiftDenBin[il,ibi]

            all_shiftsBin[il,ibi] = (((ddiv) % nint) + nint) % nint

    for iint in range(nint):

        # Different loops
        for il in range(nloop):
            for ilp in range(il+1,nloop):

                for ib in range(loopnb[il]):
                    for ibp in range(loopnb[ilp]):
                        
                        for idim in range(geodim):
                            dx[idim] = SpaceRotsUn[il,ib,idim,0]*all_pos[il,0,all_shiftsUn[il,ib]] - SpaceRotsUn[ilp,ibp,idim,0]*all_pos[ilp,0,all_shiftsUn[ilp,ibp]]
                            for jdim in range(1,geodim):
                                dx[idim] += SpaceRotsUn[il,ib,idim,jdim]*all_pos[il,jdim,all_shiftsUn[il,ib]] - SpaceRotsUn[ilp,ibp,idim,jdim]*all_pos[ilp,jdim,all_shiftsUn[ilp,ibp]]

                        dx2 = dx[0]*dx[0]
                        for idim in range(1,geodim):
                            dx2 += dx[idim]*dx[idim]
                            
                        if (dx2 < dx2min):
                            dx2min = dx2
                            
        # Same loop + symmetry
        for il in range(nloop):

            for ibi in range(loopnbi[il]):
                
                for idim in range(geodim):
                    dx[idim] = SpaceRotsBin[il,ibi,idim,0]*all_pos[il,0,all_shiftsBin[il,ibi]]
                    for jdim in range(1,geodim):
                        dx[idim] += SpaceRotsBin[il,ibi,idim,jdim]*all_pos[il,jdim,all_shiftsBin[il,ibi]]
                    
                    dx[idim] -= all_pos[il,idim,iint]
                    
                dx2 = dx[0]*dx[0]
                for idim in range(1,geodim):
                    dx2 += dx[idim]*dx[idim]
                    
                if (dx2 < dx2min):
                    dx2min = dx2

        # Increments time at the end
        for il in range(nloop):
            for ib in range(loopnb[il]):
                all_shiftsUn[il,ib] = (((all_shiftsUn[il,ib]+TimeRevsUn[il,ib]) % nint ) + nint) % nint
                
            for ibi in range(loopnbi[il]):
                all_shiftsBin[il,ibi] = (((all_shiftsBin[il,ibi]+TimeRevsBin[il,ibi]) % nint ) + nint) % nint

    return csqrt(dx2min)
   
def Compute_Loop_Dist_Cython(
    long                nloop           ,
    long                ncoeff          ,
    long                nint            ,
    double[::1]         mass            ,
    long[::1]           loopnb          ,
    long[:,::1]         Targets         ,
    double[::1]         MassSum         ,
    double[:,:,:,::1]   SpaceRotsUn     ,
    long[:,::1]         TimeRevsUn      ,
    long[:,::1]         TimeShiftNumUn  ,
    long[:,::1]         TimeShiftDenUn  ,
    long[::1]           loopnbi         ,
    double[:,::1]       ProdMassSumAll  ,
    double[:,:,:,::1]   SpaceRotsBin    ,
    long[:,::1]         TimeRevsBin     ,
    long[:,::1]         TimeShiftNumBin ,
    long[:,::1]         TimeShiftDenBin ,
    double[:,:,:,::1]   all_coeffs  
):

    cdef long geodim = all_coeffs.shape[1]
        
    cdef long il,ilp,i
    cdef long idim,idimp
    cdef long ibi
    cdef long ib,ibp
    cdef long iint
    cdef long k,k2
    cdef double sum_loop_dist2
    cdef double dx2
    cdef double[::1] dx = np.zeros((geodim),dtype=np.float64)

    sum_loop_dist2 = 0.
    for il in range(nloop-1):
        for ilp in range(il,nloop):
            
            for ib in range(loopnb[il]):
                for ibp in range(loopnb[ilp]):
                    
                    for idim in range(geodim):
                        dx[idim] = SpaceRotsUn[il,ib,idim,0]*all_coeffs[il,0,0,0] - SpaceRotsUn[ilp,ibp,idim,0]*all_coeffs[ilp,0,0,0]
                    
                        for jdim in range(1,geodim):
                            dx[idim] += SpaceRotsUn[il,ib,idim,jdim]*all_coeffs[il,jdim,0,0] - SpaceRotsUn[ilp,ibp,idim,jdim]*all_coeffs[ilp,jdim,0,0]
                            
                    dx2 = dx[0]*dx[0]
                    for idim in range(1,geodim):
                        dx2 += dx[idim]*dx[idim]

                    sum_loop_dist2 += dx2

    for il in range(nloop):
        for ibi in range(loopnbi[il]):
                
            for idim in range(geodim):
                
                dx[idim] = SpaceRotsBin[il,ibi,idim,0]*all_coeffs[il,0,0,0] - all_coeffs[il,idim,0,0]
                for jdim in range(1,geodim):
                    
                    dx[idim] += SpaceRotsBin[il,ibi,idim,jdim]*all_coeffs[il,jdim,0,0]
                    
                
                dx2 = dx[0]*dx[0]
                for idim in range(1,geodim):
                    dx2 += dx[idim]*dx[idim]

                sum_loop_dist2 += dx2

    return csqrt(sum_loop_dist2)
   
def Compute_Loop_Dist_btw_avg_Cython(
    long                nloop           ,
    long                ncoeff          ,
    long                nint            ,
    double[::1]         mass            ,
    long[::1]           loopnb          ,
    long[:,::1]         Targets         ,
    double[::1]         MassSum         ,
    double[:,:,:,::1]   SpaceRotsUn     ,
    long[:,::1]         TimeRevsUn      ,
    long[:,::1]         TimeShiftNumUn  ,
    long[:,::1]         TimeShiftDenUn  ,
    long[::1]           loopnbi         ,
    double[:,::1]       ProdMassSumAll  ,
    double[:,:,:,::1]   SpaceRotsBin    ,
    long[:,::1]         TimeRevsBin     ,
    long[:,::1]         TimeShiftNumBin ,
    long[:,::1]         TimeShiftDenBin ,
    double[:,:,:,::1]   all_coeffs  
):

    cdef long geodim = all_coeffs.shape[1]

    cdef long il,ilp,i
    cdef long idim,idimp
    cdef long ibi
    cdef long ib,ibp
    cdef long iint
    cdef long k,k2
    cdef double sum_loop_dist2
    cdef double dx2
    cdef double[::1] dx = np.zeros((geodim),dtype=np.float64)

    sum_loop_dist2 = 0.
    for il in range(nloop-1):
        
        dx2 = all_coeffs[il,0,0,0]*all_coeffs[il,0,0,0]
        for idim in range(1,geodim):
            dx2 += all_coeffs[il,idim,0,0]*all_coeffs[il,idim,0,0]

        sum_loop_dist2 += dx2


    return csqrt(sum_loop_dist2)
   
def Compute_Loop_Size_Dist_Cython(
    long                nloop           ,
    long                ncoeff          ,
    long                nint            ,
    double[::1]         mass            ,
    long[::1]           loopnb          ,
    long[:,::1]         Targets         ,
    double[::1]         MassSum         ,
    double[:,:,:,::1]   SpaceRotsUn     ,
    long[:,::1]         TimeRevsUn      ,
    long[:,::1]         TimeShiftNumUn  ,
    long[:,::1]         TimeShiftDenUn  ,
    long[::1]           loopnbi         ,
    double[:,::1]       ProdMassSumAll  ,
    double[:,:,:,::1]   SpaceRotsBin    ,
    long[:,::1]         TimeRevsBin     ,
    long[:,::1]         TimeShiftNumBin ,
    long[:,::1]         TimeShiftDenBin ,
    double[:,:,:,::1]   all_coeffs  
):

    cdef long geodim = all_coeffs.shape[1]

    cdef long il,ilp,i
    cdef long idim,idimp
    cdef long ibi
    cdef long ib,ibp
    cdef long iint
    cdef long k,k2
    cdef double loop_size,max_loop_size
    cdef double loop_dist,max_loop_dist
    cdef double dx2
    cdef double[::1] dx = np.zeros((geodim),dtype=np.float64)
    
    cdef np.ndarray[double, ndim=1, mode="c"]  res = np.zeros((2),dtype=np.float64)


    max_loop_size = 0.
    for il in range(nloop):
        
        loop_size = 0
        
        for idim in range(geodim):
            for k in range(1,ncoeff):
                
                loop_size += all_coeffs[il,idim,k,0]*all_coeffs[il,idim,k,0]+all_coeffs[il,idim,k,1]*all_coeffs[il,idim,k,1]
        
        if (loop_size > max_loop_size):
            max_loop_size = loop_size
            
    res[0] = csqrt(max_loop_size)


    # TODO : The values computed for res[1] don't seem to be invariant. Why ? Is there a bug or is the definition crappy ?
 
    max_loop_dist = 0.
    for il in range(nloop-1):
        for ilp in range(il,nloop):
            
            for ib in range(loopnb[il]):
                for ibp in range(loopnb[ilp]):
                    
                    for idim in range(geodim):
                        dx[idim] = SpaceRotsUn[il,ib,idim,0]*all_coeffs[il,0,0,0] - SpaceRotsUn[ilp,ibp,idim,0]*all_coeffs[ilp,0,0,0]
                    
                        for jdim in range(1,geodim):
                            dx[idim] += SpaceRotsUn[il,ib,idim,jdim]*all_coeffs[il,jdim,0,0] - SpaceRotsUn[ilp,ibp,idim,jdim]*all_coeffs[ilp,jdim,0,0]
                            
                    dx2 = dx[0]*dx[0]
                    for idim in range(1,geodim):
                        dx2 += dx[idim]*dx[idim]

                    if (dx2 > max_loop_dist):
                        max_loop_dist = dx2

    for il in range(nloop):
        for ibi in range(loopnbi[il]):
                
            for idim in range(geodim):
                
                dx[idim] = SpaceRotsBin[il,ibi,idim,0]*all_coeffs[il,0,0,0] - all_coeffs[il,idim,0,0]
                for jdim in range(1,geodim):
                    
                    dx[idim] += SpaceRotsBin[il,ibi,idim,jdim]*all_coeffs[il,jdim,0,0]
                    
                
                dx2 = dx[0]*dx[0]
                for idim in range(1,geodim):
                    dx2 += dx[idim]*dx[idim]

                if (dx2 > max_loop_dist):
                    max_loop_dist = dx2

    res[1] = csqrt(max_loop_dist)
    
    return res
   
def Compute_Newton_err_Cython(
    long nbody                          ,
    long nloop                          ,
    long ncoeff                         ,
    long nint                           ,
    double[::1]         mass            ,
    long[::1]           loopnb          ,
    long[:,::1]         Targets         ,
    double[:,:,:,::1]   SpaceRotsUn     ,
    long[:,::1]         TimeRevsUn      ,
    long[:,::1]         TimeShiftNumUn  ,
    long[:,::1]         TimeShiftDenUn  ,
    np.ndarray[double, ndim=4, mode="c"] all_coeffs  
):
    # Computes the "Newton error", i.e. the deviation wrt to the fundamental theorem of Newtonian dynamics m_i * a_i - \sum_j f_ij = 0
    # If the Newton error is zero, then the trajectory is physical.
    # Under some symmetry hypotheses, this is the Fourier transform of the gradient of the action.
    # Computing it explicitely is a useful safeguard.

    cdef long geodim = SpaceRotsUn.shape[2]

    cdef long il,ilp,i
    cdef long idim,idimp
    cdef long ibi
    cdef long ib,ibp
    cdef long iint
    cdef long k,k2
    cdef long ddiv,rem
    cdef double pot,potp,potpp
    cdef double prod_mass,a,b,dx2,prod_fac
    cdef double[::1] dx = np.zeros((geodim),dtype=np.float64)

    cdef long maxloopnb = 0

    for il in range(nloop):
        if (maxloopnb < loopnb[il]):
            maxloopnb = loopnb[il]
    
    cdef np.ndarray[double, ndim=4, mode="c"] acc_coeff = np.zeros((nloop,geodim,ncoeff,2),dtype=np.float64)

    for il in range(nloop):
        for idim in range(geodim):
            for k in range(ncoeff):
                
                k2 = k*k
                a = k2 *cfourpisq
                acc_coeff[il,idim,k,0] = a*all_coeffs[il,idim,k,0]
                acc_coeff[il,idim,k,1] = a*all_coeffs[il,idim,k,1]
                
    c_acc_coeffs = acc_coeff.view(dtype=np.complex128)[...,0]
    cdef np.ndarray[double, ndim=3, mode="c"] all_acc = the_irfft(c_acc_coeffs,n=nint,axis=2,norm="forward")
    
    cdef np.ndarray[double, ndim=3, mode="c"] all_Newt_err = np.zeros((nbody,geodim,nint),np.float64)
    
    c_coeffs = all_coeffs.view(dtype=np.complex128)[...,0]
    
    cdef np.ndarray[double, ndim=3, mode="c"] all_pos = the_irfft(c_coeffs,n=nint,axis=2,norm="forward")

    cdef np.ndarray[long, ndim=2, mode="c"]  all_shiftsUn = np.zeros((nloop,maxloopnb),dtype=np.int_)
    
    for il in range(nloop):
        for ib in range(loopnb[il]):

            k = (TimeRevsUn[il,ib]*nint*TimeShiftNumUn[il,ib])

            ddiv = - k // TimeShiftDenUn[il,ib]
            rem = k + ddiv * TimeShiftDenUn[il,ib]

            all_shiftsUn[il,ib] = (((ddiv) % nint) + nint) % nint
        
        
    for iint in range(nint):

        for il in range(nloop):
            for ib in range(loopnb[il]):
                for idim in range(geodim):
                    
                    b = SpaceRotsUn[il,ib,idim,0]*all_acc[il,0,all_shiftsUn[il,ib]]
                    for jdim in range(1,geodim):  
                        b += SpaceRotsUn[il,ib,idim,jdim]*all_acc[il,jdim,all_shiftsUn[il,ib]]                  

                    all_Newt_err[Targets[il,ib],idim,iint] -= mass[Targets[il,ib]]*b

        # Different loops
        for il in range(nloop):
            for ilp in range(il+1,nloop):

                for ib in range(loopnb[il]):
                    for ibp in range(loopnb[ilp]):
                        
                        prod_mass = mass[Targets[il,ib]]*mass[Targets[ilp,ibp]]

                        for idim in range(geodim):
                            dx[idim] = SpaceRotsUn[il,ib,idim,0]*all_pos[il,0,all_shiftsUn[il,ib]] - SpaceRotsUn[ilp,ibp,idim,0]*all_pos[ilp,0,all_shiftsUn[ilp,ibp]]
                            for jdim in range(1,geodim):
                                dx[idim] += SpaceRotsUn[il,ib,idim,jdim]*all_pos[il,jdim,all_shiftsUn[il,ib]] - SpaceRotsUn[ilp,ibp,idim,jdim]*all_pos[ilp,jdim,all_shiftsUn[ilp,ibp]]

                        dx2 = dx[0]*dx[0]
                        for idim in range(1,geodim):
                            dx2 += dx[idim]*dx[idim]
                            
                        pot,potp,potpp = CCpt_interbody_pot(dx2)
                        
                        a = (2*prod_mass*potp)

                        for idim in range(geodim):
                                
                            b = a*dx[idim]
                            all_Newt_err[Targets[il ,ib ],idim,iint] += b
                            all_Newt_err[Targets[ilp,ibp],idim,iint] -= b

        # Same Loop
        for il in range(nloop):

            for ib in range(loopnb[il]):
                for ibp in range(ib+1,loopnb[il]):
                    
                    prod_mass = mass[Targets[il,ib]]*mass[Targets[il,ibp]]

                    for idim in range(geodim):
                        dx[idim] = SpaceRotsUn[il,ib,idim,0]*all_pos[il,0,all_shiftsUn[il,ib]] - SpaceRotsUn[il,ibp,idim,0]*all_pos[il,0,all_shiftsUn[il,ibp]]
                        for jdim in range(1,geodim):
                            dx[idim] += SpaceRotsUn[il,ib,idim,jdim]*all_pos[il,jdim,all_shiftsUn[il,ib]] - SpaceRotsUn[il,ibp,idim,jdim]*all_pos[il,jdim,all_shiftsUn[il,ibp]]
                    
                    dx2 = dx[0]*dx[0]
                    for idim in range(1,geodim):
                        dx2 += dx[idim]*dx[idim]
                        
                    pot,potp,potpp = CCpt_interbody_pot(dx2)
                    
                    a = (2*prod_mass*potp)

                    for idim in range(geodim):
                        
                        b = a*dx[idim]
                        all_Newt_err[Targets[il,ib] ,idim,iint] += b
                        all_Newt_err[Targets[il,ibp],idim,iint] -= b

        # Increments time at the end
        for il in range(nloop):
            for ib in range(loopnb[il]):
                all_shiftsUn[il,ib] = (((all_shiftsUn[il,ib]+TimeRevsUn[il,ib]) % nint ) + nint) % nint
                
    return all_Newt_err
                                                                                                                                                                
def Assemble_Cstr_Matrix(
    long                nloop               ,
    long                ncoeff              ,
    bint                MomCons             ,
    double[::1]         mass                ,
    long[::1]           loopnb              ,
    long[:,::1]         Targets             ,
    double[:,:,:,::1]   SpaceRotsUn         ,
    long[:,::1]         TimeRevsUn          ,
    long[:,::1]         TimeShiftNumUn      ,
    long[:,::1]         TimeShiftDenUn      ,
    long[::1]           loopncstr           ,
    double[:,:,:,::1]   SpaceRotsCstr       ,
    long[:,::1]         TimeRevsCstr        ,
    long[:,::1]         TimeShiftNumCstr    ,
    long[:,::1]         TimeShiftDenCstr 
):
    # Assembles the matrix of constraints used to select constraint satisfying parameters

    cdef long geodim = SpaceRotsUn.shape[2]

    # cdef double eps_zero = 1e-14
    cdef double eps_zero = 1e-10
    
    # il,idim,k,ift => ift + 2*(k + ncoeff*(idim + geodim*il))

    cdef long nnz = 0
    cdef long il,idim,jdim,ib,k,i
    cdef long ilcstr
    
    cdef double val,dt
    cdef double masstot = 0
    cdef double invmasstot = 0
    cdef double c,s
    cdef double mul
    
    # Removes imaginary parts of c_0 and c_last
    for il in range(nloop):
        for idim in range(geodim):
             
            nnz += 2
    
    # Zero momentum constraint
    if MomCons :
        
        for il in range(nloop):
            for ib in range(loopnb[il]):
                
                masstot += mass[Targets[il,ib]]
                
        invmasstot = cpow(masstot,-1)
        
        for k in range(ncoeff):

            for idim in range(geodim):
                                      
                for il in range(nloop):
                    for ib in range(loopnb[il]):

                        dt = - TimeShiftNumUn[il,ib] / TimeShiftDenUn[il,ib]
                        c = ccos(ctwopi * k * dt)
                        s = csin(ctwopi * k * dt)  

                        for jdim in range(geodim):

                            mul = SpaceRotsUn[il,ib,idim,jdim]*mass[Targets[il,ib]]*invmasstot
                            val = mul * c

                            if (cfabs(val) > eps_zero):

                                nnz +=1

                            val = - mul * s

                            if (cfabs(val) > eps_zero):

                                nnz +=1

                for il in range(nloop):
                    for ib in range(loopnb[il]):

                        dt = - TimeShiftNumUn[il,ib] / TimeShiftDenUn[il,ib]
                        c = ccos(ctwopi * k * dt)
                        s = csin(ctwopi * k * dt)  

                        for jdim in range(geodim):

                            mul = TimeRevsUn[il,ib] * SpaceRotsUn[il,ib,idim,jdim]*mass[Targets[il,ib]]*invmasstot
                            val = mul * s

                            if (cfabs(val) > eps_zero):

                                nnz +=1
                                
                            val = mul * c

                            if (cfabs(val) > eps_zero):

                                nnz +=1
             
    # Symmetry constraints on loops
    for il in range(nloop):

        for ilcstr in range(loopncstr[il]):
            
            for k in range(ncoeff):
                
                dt = TimeShiftNumCstr[il,ilcstr]/TimeShiftDenCstr[il,ilcstr]
                c = ccos( - ctwopi * k*dt)
                s = csin( - ctwopi * k*dt)                        
                    
                for idim in range(geodim):
                        
                    for jdim in range(geodim):

                        val = SpaceRotsCstr[il,ilcstr,idim,jdim]*c
                        
                        if (idim == jdim):
                            val -=1.

                        if (cfabs(val) > eps_zero):
                        
                            nnz +=1

                        val = - SpaceRotsCstr[il,ilcstr,idim,jdim]*s
                        
                        if (cfabs(val) > eps_zero):
                        
                            nnz +=1
                        
                    for jdim in range(geodim):

                        val = SpaceRotsCstr[il,ilcstr,idim,jdim]*c
                        
                        if (idim == jdim):
                            val -=1.

                        if (cfabs(val) > eps_zero):
                        
                            nnz +=1

                        val = SpaceRotsCstr[il,ilcstr,idim,jdim]*s
                        
                        if (cfabs(val) > eps_zero):
                        
                            nnz +=1
                                             
    cdef np.ndarray[long  , ndim=1, mode="c"] cstr_row  = np.zeros((nnz),dtype=np.int_   )
    cdef np.ndarray[long  , ndim=1, mode="c"] cstr_col  = np.zeros((nnz),dtype=np.int_   )
    cdef np.ndarray[double, ndim=1, mode="c"] cstr_data = np.zeros((nnz),dtype=np.float64)

    cdef long icstr = 0
    nnz = 0

    # Removes imaginary parts of c_0 and c_last
    for il in range(nloop):
        for idim in range(geodim):
            
            i = 1 + 2*(0 + ncoeff*(idim + geodim*il))  
            
            cstr_row[nnz] = i
            cstr_col[nnz] = icstr
            cstr_data[nnz] = 1. 
              
            nnz +=1
            icstr +=1 

            i = 1 + 2*(ncoeff-1 + ncoeff*(idim + geodim*il))  
            
            cstr_row[nnz] = i
            cstr_col[nnz] = icstr
            cstr_data[nnz] = 1. 
              
            nnz +=1
            icstr +=1 

    # Zero momentum constraint
    if MomCons :
        
        for k in range(ncoeff):

            for idim in range(geodim):
                                      
                for il in range(nloop):
                    for ib in range(loopnb[il]):

                        dt = - TimeShiftNumUn[il,ib] / TimeShiftDenUn[il,ib]
                        c = ccos(ctwopi * k * dt)
                        s = csin(ctwopi * k * dt)  

                        for jdim in range(geodim):

                            mul = SpaceRotsUn[il,ib,idim,jdim]*mass[Targets[il,ib]]*invmasstot
                                
                            i =  0 + 2*(k + ncoeff*(jdim + geodim*il))

                            val = mul * c

                            if (cfabs(val) > eps_zero):
                            
                                cstr_row[nnz] = i
                                cstr_col[nnz] = icstr
                                cstr_data[nnz] = val 
                            
                                nnz +=1
                                
                            i =  1 + 2*(k + ncoeff*(jdim + geodim*il))

                            val = - mul * s

                            if (cfabs(val) > eps_zero):
                            
                                cstr_row[nnz] = i
                                cstr_col[nnz] = icstr
                                cstr_data[nnz] = val 
                            
                                nnz +=1

                icstr +=1
                    
                for il in range(nloop):
                    for ib in range(loopnb[il]):

                        dt = - TimeShiftNumUn[il,ib] / TimeShiftDenUn[il,ib]
                        c = ccos(ctwopi * k * dt)
                        s = csin(ctwopi * k * dt)  

                        for jdim in range(geodim):

                            mul = TimeRevsUn[il,ib] * SpaceRotsUn[il,ib,idim,jdim]*mass[Targets[il,ib]]*invmasstot
                                
                            i =  0 + 2*(k + ncoeff*(jdim + geodim*il))

                            val = mul * s

                            if (cfabs(val) > eps_zero):
                                                
                                cstr_row[nnz] = i
                                cstr_col[nnz] = icstr
                                cstr_data[nnz] = val
                            
                                nnz +=1
                                
                            i =  1 + 2*(k + ncoeff*(jdim + geodim*il))

                            val = mul * c

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

                    c = ccos( - ctwopi * k*dt)
                    s = csin( - ctwopi * k*dt)                        
                        
                    for idim in range(geodim):
                            
                        for jdim in range(geodim):
                                
                            i =  0 + 2*(k + ncoeff*(jdim + geodim*il))

                            val = SpaceRotsCstr[il,ilcstr,idim,jdim]*c
                            
                            if (idim == jdim):
                                val -=1.

                            if (cfabs(val) > eps_zero):
                                                
                                cstr_row[nnz] = i
                                cstr_col[nnz] = icstr
                                cstr_data[nnz] = val
                            
                                nnz +=1
                                
                            i =  1 + 2*(k + ncoeff*(jdim + geodim*il))

                            val = - SpaceRotsCstr[il,ilcstr,idim,jdim]*s
                            
                            if (cfabs(val) > eps_zero):
                            
                                cstr_row[nnz] = i
                                cstr_col[nnz] = icstr
                                cstr_data[nnz] = val
                            
                                nnz +=1
                                
                        icstr+=1
                            
                        for jdim in range(geodim):
                                
                            i =  1 + 2*(k + ncoeff*(jdim + geodim*il))

                            val = SpaceRotsCstr[il,ilcstr,idim,jdim]*c
                            
                            if (idim == jdim):
                                val -=1.

                            if (cfabs(val) > eps_zero):
                                                
                                cstr_row[nnz] = i
                                cstr_col[nnz] = icstr
                                cstr_data[nnz] = val
                            
                                nnz +=1
                                
                            i =  0 + 2*(k + ncoeff*(jdim + geodim*il))

                            val = SpaceRotsCstr[il,ilcstr,idim,jdim]*s
                            
                            if (cfabs(val) > eps_zero):
                                                
                                cstr_row[nnz] = i
                                cstr_col[nnz] = icstr
                                cstr_data[nnz] = val
                            
                                nnz +=1

                        icstr+=1
                                             
                elif (TimeRevsCstr[il,ilcstr] == -1):

                    c = ccos( ctwopi * k*dt)
                    s = csin( ctwopi * k*dt)
                    
                    for idim in range(geodim):
                            
                        for jdim in range(geodim):
                                
                            i =  0 + 2*(k + ncoeff*(jdim + geodim*il))

                            val = SpaceRotsCstr[il,ilcstr,idim,jdim]*c
                            
                            if (idim == jdim):
                                val -=1.
                            
                            if (cfabs(val) > eps_zero):
                            
                                cstr_row[nnz] = i
                                cstr_col[nnz] = icstr
                                cstr_data[nnz] = val
                            
                                nnz +=1
                                
                            i =  1 + 2*(k + ncoeff*(jdim + geodim*il))

                            val = SpaceRotsCstr[il,ilcstr,idim,jdim]*s
                            
                            if (cfabs(val) > eps_zero):
                                                
                                cstr_row[nnz] = i
                                cstr_col[nnz] = icstr
                                cstr_data[nnz] = val
                            
                                nnz +=1         
                                
                        icstr+=1
                            
                        for jdim in range(geodim):
                                
                            i =  1 + 2*(k + ncoeff*(jdim + geodim*il))

                            val = - SpaceRotsCstr[il,ilcstr,idim,jdim]*c
                            
                            if (idim == jdim):
                                val -=1.

                            if (cfabs(val) > eps_zero):
                                                    
                                cstr_row[nnz] = i
                                cstr_col[nnz] = icstr
                                cstr_data[nnz] = val
                            
                                nnz +=1
                                
                            i =  0 + 2*(k + ncoeff*(jdim + geodim*il))

                            val = SpaceRotsCstr[il,ilcstr,idim,jdim]*s
                            
                            if (cfabs(val) > eps_zero):
                                                    
                                cstr_row[nnz] = i
                                cstr_col[nnz] = icstr
                                cstr_data[nnz] = val 
                            
                                nnz +=1

                        icstr+=1
                                       
                else:
                    print(TimeRevsCstr[il,ilcstr])
                    raise ValueError("Invalid TimeRev")

    cdef long n_idx = nloop*geodim*ncoeff*2

    return scipy.sparse.coo_matrix((cstr_data,(cstr_row,cstr_col)),shape=(n_idx,icstr), dtype=np.float64)
    
@cython.cdivision(True)
def diagmat_changevar(
    long geodim,
    long ncoeff,
    long nparam,
    int [::1] param_to_coeff_csc_indptr,
    int [::1] param_to_coeff_csc_indices,
    double the_pow,
    double [::1] MassSum
):

    cdef np.ndarray[double, ndim=1, mode="c"]  diag_vect = np.zeros((nparam),dtype=np.float64)

    cdef double mass_sum
    cdef double k_avg, mass_avg, mul
    
    cdef long k_sum
    cdef long ift,idx,res,k,idim,il,iparam
    cdef long n_indptr,indptr_beg,indptr_end,i_shift

    for iparam in range(nparam):

        mass_sum = 0.
        k_sum = 0

        indptr_beg = param_to_coeff_csc_indptr[iparam]
        indptr_end = param_to_coeff_csc_indptr[iparam+1]
        n_indptr = indptr_end - indptr_beg

        for i_shift in range(n_indptr):
        
            idx = param_to_coeff_csc_indices[indptr_beg+i_shift]

            ift = idx%2
            res = idx/2
        
            k = res % ncoeff
            res = res / ncoeff
                    
            idim = res % geodim
            il = res / geodim

            if (k == 0):
                k = 1

            mass_sum += MassSum[il]
            k_sum += k

        k_sumd = k_sum
        k_avg = k_sumd / n_indptr
        mass_avg = mass_sum / n_indptr

        mul = k_avg * csqrt(mass_avg) *  ctwopisqrt2

        diag_vect[iparam] = cpow(mul,the_pow)

    cdef np.ndarray[long, ndim=1, mode="c"] diag_indices = np.array(range(nparam),dtype=np.int_)

    return scipy.sparse.coo_matrix((diag_vect,(diag_indices,diag_indices)),shape=(nparam,nparam), dtype=np.float64)
 
def Compute_square_dist(
    double[::1] x  ,
    double[::1] y  ,
    long s
):
        
    cdef double diff
    cdef double res = 0.
    cdef long i
    
    for i in range(s):
        
        diff = x[i]-y[i]
        res+=diff*diff
    
    return res
    
def Compute_Forces_Cython(
    double[:,::1] x ,
    double[::1] mass ,
    long nbody,
):
    # Does not actually computes the forces on every body, but rather the force divided by the mass.

    cdef long ib, ibp
    cdef long idim
    cdef long geodim = x.shape[1]
    cdef np.ndarray[double, ndim=2, mode="c"] f = np.zeros((nbody,geodim),dtype=np.float64)

    cdef double[::1] dx = np.zeros((geodim),dtype=np.float64)

    cdef double dx2,a
    cdef double b,bp

    for ib in range(nbody-1):
        for ibp in range(ib+1,nbody):

            for idim in range(geodim):
                dx[idim] = x[ib,idim]-x[ibp,idim]

            dx2 = dx[0]*dx[0]
            for idim in range(1,geodim):
                dx2 += dx[idim]*dx[idim]

            pot,potp,potpp = CCpt_interbody_pot(dx2)

            a = 2*potp

            b  = a*mass[ibp]
            bp = a*mass[ib ]

            for idim in range(geodim):

                f[ib,idim] -= b*dx[idim]
                f[ibp,idim] += bp*dx[idim]

    return f


def Compute_JacMat_Forces_Cython(
    double[:,::1] x ,
    double[::1] mass ,
    long nbody,
):
    # Does not actually computes the forces on every body, but rather the force divided by the mass.

    cdef long ib, ibp
    cdef long idim,jdim
    cdef long geodim = x.shape[1]
    cdef np.ndarray[double, ndim=4, mode="c"] Jf = np.zeros((nbody,geodim,nbody,geodim),dtype=np.float64)

    cdef double[::1] dx = np.zeros((geodim),dtype=np.float64)

    cdef double dx2
    cdef double a,aa,aap
    cdef double b,bb,bpp
    cdef double c

    for ib in range(nbody-1):
        for ibp in range(ib+1,nbody):

            for idim in range(geodim):
                dx[idim] = x[ib,idim]-x[ibp,idim]

            dx2 = dx[0]*dx[0]
            for idim in range(1,geodim):
                dx2 += dx[idim]*dx[idim]

            pot,potp,potpp = CCpt_interbody_pot(dx2)

            a = 2*potp
            aa  = a*mass[ibp]
            aap = a*mass[ib ]

            b = (4*potpp)
            bb  = b*mass[ibp]
            bpp = b*mass[ib ]

            for idim in range(geodim):

                Jf[ib ,idim,ib ,idim] -= aa
                Jf[ib ,idim,ibp,idim] += aa

                Jf[ibp,idim,ib ,idim] += aap
                Jf[ibp,idim,ibp,idim] -= aap

                for jdim in range(geodim):

                    dx2 = dx[idim]*dx[jdim]
                    c =  bb*dx2
                    Jf[ib ,idim,ib ,jdim] -= c
                    Jf[ib ,idim,ibp,jdim] += c

                    c = bpp*dx2
                    Jf[ibp,idim,ib ,jdim] += c
                    Jf[ibp,idim,ibp,jdim] -= c

    return Jf

def Compute_JacMul_Forces_Cython(
    double[:,::1] x     ,
    double[:,::1] x_d   ,
    double[::1] mass    ,
    long nbody          ,
):
    # Does not actually computes the forces on every body, but rather the force divided by the mass.

    cdef long ib, ibp
    cdef long idim,jdim
    cdef long geodim = x.shape[1]
    cdef np.ndarray[double, ndim=2, mode="c"] df = np.zeros((nbody,geodim),dtype=np.float64)

    cdef double[::1]  dx = np.zeros((geodim),dtype=np.float64)
    cdef double[::1]  ddx = np.zeros((geodim),dtype=np.float64)

    cdef double dx2,dxtddx
    cdef double a,aa,aap
    cdef double b,bb,bbp
    cdef double cc,ccp

    for ib in range(nbody-1):
        for ibp in range(ib+1,nbody):

            for idim in range(geodim):
                dx[idim] = x[ib,idim]-x[ibp,idim]
                ddx[idim] = x_d[ib,idim]-x_d[ibp,idim]

            dx2 = dx[0]*dx[0]
            dxtddx = dx[0]*ddx[0]
            for idim in range(1,geodim):
                dx2 += dx[idim]*dx[idim]
                dxtddx += dx[idim]*ddx[idim]

            pot,potp,potpp = CCpt_interbody_pot(dx2)

            a = 2*potp
            aa  = a*mass[ibp]
            aap = a*mass[ib ]

            b = (4*potpp*dxtddx)
            bb  = b*mass[ibp]
            bbp = b*mass[ib ]

            for idim in range(geodim):
                df[ib ,idim] -= bb *dx[idim] + aa *ddx[idim]
                df[ibp,idim] += bbp*dx[idim] + aap*ddx[idim]

    return df

def Transform_Coeffs_Single_Loop(
        double[:,::1] SpaceRot,
        double TimeRev, 
        double TimeShiftNum,
        double TimeShiftDen,
        double[:,:,::1] one_loop_coeffs,
        long ncoeff,
):
    # Transforms coeffs defining a single loop and returns updated coeffs
    
    cdef long geodim = one_loop_coeffs.shape[0]

    cdef long  k,i,j
    cdef double c,s,dt,dphi

    cdef double x,y

    cdef np.ndarray[double, ndim=3, mode="c"] all_coeffs_new_np = np.empty((geodim,ncoeff,2),dtype=np.float64)
    cdef double[:,:,::1] all_coeffs_new = all_coeffs_new_np

    cdef double[::1] v = np.empty((geodim),dtype=np.float64)
    cdef double[::1] w = np.empty((geodim),dtype=np.float64)

    for k in range(ncoeff):
        
        dt = TimeShiftNum / TimeShiftDen
        dphi = - ctwopi * k*dt

        c = ccos(dphi)
        s = csin(dphi)  

        for i in range(geodim):
            v[i] = one_loop_coeffs[i,k,0] * c - TimeRev * one_loop_coeffs[i,k,1] * s
            w[i] = one_loop_coeffs[i,k,0] * s + TimeRev * one_loop_coeffs[i,k,1] * c
        
        for i in range(geodim):
            x = SpaceRot[i,0] * v[0]
            y = SpaceRot[i,0] * w[0]
            for j in range(1,geodim): 
                x += SpaceRot[i,j] * v[j]
                y += SpaceRot[i,j] * w[j]

            all_coeffs_new[i,k,0] = x
            all_coeffs_new[i,k,1] = y
        
    return all_coeffs_new_np


def SparseScaleCoeffs(
    double[:,:,::1] one_loop_coeffs_in,
    long ncoeff_out,
    long ncoeff_in,
    long k_fac,
    double rfac,
):

    cdef long geodim = one_loop_coeffs_in.shape[0]

    cdef long idim
    cdef long k
    cdef long kmax = min(ncoeff_out//k_fac,ncoeff_in)

    cdef np.ndarray[double, ndim=3, mode="c"] all_coeffs_scale_np = np.zeros((geodim,ncoeff_out,2),dtype=np.float64)
    cdef double[:,:,::1] all_coeffs_scale = all_coeffs_scale_np

    for idim in range(geodim):
        for k in range(kmax):
            
            all_coeffs_scale[idim,k*k_fac,0]  = rfac * one_loop_coeffs_in[idim,k,0]
            all_coeffs_scale[idim,k*k_fac,1]  = rfac * one_loop_coeffs_in[idim,k,1]

    return all_coeffs_scale_np

def ComputeSpeedCoeffs(
    double[:,:,::1] one_loop_coeffs,
    long ncoeff,
):

    cdef long geodim = one_loop_coeffs.shape[0]

    cdef long idim
    cdef long k
    cdef double prod_fac

    cdef np.ndarray[double, ndim=3, mode="c"] one_loop_coeffs_speed_np = np.zeros((geodim,ncoeff,2),dtype=np.float64)
    cdef double[:,:,::1] one_loop_coeffs_speed = one_loop_coeffs_speed_np

    for idim in range(geodim):
        for k in range(ncoeff):

            prod_fac = ctwopi*k

            one_loop_coeffs_speed[idim,k,0] = - prod_fac * one_loop_coeffs[idim,k,1] 
            one_loop_coeffs_speed[idim,k,1] =   prod_fac * one_loop_coeffs[idim,k,0] 

    return one_loop_coeffs_speed_np


def Compute_hamil_hess_mul_Cython_nosym(
    long nbody                          ,
    long ncoeff                         ,
    long nint                           ,
    double[::1]       mass              ,
    double[:,:,::1]   all_pos           ,
    np.ndarray[double, ndim=5, mode="c"]  all_coeffs_d_xv  
):

    cdef long geodim = all_pos.shape[1]

    cdef Py_ssize_t il,ilp,i
    cdef Py_ssize_t idim,jdim
    cdef Py_ssize_t ibi
    cdef Py_ssize_t ib,ibp
    cdef Py_ssize_t iint
    cdef Py_ssize_t k
    cdef long k2
    cdef long ddiv,rem
    cdef double pot,potp,potpp
    cdef double prod_mass,a,b,c,dx2,prod_fac,dxtddx
    cdef double aa,aap,bb,bbp
    cdef double[::1] dx  = np.zeros((geodim),dtype=np.float64)
    cdef double[::1] ddx = np.zeros((geodim),dtype=np.float64)


    # 0 = -d/dt x + v
    # cdef np.ndarray[double, ndim=5, mode="c"] Hamil_hess_dxv_np = np.empty((2,nbody,geodim,ncoeff,2),dtype=np.float64)
    cdef np.ndarray[double, ndim=5, mode="c"] Hamil_hess_dxv_np = np.zeros((2,nbody,geodim,ncoeff,2),dtype=np.float64)
    cdef double[:,:,:,:,::1] Hamil_hess_dxv = Hamil_hess_dxv_np

    for ib in range(nbody):
        for idim in range(geodim):
            for k in range(ncoeff):

                prod_fac = ctwopi*k

                Hamil_hess_dxv[0,ib,idim,k,0] =   prod_fac * all_coeffs_d_xv[0,ib,idim,k,1] + all_coeffs_d_xv[1,ib,idim,k,0]
                Hamil_hess_dxv[0,ib,idim,k,1] = - prod_fac * all_coeffs_d_xv[0,ib,idim,k,0] + all_coeffs_d_xv[1,ib,idim,k,1]



    # 0 = -d/dt v + f/m

    c_coeffs_d_x = all_coeffs_d_xv[0,:,:,:,:].view(dtype=np.complex128)[...,0]
    cdef double[:,:,::1] all_pos_d_x = the_irfft(c_coeffs_d_x,norm="forward")

    cdef double[:,:,::1] hess_pot_all_d = np.zeros((nbody,geodim,nint),dtype=np.float64) # size ????

    for iint in range(nint):

        for ib in range(nbody):
            for ibp in range(ib+1,nbody):

                for idim in range(geodim):
                    dx[idim] = all_pos[ib,idim,iint] - all_pos[ibp,idim,iint] 
                    ddx[idim] = all_pos_d_x[ib,idim,iint] - all_pos_d_x[ibp,idim,iint] 

                dx2 = dx[0]*dx[0]
                dxtddx = dx[0]*ddx[0]
                for idim in range(1,geodim):
                    dx2 += dx[idim]*dx[idim]
                    dxtddx += dx[idim]*ddx[idim]

                pot,potp,potpp = CCpt_interbody_pot(dx2)

                a = 2*potp
                aa  = a*mass[ibp]
                aap = a*mass[ib ]

                b = (4*potpp*dxtddx)
                bb  = b*mass[ibp]
                bbp = b*mass[ib ]

                for idim in range(geodim):

                    hess_pot_all_d[ib ,idim,iint] += bb *dx[idim] + aa *ddx[idim]
                    hess_pot_all_d[ibp,idim,iint] -= bbp*dx[idim] + aap*ddx[idim]








                

    cdef double complex[:,:,::1]  hess_dx_pot_fft = the_rfft(hess_pot_all_d,norm="forward")


    for ib in range(nbody):

        for idim in range(geodim):
            
            Hamil_hess_dxv[1,ib,idim,0,0] = -hess_dx_pot_fft[ib,idim,0].real
            Hamil_hess_dxv[1,ib,idim,0,1] = 0 

            for k in range(1,ncoeff):
                
                prod_fac = ctwopi*k

                Hamil_hess_dxv[1,ib,idim,k,0] =   prod_fac * all_coeffs_d_xv[1,ib,idim,k,1] - hess_dx_pot_fft[ib,idim,k].real
                Hamil_hess_dxv[1,ib,idim,k,1] = - prod_fac * all_coeffs_d_xv[1,ib,idim,k,0] - hess_dx_pot_fft[ib,idim,k].imag





    return Hamil_hess_dxv_np



def Compute_hamil_hess_mul_Cython_nosym_split(
    long nbody                          ,
    long ncoeff                         ,
    long nint                           ,
    double[::1]       mass              ,
    double[:,:,::1]   all_pos           ,
    np.ndarray[double, ndim=4, mode="c"]  all_coeffs_d_x  ,
    np.ndarray[double, ndim=4, mode="c"]  all_coeffs_d_v  
):

    cdef long geodim = all_pos.shape[1]

    cdef Py_ssize_t il,ilp,i
    cdef Py_ssize_t idim,jdim
    cdef Py_ssize_t ibi
    cdef Py_ssize_t ib,ibp
    cdef Py_ssize_t iint
    cdef Py_ssize_t k
    cdef long k2
    cdef long ddiv,rem
    cdef double pot,potp,potpp
    cdef double prod_mass,a,b,c,dx2,prod_fac,dxtddx
    cdef double aa,aap,bb,bbp
    cdef double[::1] dx  = np.zeros((geodim),dtype=np.float64)
    cdef double[::1] ddx = np.zeros((geodim),dtype=np.float64)


    # 0 = -d/dt x + v
    cdef np.ndarray[double, ndim=4, mode="c"] Hamil_hess_dx_np = np.zeros((nbody,geodim,ncoeff,2),dtype=np.float64)
    cdef double[:,:,:,::1] Hamil_hess_dx = Hamil_hess_dx_np

    for ib in range(nbody):
        for idim in range(geodim):
            for k in range(ncoeff):

                prod_fac = ctwopi*k

                Hamil_hess_dx[ib,idim,k,0] =   prod_fac * all_coeffs_d_x[ib,idim,k,1] + all_coeffs_d_v[ib,idim,k,0]
                Hamil_hess_dx[ib,idim,k,1] = - prod_fac * all_coeffs_d_x[ib,idim,k,0] + all_coeffs_d_v[ib,idim,k,1]



    # 0 = -d/dt v + f/m

    c_coeffs_d_x = all_coeffs_d_x[:,:,:,:].view(dtype=np.complex128)[...,0]
    cdef double[:,:,::1] all_pos_d_x = the_irfft(c_coeffs_d_x,norm="forward")

    cdef double[:,:,::1] hess_pot_all_d = np.zeros((nbody,geodim,nint),dtype=np.float64) # size ????

    for iint in range(nint):

        for ib in range(nbody):
            for ibp in range(ib+1,nbody):

                for idim in range(geodim):
                    dx[idim] = all_pos[ib,idim,iint] - all_pos[ibp,idim,iint] 
                    ddx[idim] = all_pos_d_x[ib,idim,iint] - all_pos_d_x[ibp,idim,iint] 

                dx2 = dx[0]*dx[0]
                dxtddx = dx[0]*ddx[0]
                for idim in range(1,geodim):
                    dx2 += dx[idim]*dx[idim]
                    dxtddx += dx[idim]*ddx[idim]

                pot,potp,potpp = CCpt_interbody_pot(dx2)

                a = 2*potp
                aa  = a*mass[ibp]
                aap = a*mass[ib ]

                b = (4*potpp*dxtddx)
                bb  = b*mass[ibp]
                bbp = b*mass[ib ]

                for idim in range(geodim):

                    hess_pot_all_d[ib ,idim,iint] += bb *dx[idim] + aa *ddx[idim]
                    hess_pot_all_d[ibp,idim,iint] -= bbp*dx[idim] + aap*ddx[idim]

    cdef double complex[:,:,::1]  hess_dx_pot_fft = the_rfft(hess_pot_all_d,norm="forward")

    cdef np.ndarray[double, ndim=4, mode="c"] Hamil_hess_dv_np = np.zeros((nbody,geodim,ncoeff,2),dtype=np.float64)
    cdef double[:,:,:,::1] Hamil_hess_dv = Hamil_hess_dv_np

    for ib in range(nbody):

        for idim in range(geodim):
            
            Hamil_hess_dv[ib,idim,0,0] = -hess_dx_pot_fft[ib,idim,0].real
            Hamil_hess_dv[ib,idim,0,1] = 0 

            for k in range(1,ncoeff):
                
                prod_fac = ctwopi*k

                Hamil_hess_dv[ib,idim,k,0] =   prod_fac * all_coeffs_d_v[ib,idim,k,1] - hess_dx_pot_fft[ib,idim,k].real
                Hamil_hess_dv[ib,idim,k,1] = - prod_fac * all_coeffs_d_v[ib,idim,k,0] - hess_dx_pot_fft[ib,idim,k].imag



    return Hamil_hess_dx_np, Hamil_hess_dv_np






def Compute_Derivative_Cython_nosym(
    long nbody                          ,
    long ncoeff                         ,
    double[:,:,:,::1]  all_coeffs
):

    cdef long geodim = all_coeffs.shape[1]

    cdef Py_ssize_t idim,jdim
    cdef Py_ssize_t ib,ibp
    cdef Py_ssize_t iint
    cdef Py_ssize_t k
    cdef double prod_fac

    cdef np.ndarray[double, ndim=4, mode="c"] all_coeffs_d_np = np.zeros((nbody,geodim,ncoeff,2),dtype=np.float64)
    cdef double[:,:,:,::1] all_coeffs_d = all_coeffs_d_np

    for ib in range(nbody):
        for idim in range(geodim):
            for k in range(ncoeff):

                prod_fac = ctwopi*k

                all_coeffs_d[ib,idim,k,0] =   prod_fac * all_coeffs[ib,idim,k,1]
                all_coeffs_d[ib,idim,k,1] = - prod_fac * all_coeffs[ib,idim,k,0]

    return all_coeffs_d_np



def Compute_Derivative_precond_Cython_nosym(
    long nbody                          ,
    long ncoeff                         ,
    double[:,:,:,::1]  all_coeffs
):

    cdef long geodim = all_coeffs.shape[1]

    cdef Py_ssize_t idim,jdim
    cdef Py_ssize_t ib,ibp
    cdef Py_ssize_t iint
    cdef Py_ssize_t k
    cdef double prod_fac

    cdef np.ndarray[double, ndim=4, mode="c"] all_coeffs_d_np = np.zeros((nbody,geodim,ncoeff,2),dtype=np.float64)
    cdef double[:,:,:,::1] all_coeffs_d = all_coeffs_d_np

    for ib in range(nbody):
        for idim in range(geodim):

            all_coeffs_d[ib,idim,0,0] = ctwopi * all_coeffs[ib,idim,0,0]
            all_coeffs_d[ib,idim,0,1] = ctwopi * all_coeffs[ib,idim,0,1]

            for k in range(1,ncoeff):

                prod_fac = ctwopi*k

                all_coeffs_d[ib,idim,k,0] = prod_fac * all_coeffs[ib,idim,k,0]
                all_coeffs_d[ib,idim,k,1] = prod_fac * all_coeffs[ib,idim,k,1]
    
    return all_coeffs_d_np

def Compute_Derivative_precond_inv_Cython_nosym(
    long nbody                          ,
    long ncoeff                         ,
    double[:,:,:,::1]  all_coeffs
):

    cdef long geodim = all_coeffs.shape[1]

    cdef Py_ssize_t idim,jdim
    cdef Py_ssize_t ib,ibp
    cdef Py_ssize_t iint
    cdef Py_ssize_t k
    cdef double prod_fac

    cdef np.ndarray[double, ndim=4, mode="c"] all_coeffs_d_np = np.zeros((nbody,geodim,ncoeff,2),dtype=np.float64)
    cdef double[:,:,:,::1] all_coeffs_d = all_coeffs_d_np

    for ib in range(nbody):
        for idim in range(geodim):

            prod_fac = 1. / ctwopi

            all_coeffs_d[ib,idim,0,0] = prod_fac * all_coeffs[ib,idim,0,0]
            all_coeffs_d[ib,idim,0,1] = prod_fac * all_coeffs[ib,idim,0,1]

            for k in range(1,ncoeff):

                prod_fac = 1. / (ctwopi*k)

                all_coeffs_d[ib,idim,k,0] = prod_fac * all_coeffs[ib,idim,k,0]
                all_coeffs_d[ib,idim,k,1] = prod_fac * all_coeffs[ib,idim,k,1]
    
    return all_coeffs_d_np




    


def Compute_hamil_hess_mul_xonly_Cython_nosym(
    long nbody                          ,
    long ncoeff                         ,
    long nint                           ,
    double[::1]       mass              ,
    double[:,:,::1]   all_pos           ,
    np.ndarray[double, ndim=4, mode="c"]  all_coeffs_d_x 
):

    cdef long geodim = all_pos.shape[1]

    cdef Py_ssize_t il,ilp,i
    cdef Py_ssize_t idim,jdim
    cdef Py_ssize_t ibi
    cdef Py_ssize_t ib,ibp
    cdef Py_ssize_t iint
    cdef Py_ssize_t k
    cdef long k2
    cdef long ddiv,rem
    cdef double pot,potp,potpp
    cdef double prod_mass,a,b,c,dx2,prod_fac,dxtddx
    cdef double aa,aap,bb,bbp
    cdef double[::1] dx  = np.zeros((geodim),dtype=np.float64)
    cdef double[::1] ddx = np.zeros((geodim),dtype=np.float64)



    cdef np.ndarray[double, ndim=4, mode="c"] Hamil_hess_dx_np = np.empty((nbody,geodim,ncoeff,2),dtype=np.float64)
    cdef double[:,:,:,::1] Hamil_hess_dx = Hamil_hess_dx_np



    # 0 = -d2/dt2 x + f/m

    c_coeffs_d_x = all_coeffs_d_x[:,:,:,:].view(dtype=np.complex128)[...,0]
    cdef double[:,:,::1] all_pos_d_x = the_irfft(c_coeffs_d_x,norm="forward")

    cdef double[:,:,::1] hess_pot_all_d = np.zeros((nbody,geodim,nint),dtype=np.float64) # size ????

    for iint in range(nint):

        for ib in range(nbody):
            for ibp in range(ib+1,nbody):

                for idim in range(geodim):
                    dx[idim] = all_pos[ib,idim,iint] - all_pos[ibp,idim,iint] 
                    ddx[idim] = all_pos_d_x[ib,idim,iint] - all_pos_d_x[ibp,idim,iint] 

                dx2 = dx[0]*dx[0]
                dxtddx = dx[0]*ddx[0]
                for idim in range(1,geodim):
                    dx2 += dx[idim]*dx[idim]
                    dxtddx += dx[idim]*ddx[idim]

                pot,potp,potpp = CCpt_interbody_pot(dx2)

                a = 2*potp
                aa  = a*mass[ibp]
                aap = a*mass[ib ]

                b = (4*potpp*dxtddx)
                bb  = b*mass[ibp]
                bbp = b*mass[ib ]

                for idim in range(geodim):

                    hess_pot_all_d[ib ,idim,iint] += bb *dx[idim] + aa *ddx[idim]
                    hess_pot_all_d[ibp,idim,iint] -= bbp*dx[idim] + aap*ddx[idim]




    cdef double complex[:,:,::1]  hess_dx_pot_fft = the_rfft(hess_pot_all_d,norm="forward")


    for ib in range(nbody):

        for idim in range(geodim):
            
            Hamil_hess_dx[ib,idim,0,0] = -hess_dx_pot_fft[ib,idim,0].real
            Hamil_hess_dx[ib,idim,0,1] = 0 

            for k in range(1,ncoeff):
                 
                k2 = k*k
                a = cfourpisq*k2
                
                Hamil_hess_dx[ib,idim,k,0] = a * all_coeffs_d_x[ib,idim,k,0] - hess_dx_pot_fft[ib,idim,k].real
                Hamil_hess_dx[ib,idim,k,1] = a * all_coeffs_d_x[ib,idim,k,1] - hess_dx_pot_fft[ib,idim,k].imag





    return Hamil_hess_dx_np


def InplaceSmoothCoeffs(
    long nloop                          ,
    long ncoeff                         ,
    long ncoeff_smooth_init             ,
    double smooth_mul                   ,
    double[:,:,:,::1]  all_coeffs 
):

    cdef Py_ssize_t il,k,idim
    cdef double prod_mul
    cdef long geodim = all_coeffs.shape[1]

    for il in range(nloop):
        for idim in range(geodim):
            prod_mul = 1.
            for k in range(ncoeff_smooth_init,ncoeff):

                prod_mul *= smooth_mul

                all_coeffs[il,idim,k,0] *= prod_mul
                all_coeffs[il,idim,k,1] *= prod_mul




