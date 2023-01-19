'''
Choreo_cython_funs_2D.pyx : Defines useful compiled functions in the Choreographies2 project.

The functions in this file are (as much as possible) written is Cython.
They will be cythonized (i.e. processed by Cython into a C code, which will be compiled ) in setup.py.

Hence, in this file, performance is favored against readability or ease of use.

Functions in this files are specilized to 2 space dimensions with manual loop unrolling


'''

import os
import numpy as np
cimport numpy as np
np.import_array()

cimport cython

import scipy.sparse as sp

from libc.math cimport pow as cpow
from libc.math cimport fabs as cfabs
from libc.math cimport cos as ccos
from libc.math cimport sin as csin
from libc.math cimport sqrt as csqrt
from libc.math cimport isnan as cisnan
from libc.math cimport isinf as cisinf


from choreo.Choreo_cython_funs import the_rfft,the_irfft


cdef double cn = -0.5  #coeff of x^2 in the potential power law
cdef double cnm1 = cn-1
cdef double cnm2 = cn-2

cdef double ctwopi = 2* np.pi
cdef double cfourpi = 4 * np.pi
cdef double cfourpisq = ctwopi*ctwopi

cdef double cnnm1 = cn*(cn-1)

cdef double cmn = -cn
cdef double cmnnm1 = -cnnm1

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


@cython.cdivision(True)
def Compute_action_Cython_2D(
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

    cdef Py_ssize_t il,ilp
    cdef Py_ssize_t idim
    cdef Py_ssize_t ibi
    cdef Py_ssize_t ib,ibp
    cdef Py_ssize_t iint
    cdef Py_ssize_t k
    cdef Py_ssize_t shift_i, shift_ip
    cdef long k2
    cdef long rem, ddiv
    cdef double pot,potp,potpp
    cdef double prod_mass,a,b,dx2,prod_fac

    cdef double[::1] dx = np.zeros((2),dtype=np.float64)
    cdef double[::1] df = np.zeros((2),dtype=np.float64)

    cdef long maxloopnb = 0
    cdef long maxloopnbi = 0

    for il in range(nloop):
        if (maxloopnb < loopnb[il]):
            maxloopnb = loopnb[il]
        if (maxloopnbi < loopnbi[il]):
            maxloopnbi = loopnbi[il]

    cdef double Pot_en = 0.

    cdef Py_ssize_t[:,::1] all_shiftsUn = np.zeros((nloop,maxloopnb),dtype=np.intp)
    cdef Py_ssize_t[:,::1] all_shiftsBin = np.zeros((nloop,maxloopnbi),dtype=np.intp)

    for il in range(nloop):
        for ib in range(loopnb[il]):

            k = (TimeRevsUn[il,ib]*nint*TimeShiftNumUn[il,ib])

            ddiv = - k // TimeShiftDenUn[il,ib]
            rem = k + ddiv * TimeShiftDenUn[il,ib]

            if (rem != 0):
                print("WARNING: remainder in integer division. Gradient computation will fail.")

            all_shiftsUn[il,ib] = (((ddiv) % nint) + nint) % nint
        
        for ibi in range(loopnbi[il]):

            k = (TimeRevsBin[il,ibi]*nint*TimeShiftNumBin[il,ibi])

            ddiv = - k // TimeShiftDenBin[il,ibi]
            rem = k + ddiv * TimeShiftDenBin[il,ibi]

            if (rem != 0):
                print("WARNING: remainder in integer division. Gradient computation will fail.")

            all_shiftsBin[il,ibi] = ((ddiv % nint) + nint) % nint
    
    cdef double[:,:,::1] grad_pot_all = np.zeros((nloop,2,nint),dtype=np.float64)

    for iint in range(nint):

        # Different loops
        for il in range(nloop):
            for ilp in range(il+1,nloop):

                for ib in range(loopnb[il]):
                    for ibp in range(loopnb[ilp]):
                        
                        prod_mass = mass[Targets[il,ib]]*mass[Targets[ilp,ibp]]

                        shift_i  = all_shiftsUn[il ,ib ]
                        shift_ip = all_shiftsUn[ilp,ibp]

                        dx[0]  = SpaceRotsUn[il ,ib ,0,0]*all_pos[il ,0,shift_i ]
                        dx[0] -= SpaceRotsUn[ilp,ibp,0,0]*all_pos[ilp,0,shift_ip]
                        dx[0] += SpaceRotsUn[il ,ib ,0,1]*all_pos[il ,1,shift_i ]
                        dx[0] -= SpaceRotsUn[ilp,ibp,0,1]*all_pos[ilp,1,shift_ip]

                        dx[1]  = SpaceRotsUn[il ,ib ,1,0]*all_pos[il ,0,shift_i ]
                        dx[1] -= SpaceRotsUn[ilp,ibp,1,0]*all_pos[ilp,0,shift_ip]
                        dx[1] += SpaceRotsUn[il ,ib ,1,1]*all_pos[il ,1,shift_i ]
                        dx[1] -= SpaceRotsUn[ilp,ibp,1,1]*all_pos[ilp,1,shift_ip]

                        dx2 = dx[0]*dx[0]+dx[1]*dx[1]
                            
                        pot,potp,potpp = CCpt_interbody_pot(dx2)
                        
                        Pot_en += pot*prod_mass

                        a = (2*prod_mass*potp)

                        dx[0] *= a
                        dx[1] *= a

                        grad_pot_all[il ,0,shift_i ] += SpaceRotsUn[il ,ib ,0,0]*dx[0] 
                        grad_pot_all[il ,0,shift_i ] += SpaceRotsUn[il ,ib ,1,0]*dx[1]
                        grad_pot_all[ilp,0,shift_ip] -= SpaceRotsUn[ilp,ibp,0,0]*dx[0]
                        grad_pot_all[ilp,0,shift_ip] -= SpaceRotsUn[ilp,ibp,1,0]*dx[1]

                        grad_pot_all[il ,1,shift_i ] += SpaceRotsUn[il ,ib ,0,1]*dx[0]
                        grad_pot_all[il ,1,shift_i ] += SpaceRotsUn[il ,ib ,1,1]*dx[1]
                        grad_pot_all[ilp,1,shift_ip] -= SpaceRotsUn[ilp,ibp,0,1]*dx[0]
                        grad_pot_all[ilp,1,shift_ip] -= SpaceRotsUn[ilp,ibp,1,1]*dx[1]

        # Same loop + symmetry
        for il in range(nloop):

            for ibi in range(loopnbi[il]):

                shift_i  = all_shiftsBin[il,ibi]

                # print(iint,ibi,shift_i)

                dx[0]  = SpaceRotsBin[il,ibi,0,0]*all_pos[il,0,shift_i]
                dx[0] += SpaceRotsBin[il,ibi,0,1]*all_pos[il,1,shift_i]
                dx[0] -= all_pos[il,0,iint]

                dx[1]  = SpaceRotsBin[il,ibi,1,0]*all_pos[il,0,shift_i]
                dx[1] += SpaceRotsBin[il,ibi,1,1]*all_pos[il,1,shift_i]
                dx[1] -= all_pos[il,1,iint]

                dx2 = dx[0]*dx[0] + dx[1]*dx[1] 

                pot,potp,potpp = CCpt_interbody_pot(dx2)
                
                Pot_en += pot*ProdMassSumAll[il,ibi]
                
                a = (2*ProdMassSumAll[il,ibi]*potp)

                dx[0] *= a
                dx[1] *= a

                b  = SpaceRotsBin[il,ibi,0,0]*dx[0]
                b += SpaceRotsBin[il,ibi,1,0]*dx[1]

                grad_pot_all[il ,0,shift_i] += b
                grad_pot_all[il ,0,iint   ] -= dx[0]

                b  = SpaceRotsBin[il,ibi,0,1]*dx[0]
                b += SpaceRotsBin[il,ibi,1,1]*dx[1]
                grad_pot_all[il ,1,shift_i] += b
                grad_pot_all[il ,1,iint   ] -= dx[1]

        # Increments time at the end
        for il in range(nloop):
            for ib in range(loopnb[il]):
                all_shiftsUn[il,ib] = ((((all_shiftsUn[il,ib]+TimeRevsUn[il,ib]) % nint) + nint) % nint)
                
            for ibi in range(loopnbi[il]):

                all_shiftsBin[il,ibi] = ((((all_shiftsBin[il,ibi]+TimeRevsBin[il,ibi]) % nint) + nint) % nint)


    Pot_en = Pot_en / nint
    cdef double complex[:,:,::1]  grad_pot_fft = the_rfft(grad_pot_all,norm="forward")  #
    cdef double Kin_en = 0 
    cdef np.ndarray[double, ndim=4, mode="c"] Action_grad_np = np.empty((nloop,2,ncoeff,2),np.float64)
    cdef double[:,:,:,::1] Action_grad = Action_grad_np #
    for il in range(nloop):
        
        prod_fac = MassSum[il]*cfourpisq
        
        for idim in range(2): 
            Action_grad[il,idim,0,0] = -grad_pot_fft[il,idim,0].real
            Action_grad[il,idim,0,1] = 0  
            for k in range(1,ncoeff):
                
                k2 = k*k
                a = prod_fac*k2
                b=2*a  
                Kin_en += a *((all_coeffs[il,idim,k,0]*all_coeffs[il,idim,k,0]) + (all_coeffs[il,idim,k,1]*all_coeffs[il,idim,k,1]))
                
                Action_grad[il,idim,k,0] = b*all_coeffs[il,idim,k,0] - 2*grad_pot_fft[il,idim,k].real
                Action_grad[il,idim,k,1] = b*all_coeffs[il,idim,k,1] - 2*grad_pot_fft[il,idim,k].imag
            
    Action = Kin_en-Pot_en
    
    return Action,Action_grad_np

@cython.cdivision(True)
def Compute_action_hess_mul_Cython_2D(
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

    cdef Py_ssize_t il,ilp
    cdef Py_ssize_t idim,jdim
    cdef Py_ssize_t ibi
    cdef Py_ssize_t ib,ibp
    cdef Py_ssize_t iint
    cdef Py_ssize_t k
    cdef long k2
    cdef double pot,potp,potpp
    cdef double prod_mass,a,b,c,dx2,prod_fac,dxtddx
    cdef double[::1] dx  = np.zeros((2),dtype=np.float64)
    cdef double[::1] ddx = np.zeros((2),dtype=np.float64)
    cdef double[::1] ddf = np.zeros((2),dtype=np.float64)
        
    cdef Py_ssize_t maxloopnb = 0
    cdef Py_ssize_t maxloopnbi = 0

    cdef Py_ssize_t shift_i,shift_ip

    for il in range(nloop):
        if (maxloopnb < loopnb[il]):
            maxloopnb = loopnb[il]
        if (maxloopnbi < loopnbi[il]):
            maxloopnbi = loopnbi[il]

    c_coeffs_d = all_coeffs_d.view(dtype=np.complex128)[...,0]
    cdef double[:,:,::1]  all_pos_d = the_irfft(c_coeffs_d,n=nint,axis=2,norm="forward")

    cdef Py_ssize_t[:,::1] all_shiftsUn = np.zeros((nloop,maxloopnb),dtype=np.intp)
    cdef Py_ssize_t[:,::1] all_shiftsBin = np.zeros((nloop,maxloopnbi),dtype=np.intp)
    
    for il in range(nloop):
        for ib in range(loopnb[il]):

            k = (TimeRevsUn[il,ib]*nint*TimeShiftNumUn[il,ib])

            ddiv = - k // TimeShiftDenUn[il,ib]
            rem = k + ddiv * TimeShiftDenUn[il,ib]

            if (rem != 0):
                print("WARNING: remainder in integer division. Gradient computation will fail.")

            all_shiftsUn[il,ib] = (((ddiv) % nint) + nint) % nint
        
        for ibi in range(loopnbi[il]):

            k = (TimeRevsBin[il,ibi]*nint*TimeShiftNumBin[il,ibi])

            ddiv = - k // TimeShiftDenBin[il,ibi]
            rem = k + ddiv * TimeShiftDenBin[il,ibi]

            if (rem != 0):
                print("WARNING: remainder in integer division. Gradient computation will fail.")

            all_shiftsBin[il,ibi] = (((ddiv) % nint) + nint) % nint
    
    cdef double[:,:,::1] hess_pot_all_d = np.zeros((nloop,2,nint),dtype=np.float64)

    for iint in range(nint):

        # Different loops
        for il in range(nloop):
            for ilp in range(il+1,nloop):

                for ib in range(loopnb[il]):
                    for ibp in range(loopnb[ilp]):
                        
                        prod_mass = mass[Targets[il,ib]]*mass[Targets[ilp,ibp]]

                        shift_i  = all_shiftsUn[il ,ib ]
                        shift_ip = all_shiftsUn[ilp,ibp]

                        dx[0]  = SpaceRotsUn[il ,ib ,0,0]*all_pos[il ,0,shift_i ] 
                        dx[0] -= SpaceRotsUn[ilp,ibp,0,0]*all_pos[ilp,0,shift_ip]
                        dx[0] += SpaceRotsUn[il ,ib ,0,1]*all_pos[il ,1,shift_i ]
                        dx[0] -= SpaceRotsUn[ilp,ibp,0,1]*all_pos[ilp,1,shift_ip]

                        dx[1]  = SpaceRotsUn[il ,ib ,1,0]*all_pos[il ,0,shift_i ] 
                        dx[1] -= SpaceRotsUn[ilp,ibp,1,0]*all_pos[ilp,0,shift_ip]
                        dx[1] += SpaceRotsUn[il ,ib ,1,1]*all_pos[il ,1,shift_i ]
                        dx[1] -= SpaceRotsUn[ilp,ibp,1,1]*all_pos[ilp,1,shift_ip]
                        
                        ddx[0]  = SpaceRotsUn[il ,ib ,0,0]*all_pos_d[il ,0,shift_i ] 
                        ddx[0] -= SpaceRotsUn[ilp,ibp,0,0]*all_pos_d[ilp,0,shift_ip]
                        ddx[0] += SpaceRotsUn[il ,ib ,0,1]*all_pos_d[il ,1,shift_i ]
                        ddx[0] -= SpaceRotsUn[ilp,ibp,0,1]*all_pos_d[ilp,1,shift_ip]

                        ddx[1]  = SpaceRotsUn[il ,ib ,1,0]*all_pos_d[il ,0,shift_i ] 
                        ddx[1] -= SpaceRotsUn[ilp,ibp,1,0]*all_pos_d[ilp,0,shift_ip]
                        ddx[1] += SpaceRotsUn[il ,ib ,1,1]*all_pos_d[il ,1,shift_i ]
                        ddx[1] -= SpaceRotsUn[ilp,ibp,1,1]*all_pos_d[ilp,1,shift_ip]

                        dx2 = dx[0]*dx[0] + dx[1]*dx[1]
                        dxtddx = dx[0]*ddx[0] + dx[1]*ddx[1]
                            
                        pot,potp,potpp = CCpt_interbody_pot(dx2)

                        a = (2*prod_mass*potp)
                        b = (4*prod_mass*potpp*dxtddx)
                        
                        ddf[0] = b*dx[0]+a*ddx[0]
                        ddf[1] = b*dx[1]+a*ddx[1]
                            
                        hess_pot_all_d[il ,0,shift_i] += SpaceRotsUn[il,ib,0,0]*ddf[0]
                        hess_pot_all_d[il ,0,shift_i] += SpaceRotsUn[il,ib,1,0]*ddf[1]

                        hess_pot_all_d[ilp,0,shift_ip] -= SpaceRotsUn[ilp,ibp,0,0]*ddf[0]
                        hess_pot_all_d[ilp,0,shift_ip] -= SpaceRotsUn[ilp,ibp,1,0]*ddf[1]

                        hess_pot_all_d[il ,1,shift_i] += SpaceRotsUn[il,ib,0,1]*ddf[0]
                        hess_pot_all_d[il ,1,shift_i] += SpaceRotsUn[il,ib,1,1]*ddf[1]

                        hess_pot_all_d[ilp,1,shift_ip] -= SpaceRotsUn[ilp,ibp,0,1]*ddf[0]
                        hess_pot_all_d[ilp,1,shift_ip] -= SpaceRotsUn[ilp,ibp,1,1]*ddf[1]


        # Same loop + symmetry
        for il in range(nloop):

            for ibi in range(loopnbi[il]):

                shift_i  = all_shiftsBin[il,ibi]

                dx[0]  = SpaceRotsBin[il,ibi,0,0]*all_pos[il,0,shift_i]
                dx[0] += SpaceRotsBin[il,ibi,0,1]*all_pos[il,1,shift_i]
                dx[0] -= all_pos[il,0,iint]

                ddx[0]  = SpaceRotsBin[il,ibi,0,0]*all_pos_d[il,0,shift_i]
                ddx[0] += SpaceRotsBin[il,ibi,0,1]*all_pos_d[il,1,shift_i]
                ddx[0] -= all_pos_d[il,0,iint]

                dx[1]  = SpaceRotsBin[il,ibi,1,0]*all_pos[il,0,shift_i]
                dx[1] += SpaceRotsBin[il,ibi,1,1]*all_pos[il,1,shift_i]
                dx[1] -= all_pos[il,1,iint]

                ddx[1]  = SpaceRotsBin[il,ibi,1,0]*all_pos_d[il,0,shift_i]
                ddx[1] += SpaceRotsBin[il,ibi,1,1]*all_pos_d[il,1,shift_i]
                ddx[1] -= all_pos_d[il,1,iint]

                dx2 = dx[0]*dx[0]+dx[1]*dx[1]
                dxtddx = dx[0]*ddx[0]+dx[1]*ddx[1]

                pot,potp,potpp = CCpt_interbody_pot(dx2)
                
                a = (2*ProdMassSumAll[il,ibi]*potp)
                b = (4*ProdMassSumAll[il,ibi]*potpp*dxtddx)
        
                ddf[0] = b*dx[0]+a*ddx[0]
                ddf[1] = b*dx[1]+a*ddx[1]

                hess_pot_all_d[il ,0,shift_i] += SpaceRotsBin[il,ibi,0,0]*ddf[0]
                hess_pot_all_d[il ,0,shift_i] += SpaceRotsBin[il,ibi,1,0]*ddf[1]
                hess_pot_all_d[il ,0,iint   ] -= ddf[0]

                hess_pot_all_d[il ,1,shift_i] += SpaceRotsBin[il,ibi,0,1]*ddf[0]
                hess_pot_all_d[il ,1,shift_i] += SpaceRotsBin[il,ibi,1,1]*ddf[1]
                hess_pot_all_d[il ,1,iint   ] -= ddf[1]

        # Increments time at the end
        for il in range(nloop):
            for ib in range(loopnb[il]):
                all_shiftsUn[il,ib] = ((((all_shiftsUn[il,ib]+TimeRevsUn[il,ib]) % nint) + nint) % nint)
                
            for ibi in range(loopnbi[il]):
                all_shiftsBin[il,ibi] = ((((all_shiftsBin[il,ibi]+TimeRevsBin[il,ibi]) % nint) + nint) % nint)

    cdef double complex[:,:,::1]  hess_dx_pot_fft = the_rfft(hess_pot_all_d,norm="forward")

    cdef np.ndarray[double, ndim=4, mode="c"] Action_hess_dx_np = np.empty((nloop,2,ncoeff,2),np.float64)
    cdef double[:,:,:,::1] Action_hess_dx = Action_hess_dx_np

    for il in range(nloop):
        
        prod_fac = MassSum[il]*cfourpisq
        
        for idim in range(2):
            
            Action_hess_dx[il,idim,0,0] = -hess_dx_pot_fft[il,idim,0].real
            Action_hess_dx[il,idim,0,1] = 0 

            for k in range(1,ncoeff):
                
                k2 = k*k
                a = 2*prod_fac*k2
                
                Action_hess_dx[il,idim,k,0] = a*all_coeffs_d[il,idim,k,0] - 2*hess_dx_pot_fft[il,idim,k].real
                Action_hess_dx[il,idim,k,1] = a*all_coeffs_d[il,idim,k,1] - 2*hess_dx_pot_fft[il,idim,k].imag


    return Action_hess_dx_np
    
@cython.cdivision(True)
def RotateFastWithSlow_2D(
    double[:,::1] all_pos_slow,
    double[:,::1] all_pos_slow_speed,
    double[:,::1] all_pos_fast,
    long nint
):

    cdef long iint

    cdef double vx,vy
    cdef double vnorminv, vnormsq

    cdef np.ndarray[double, ndim=2, mode="c"] all_pos_avg_np = np.zeros((2,nint),dtype=np.float64)
    cdef double[:,::1] all_pos_avg = all_pos_avg_np

    for iint in range(nint):
        
        vnormsq = all_pos_slow_speed[0,iint]**2 + all_pos_slow_speed[1,iint]**2
        vnorminv = 1./csqrt(vnormsq)

        vx = all_pos_slow_speed[0,iint] * vnorminv
        vy = all_pos_slow_speed[1,iint] * vnorminv

        all_pos_avg[0,iint] = all_pos_slow[0,iint] + vx * all_pos_fast[0,iint] - vy * all_pos_fast[1,iint] 
        all_pos_avg[1,iint] = all_pos_slow[1,iint] + vy * all_pos_fast[0,iint] + vx * all_pos_fast[1,iint] 

    return all_pos_avg_np


