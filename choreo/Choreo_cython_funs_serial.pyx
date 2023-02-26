'''
Choreo_cython_funs_2D.pyx : Defines useful compiled functions in the choreo project.

The functions in this file are (as much as possible) written is Cython.
They will be cythonized (i.e. processed by Cython into a C code, which will be compiled ) in setup.py.

Hence, in this file, performance is favored against readability or ease of use.

Functions in this files are specialized to 2 space dimensions with manual loop unrolling


'''

import os
import numpy as np
cimport numpy as np
np.import_array()

cimport cython

from libc.math cimport pow as cpow
from libc.math cimport fabs as cfabs
from libc.math cimport cos as ccos
from libc.math cimport sin as csin
from libc.math cimport sqrt as csqrt
from libc.math cimport isnan as cisnan
from libc.math cimport isinf as cisinf


cdef double cn = -0.5  #coeff of x^2 in the potential power law
cdef double cnm1 = cn-1
cdef double cnm2 = cn-2

cdef double ctwopi = 2* np.pi
cdef double cfourpi = 4 * np.pi
cdef double cfourpisq = ctwopi*ctwopi

cdef double cnnm1 = cn*(cn-1)

cdef double cmn = -cn
cdef double cmnnm1 = -cnnm1

# xsq is the square of the distance between two bodies !
@cython.profile(False)
@cython.linetrace(False)
cdef inline (double, double, double) CCpt_interbody_pot(double xsq) nogil:
    # Cython definition of the potential law
    
    cdef double a = cpow(xsq,cnm2)
    cdef double b = xsq*a
    
    cdef double pot = -xsq*b
    cdef double potp = cmn*b
    cdef double potpp = cmnnm1*a
    
    return pot,potp,potpp

@cython.cdivision(True)
# cdef (double, double[:,:,::1]) Compute_action_Cython_time_loop_2D(
def Compute_action_Cython_time_loop_2D(
    long              nloop             ,
    long              nint              ,
    double[::1]       mass              ,
    long[::1]         loopnb            ,
    long[:,::1]       Targets           ,
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
    double[:,:,::1]   all_pos           ,
):

    cdef Py_ssize_t il,ilp
    cdef Py_ssize_t ibi
    cdef Py_ssize_t ib,ibp
    cdef Py_ssize_t iint
    cdef Py_ssize_t shift_i, shift_ip
    cdef double pot,potp,potpp
    cdef double dx0,dx1
    cdef double ddx0,ddx1
    cdef double prod_mass,a,b,dx2,prod_fac
    cdef double Pot_en = 0.

    cdef np.ndarray[double, ndim=3, mode="c"] grad_pot_all_local_np = np.zeros((nloop,2,nint),dtype=np.float64)
    cdef double[:,:,::1] grad_pot_all_local = grad_pot_all_local_np

    for iint in range(nint):

        # Different loops
        for il in range(nloop):
            for ilp in range(il+1,nloop):

                for ib in range(loopnb[il]):
                    for ibp in range(loopnb[ilp]):

                        shift_i  = (((((iint - ((nint*TimeShiftNumUn[il ,ib ]) // TimeShiftDenUn[il ,ib ])) * TimeRevsUn[il ,ib ]) % nint) + nint) % nint)
                        shift_ip = (((((iint - ((nint*TimeShiftNumUn[ilp,ibp]) // TimeShiftDenUn[ilp,ibp])) * TimeRevsUn[ilp,ibp]) % nint) + nint) % nint)
                        
                        prod_mass = mass[Targets[il,ib]]*mass[Targets[ilp,ibp]]

                        dx0  = ( SpaceRotsUn[il ,ib ,0,0]*all_pos[il ,0,shift_i ]
                               - SpaceRotsUn[ilp,ibp,0,0]*all_pos[ilp,0,shift_ip]
                               + SpaceRotsUn[il ,ib ,0,1]*all_pos[il ,1,shift_i ]
                               - SpaceRotsUn[ilp,ibp,0,1]*all_pos[ilp,1,shift_ip] )

                        dx1  =  ( SpaceRotsUn[il ,ib ,1,0]*all_pos[il ,0,shift_i ]
                                - SpaceRotsUn[ilp,ibp,1,0]*all_pos[ilp,0,shift_ip]
                                + SpaceRotsUn[il ,ib ,1,1]*all_pos[il ,1,shift_i ]
                                - SpaceRotsUn[ilp,ibp,1,1]*all_pos[ilp,1,shift_ip] )

                        dx2 = dx0*dx0+dx1*dx1
                            
                        pot,potp,potpp = CCpt_interbody_pot(dx2)
                        
                        Pot_en += pot*prod_mass

                        a = (2*prod_mass*potp)

                        dx0 = a * dx0
                        dx1 = a * dx1

                        grad_pot_all_local[il ,0,shift_i ] += SpaceRotsUn[il ,ib ,0,0] * dx0 + SpaceRotsUn[il ,ib ,1,0] * dx1
                        grad_pot_all_local[ilp,0,shift_ip] -= SpaceRotsUn[ilp,ibp,0,0] * dx0 + SpaceRotsUn[ilp,ibp,1,0] * dx1
  
                        grad_pot_all_local[il ,1,shift_i ] += SpaceRotsUn[il ,ib ,0,1] * dx0 + SpaceRotsUn[il ,ib ,1,1] * dx1
                        grad_pot_all_local[ilp,1,shift_ip] -= SpaceRotsUn[ilp,ibp,0,1] * dx0 + SpaceRotsUn[ilp,ibp,1,1] * dx1

        # Same loop + symmetry
        for il in range(nloop):

            for ibi in range(loopnbi[il]):

                shift_i = (((((iint - ((nint*TimeShiftNumBin[il ,ibi]) // TimeShiftDenBin[il ,ibi])) * TimeRevsBin[il ,ibi]) % nint) + nint) % nint)

                dx0  = ( SpaceRotsBin[il,ibi,0,0] * all_pos[il,0,shift_i]
                       + SpaceRotsBin[il,ibi,0,1] * all_pos[il,1,shift_i]
                       - all_pos[il,0,iint] )

                dx1  = ( SpaceRotsBin[il,ibi,1,0] * all_pos[il,0,shift_i]
                       + SpaceRotsBin[il,ibi,1,1] * all_pos[il,1,shift_i]
                       - all_pos[il,1,iint] )

                dx2 = dx0*dx0 + dx1*dx1

                pot,potp,potpp = CCpt_interbody_pot(dx2)
                
                Pot_en += pot*ProdMassSumAll[il,ibi]
                
                a = (2*ProdMassSumAll[il,ibi]*potp)

                dx0 = dx0*a
                dx1 = dx1*a

                b = SpaceRotsBin[il,ibi,0,0]*dx0 + SpaceRotsBin[il,ibi,1,0]*dx1

                grad_pot_all_local[il ,0,shift_i] += b
                grad_pot_all_local[il ,0,iint   ] -= dx0

                b = SpaceRotsBin[il,ibi,0,1]*dx0 + SpaceRotsBin[il,ibi,1,1]*dx1

                grad_pot_all_local[il ,1,shift_i] += b
                grad_pot_all_local[il ,1,iint   ] -= dx1

    Pot_en = Pot_en / nint

    return Pot_en, grad_pot_all_local_np


@cython.cdivision(True)
# cdef np.ndarray[double, ndim=3, mode="c"] Compute_action_hess_mul_Cython_time_loop_2D(
def Compute_action_hess_mul_Cython_time_loop_2D(
    long              nloop             ,
    long              nint              ,
    double[::1]       mass              ,
    long[::1]         loopnb            ,
    long[:,::1]       Targets           ,
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
    double[:,:,::1]   all_pos           ,
    double[:,:,::1]   all_pos_d         ,
):

    cdef Py_ssize_t il,ilp
    cdef Py_ssize_t idim,jdim
    cdef Py_ssize_t ibi
    cdef Py_ssize_t ib,ibp
    cdef Py_ssize_t iint
    cdef Py_ssize_t k
    cdef long k2
    cdef double pot,potp,potpp
    cdef double dx0,dx1
    cdef double ddx0,ddx1
    cdef double ddf0,ddf1
    cdef double prod_mass,a,b,dx2,prod_fac,dxtddx
    cdef Py_ssize_t shift_i,shift_ip

    cdef np.ndarray[double, ndim=3, mode="c"] hess_pot_all_d_local_np = np.zeros((nloop,2,nint),dtype=np.float64)
    cdef double[:,:,::1] hess_pot_all_d_local = hess_pot_all_d_local_np

    for iint in range(nint):

        # Different loops
        for il in range(nloop):
            for ilp in range(il+1,nloop):

                for ib in range(loopnb[il]):
                    for ibp in range(loopnb[ilp]):

                        shift_i  = (((((iint - ((nint*TimeShiftNumUn[il ,ib ]) // TimeShiftDenUn[il ,ib ])) * TimeRevsUn[il ,ib ]) % nint) + nint) % nint)
                        shift_ip = (((((iint - ((nint*TimeShiftNumUn[ilp,ibp]) // TimeShiftDenUn[ilp,ibp])) * TimeRevsUn[ilp,ibp]) % nint) + nint) % nint)

                        prod_mass = mass[Targets[il,ib]]*mass[Targets[ilp,ibp]]

                        dx0  =  ( SpaceRotsUn[il ,ib ,0,0]*all_pos[il ,0,shift_i ]
                                - SpaceRotsUn[ilp,ibp,0,0]*all_pos[ilp,0,shift_ip]
                                + SpaceRotsUn[il ,ib ,0,1]*all_pos[il ,1,shift_i ]
                                - SpaceRotsUn[ilp,ibp,0,1]*all_pos[ilp,1,shift_ip] )

                        dx1  =  ( SpaceRotsUn[il ,ib ,1,0]*all_pos[il ,0,shift_i ] 
                                - SpaceRotsUn[ilp,ibp,1,0]*all_pos[ilp,0,shift_ip]
                                + SpaceRotsUn[il ,ib ,1,1]*all_pos[il ,1,shift_i ]
                                - SpaceRotsUn[ilp,ibp,1,1]*all_pos[ilp,1,shift_ip] )
                        
                        ddx0  = ( SpaceRotsUn[il ,ib ,0,0]*all_pos_d[il ,0,shift_i ] 
                                - SpaceRotsUn[ilp,ibp,0,0]*all_pos_d[ilp,0,shift_ip]
                                + SpaceRotsUn[il ,ib ,0,1]*all_pos_d[il ,1,shift_i ]
                                - SpaceRotsUn[ilp,ibp,0,1]*all_pos_d[ilp,1,shift_ip] )

                        ddx1  = ( SpaceRotsUn[il ,ib ,1,0]*all_pos_d[il ,0,shift_i ] 
                                - SpaceRotsUn[ilp,ibp,1,0]*all_pos_d[ilp,0,shift_ip]
                                + SpaceRotsUn[il ,ib ,1,1]*all_pos_d[il ,1,shift_i ]
                                - SpaceRotsUn[ilp,ibp,1,1]*all_pos_d[ilp,1,shift_ip] )

                        dx2 = dx0*dx0 + dx1*dx1
                        dxtddx = dx0*ddx0 + dx1*ddx1
                            
                        pot,potp,potpp = CCpt_interbody_pot(dx2)

                        a = (2*prod_mass*potp)
                        b = (4*prod_mass*potpp*dxtddx)
                        
                        ddf0 = b*dx0+a*ddx0
                        ddf1 = b*dx1+a*ddx1
                            
                        hess_pot_all_d_local[il ,0,shift_i ] += SpaceRotsUn[il ,ib ,0,0] * ddf0 + SpaceRotsUn[il ,ib ,1,0] * ddf1
                        hess_pot_all_d_local[ilp,0,shift_ip] -= SpaceRotsUn[ilp,ibp,0,0] * ddf0 + SpaceRotsUn[ilp,ibp,1,0] * ddf1

                        hess_pot_all_d_local[il ,1,shift_i ] += SpaceRotsUn[il ,ib ,0,1] * ddf0 + SpaceRotsUn[il ,ib ,1,1] * ddf1
                        hess_pot_all_d_local[ilp,1,shift_ip] -= SpaceRotsUn[ilp,ibp,0,1] * ddf0 + SpaceRotsUn[ilp,ibp,1,1] * ddf1


        # Same loop + symmetry
        for il in range(nloop):

            for ibi in range(loopnbi[il]):

                shift_i  = (((((iint - ((nint*TimeShiftNumBin[il ,ibi]) // TimeShiftDenBin[il ,ibi])) * TimeRevsBin[il ,ibi]) % nint) + nint) % nint)

                dx0  =  ( SpaceRotsBin[il,ibi,0,0]*all_pos[il,0,shift_i]
                        + SpaceRotsBin[il,ibi,0,1]*all_pos[il,1,shift_i]
                        - all_pos[il,0,iint] )

                ddx0  = ( SpaceRotsBin[il,ibi,0,0]*all_pos_d[il,0,shift_i]
                        + SpaceRotsBin[il,ibi,0,1]*all_pos_d[il,1,shift_i]
                        - all_pos_d[il,0,iint] )

                dx1  = (  SpaceRotsBin[il,ibi,1,0]*all_pos[il,0,shift_i]
                        + SpaceRotsBin[il,ibi,1,1]*all_pos[il,1,shift_i]
                        - all_pos[il,1,iint] )

                ddx1  = ( SpaceRotsBin[il,ibi,1,0]*all_pos_d[il,0,shift_i]
                        + SpaceRotsBin[il,ibi,1,1]*all_pos_d[il,1,shift_i]
                        - all_pos_d[il,1,iint] )

                dx2 = dx0*dx0+dx1*dx1
                dxtddx = dx0*ddx0+dx1*ddx1

                pot,potp,potpp = CCpt_interbody_pot(dx2)
                
                a = (2*ProdMassSumAll[il,ibi]*potp)
                b = (4*ProdMassSumAll[il,ibi]*potpp*dxtddx)
        
                ddf0 = b*dx0+a*ddx0
                ddf1 = b*dx1+a*ddx1

                hess_pot_all_d_local[il ,0,shift_i] += SpaceRotsBin[il,ibi,0,0]*ddf0 + SpaceRotsBin[il,ibi,1,0]*ddf1
                hess_pot_all_d_local[il ,0,iint   ] -= ddf0

                hess_pot_all_d_local[il ,1,shift_i] += SpaceRotsBin[il,ibi,0,1]*ddf0 + SpaceRotsBin[il,ibi,1,1]*ddf1
                hess_pot_all_d_local[il ,1,iint   ] -= ddf1


    return hess_pot_all_d_local_np
