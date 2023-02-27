import os
import numpy as np
import numba
import multiprocessing

max_num_threads = multiprocessing.cpu_count()

from choreo.Choreo_cython_funs import the_rfft,the_irfft

numba_kwargs = {
    'nopython':True     ,
    'cache':True        ,
    'fastmath':True     ,
    'nogil':True        ,
}


cn = -0.5  #coeff of x^2 in the potential power law
cnm1 = cn-1
cnm2 = cn-2

ctwopi = 2* np.pi
cfourpi = 4 * np.pi
cfourpisq = ctwopi*ctwopi

cnnm1 = cn*(cn-1)
cmn = -cn
cmnnm1 = -cnnm1

@numba.jit(inline='always',**numba_kwargs)
def Cpt_interbody_pot(xsq):
    
    a = xsq ** cnm2
    b = xsq*a
    
    pot = -xsq*b
    potp = cmn*b
    potpp = cmnnm1*a
    
    return pot,potp,potpp

def Empty_Backend_action(
    nloop             ,
    ncoeff            ,
    nint              ,
    mass              ,
    loopnb            ,
    Targets           ,
    MassSum           ,
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
    all_coeffs        ,
    all_pos 
):
    
    grad_pot_fft = the_rfft(all_pos,norm="forward")
    
    return 0.,all_coeffs

def Empty_Backend_hess_mul(
    nloop             ,
    ncoeff            ,
    nint              ,
    mass              ,
    loopnb            ,
    Targets           ,
    MassSum           ,
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
    all_coeffs        ,
    all_coeffs_d      , 
    all_pos 
):
    
    c_coeffs_d = all_coeffs_d.view(dtype=np.complex128)[...,0]
    all_pos_d = the_irfft(c_coeffs_d,n=nint,axis=2,norm="forward")

    hess_dx_pot_fft = the_rfft(all_pos,norm="forward")
    
    return all_coeffs_d

def Compute_action_Numba_nD_serial(
    nloop             ,
    ncoeff            ,
    nint              ,
    mass              ,
    loopnb            ,
    Targets           ,
    MassSum           ,
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
    all_coeffs        ,
    all_pos 
):
    # This function is probably the most important one.
    # Computes the action and its gradient with respect to the Fourier coefficients of the generator in each loop.
    
    Pot_en, grad_pot_all = Compute_action_Numba_time_loop_nD_serial(
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

    grad_pot_fft = the_rfft(grad_pot_all,norm="forward") 

    Kin_en, Action_grad = Compute_action_Numba_Kin_loop_nD_serial(
        nloop             ,
        ncoeff            ,
        MassSum           ,
        all_coeffs        ,
        grad_pot_fft 
    )

    Action = Kin_en-Pot_en/nint
    
    return Action,Action_grad

@numba.jit(**numba_kwargs)
def Compute_action_Numba_Kin_loop_nD_serial(
        nloop             ,
        ncoeff            ,
        MassSum           ,
        all_coeffs        ,
        grad_pot_fft 
    ):

    geodim = all_coeffs.shape[1]
    
    Kin_en = 0.
    Action_grad = np.empty((nloop,geodim,ncoeff,2),np.float64)

    for il in range(nloop):
        
        prod_fac = MassSum[il]*cfourpisq
        
        for idim in range(geodim): 

            Action_grad[il,idim,0,0] = - grad_pot_fft[il,idim,0].real
            Action_grad[il,idim,0,1] = 0  

            for k in range(1,ncoeff-1):
                
                k2 = k*k

                a = prod_fac*k2
                b=2*a  

                Kin_en += a *((all_coeffs[il,idim,k,0]*all_coeffs[il,idim,k,0]) + (all_coeffs[il,idim,k,1]*all_coeffs[il,idim,k,1]))
                
                Action_grad[il,idim,k,0] = b*all_coeffs[il,idim,k,0] - 2*grad_pot_fft[il,idim,k].real
                Action_grad[il,idim,k,1] = b*all_coeffs[il,idim,k,1] - 2*grad_pot_fft[il,idim,k].imag


            k = ncoeff-1
            k2 = k*k
            
            a = prod_fac*k2
            b=2*a  

            Kin_en += a *((all_coeffs[il,idim,k,0]*all_coeffs[il,idim,k,0]))
            Action_grad[il,idim,k,0] = b*all_coeffs[il,idim,k,0] - grad_pot_fft[il,idim,k].real

    return Kin_en, Action_grad 
            
@numba.jit(**numba_kwargs)
def Compute_action_Numba_time_loop_nD_serial(
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
):

    geodim = all_pos.shape[1]

    dx  = np.zeros((geodim),dtype=np.float64)
    
    Pot_en = 0.

    grad_pot_all = np.zeros((nloop,2,nint),dtype=np.float64)

    for iint in range(nint):

        # Different loops
        for il in range(nloop):
            for ilp in range(il+1,nloop):

                for ib in range(loopnb[il]):
                    for ibp in range(loopnb[ilp]):

                        shift_i  = (((((iint - ((nint*TimeShiftNumUn[il ,ib ]) // TimeShiftDenUn[il ,ib ])) * TimeRevsUn[il ,ib ]) % nint) + nint) % nint)
                        shift_ip = (((((iint - ((nint*TimeShiftNumUn[ilp,ibp]) // TimeShiftDenUn[ilp,ibp])) * TimeRevsUn[ilp,ibp]) % nint) + nint) % nint)
                        
                        prod_mass = mass[Targets[il,ib]]*mass[Targets[ilp,ibp]]

                        for idim in range(geodim):
                            dx[idim] = SpaceRotsUn[il,ib,idim,0]*all_pos[il,0,shift_i] - SpaceRotsUn[ilp,ibp,idim,0]*all_pos[ilp,0,shift_ip]
                            for jdim in range(1,geodim):
                                dx[idim] = dx[idim] + SpaceRotsUn[il,ib,idim,jdim]*all_pos[il,jdim,shift_i] - SpaceRotsUn[ilp,ibp,idim,jdim]*all_pos[ilp,jdim,shift_ip]

                        dx2 = dx[0]*dx[0]
                        for idim in range(1,geodim):
                            dx2 = dx2 + dx[idim]*dx[idim]
                            
                        pot,potp,potpp = Cpt_interbody_pot(dx2)
                        
                        Pot_en += pot*prod_mass

                        a = (2*prod_mass*potp)

                        for idim in range(geodim):
                            dx[idim] = dx[idim] * a

                        for idim in range(geodim):
                            
                            b = SpaceRotsUn[il,ib,0,idim]*dx[0]
                            for jdim in range(1,geodim):
                                b=b+SpaceRotsUn[il,ib,jdim,idim]*dx[jdim]
                            
                            grad_pot_all[il ,idim,shift_i] += b
                            
                            b = SpaceRotsUn[ilp,ibp,0,idim]*dx[0]
                            for jdim in range(1,geodim):
                                b=b+SpaceRotsUn[ilp,ibp,jdim,idim]*dx[jdim]
                            
                            grad_pot_all[ilp,idim,shift_ip] -= b

        # Same loop + symmetry
        for il in range(nloop):

            for ibi in range(loopnbi[il]):

                shift_i = (((((iint - ((nint*TimeShiftNumBin[il ,ibi]) // TimeShiftDenBin[il ,ibi])) * TimeRevsBin[il ,ibi]) % nint) + nint) % nint)
                
                for idim in range(geodim):
                    dx[idim] = SpaceRotsBin[il,ibi,idim,0]*all_pos[il,0,shift_i]
                    for jdim in range(1,geodim):
                        dx[idim] = dx[idim] + SpaceRotsBin[il,ibi,idim,jdim]*all_pos[il,jdim,shift_i]
                    
                    dx[idim] = dx[idim] - all_pos[il,idim,iint]
                    
                dx2 = dx[0]*dx[0]
                for idim in range(1,geodim):
                    dx2 = dx2 +dx[idim]*dx[idim]

                pot,potp,potpp = Cpt_interbody_pot(dx2)
                
                Pot_en += pot*ProdMassSumAll[il,ibi]
                
                a = (2*ProdMassSumAll[il,ibi]*potp)

                for idim in range(geodim):
                    dx[idim] = a*dx[idim]

                for idim in range(geodim):
                    
                    b = SpaceRotsBin[il,ibi,0,idim]*dx[0]
                    for jdim in range(1,geodim):
                        b= b + SpaceRotsBin[il,ibi,jdim,idim]*dx[jdim]
                    
                    grad_pot_all[il ,idim, shift_i] += b
                    grad_pot_all[il ,idim, iint] -= dx[idim]

    return Pot_en, grad_pot_all

def Compute_action_hess_mul_Numba_nD_serial(
    nloop             ,
    ncoeff            ,
    nint              ,
    mass              ,
    loopnb            ,
    Targets           ,
    MassSum           ,
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
    all_coeffs        ,
    all_coeffs_d      , 
    all_pos 
):

    c_coeffs_d = all_coeffs_d.view(dtype=np.complex128)[...,0]
    all_pos_d = the_irfft(c_coeffs_d,n=nint,axis=2,norm="forward")

    hess_pot_all_d = Compute_action_hess_mul_Numba_time_loop_nD_serial(
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

    hess_dx_pot_fft = the_rfft(hess_pot_all_d,norm="forward")

    return Compute_action_hess_mul_Numba_Kin_loop_nD_serial(
        nloop             ,
        ncoeff            ,
        MassSum           ,
        all_coeffs_d      ,
        hess_dx_pot_fft 
    )

@numba.jit(**numba_kwargs)
def Compute_action_hess_mul_Numba_Kin_loop_nD_serial(
        nloop             ,
        ncoeff            ,
        MassSum           ,
        all_coeffs_d      ,
        hess_dx_pot_fft 
    ):

    geodim = all_coeffs_d.shape[1]

    Action_hess_dx = np.empty((nloop,geodim,ncoeff,2),np.float64)

    for il in range(nloop):
        
        prod_fac = MassSum[il]*cfourpisq
        
        for idim in range(geodim):
            
            Action_hess_dx[il,idim,0,0] = -hess_dx_pot_fft[il,idim,0].real
            Action_hess_dx[il,idim,0,1] = 0 

            for k in range(1,ncoeff-1):
                
                k2 = k*k

                a = 2*prod_fac*k2
                
                Action_hess_dx[il,idim,k,0] = a*all_coeffs_d[il,idim,k,0] - 2*hess_dx_pot_fft[il,idim,k].real
                Action_hess_dx[il,idim,k,1] = a*all_coeffs_d[il,idim,k,1] - 2*hess_dx_pot_fft[il,idim,k].imag

            k = ncoeff-1
            k2 = k*k
            a = 2*prod_fac*k2

            Action_hess_dx[il,idim,k,0] = a*all_coeffs_d[il,idim,k,0] - hess_dx_pot_fft[il,idim,k].real
    return Action_hess_dx

@numba.jit(**numba_kwargs)
def Compute_action_hess_mul_Numba_time_loop_nD_serial(
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
    all_pos_d         ,
):

    geodim = all_pos.shape[1]

    dx  = np.zeros((geodim),dtype=np.float64)
    ddx = np.zeros((geodim),dtype=np.float64)
    ddf = np.zeros((geodim),dtype=np.float64)

    hess_pot_all_d = np.zeros((nloop,2,nint),dtype=np.float64)

    for iint in range(nint):

        # Different loops
        for il in range(nloop):
            for ilp in range(il+1,nloop):

                for ib in range(loopnb[il]):
                    for ibp in range(loopnb[ilp]):

                        shift_i  = (((((iint - ((nint*TimeShiftNumUn[il ,ib ]) // TimeShiftDenUn[il ,ib ])) * TimeRevsUn[il ,ib ]) % nint) + nint) % nint)
                        shift_ip = (((((iint - ((nint*TimeShiftNumUn[ilp,ibp]) // TimeShiftDenUn[ilp,ibp])) * TimeRevsUn[ilp,ibp]) % nint) + nint) % nint)
                        
                        prod_mass = mass[Targets[il,ib]]*mass[Targets[ilp,ibp]]

                        for idim in range(geodim):
                            dx[idim] = SpaceRotsUn[il,ib,idim,0]*all_pos[il,0,shift_i] - SpaceRotsUn[ilp,ibp,idim,0]*all_pos[ilp,0,shift_ip]
                            ddx[idim] = SpaceRotsUn[il,ib,idim,0]*all_pos_d[il,0,shift_i] - SpaceRotsUn[ilp,ibp,idim,0]*all_pos_d[ilp,0,shift_ip]
                            for jdim in range(1,geodim):
                                dx[idim] = dx[idim] + SpaceRotsUn[il,ib,idim,jdim]*all_pos[il,jdim,shift_i] - SpaceRotsUn[ilp,ibp,idim,jdim]*all_pos[ilp,jdim,shift_ip]
                                ddx[idim] = ddx[idim] + SpaceRotsUn[il,ib,idim,jdim]*all_pos_d[il,jdim,shift_i] - SpaceRotsUn[ilp,ibp,idim,jdim]*all_pos_d[ilp,jdim,shift_ip]

                        dx2 = dx[0]*dx[0]
                        dxtddx = dx[0]*ddx[0]
                        for idim in range(1,geodim):
                            dx2 = dx2 + dx[idim]*dx[idim]
                            dxtddx = dxtddx + dx[idim]*ddx[idim]
                            
                        pot,potp,potpp = Cpt_interbody_pot(dx2)

                        a = (2*prod_mass*potp)
                        b = (4*prod_mass*potpp*dxtddx)
                        
                        for idim in range(geodim):
                            ddf[idim] = b*dx[idim]+a*ddx[idim]
                            
                        for idim in range(geodim):
                            
                            c = SpaceRotsUn[il,ib,0,idim]*ddf[0]
                            for jdim in range(1,geodim):
                                c = c+SpaceRotsUn[il,ib,jdim,idim]*ddf[jdim]
                            
                            hess_pot_all_d[il ,idim,shift_i] += c
                            
                            c = SpaceRotsUn[ilp,ibp,0,idim]*ddf[0]
                            for jdim in range(1,geodim):
                                c = c+SpaceRotsUn[ilp,ibp,jdim,idim]*ddf[jdim]
                            
                            hess_pot_all_d[ilp,idim,shift_ip] -= c

        # Same loop + symmetry
        for il in range(nloop):

            for ibi in range(loopnbi[il]):

                shift_i = (((((iint - ((nint*TimeShiftNumBin[il ,ibi]) // TimeShiftDenBin[il ,ibi])) * TimeRevsBin[il ,ibi]) % nint) + nint) % nint)
                
                for idim in range(geodim):
                    dx[idim]  = SpaceRotsBin[il,ibi,idim,0]*all_pos[il,0,shift_i]
                    ddx[idim] = SpaceRotsBin[il,ibi,idim,0]*all_pos_d[il,0,shift_i]
                    for jdim in range(1,geodim):
                        dx[idim]  = dx[idim] + SpaceRotsBin[il,ibi,idim,jdim]*all_pos[il,jdim,shift_i]
                        ddx[idim] = ddx[idim] + SpaceRotsBin[il,ibi,idim,jdim]*all_pos_d[il,jdim,shift_i]
                    
                    dx[idim]  = dx[idim] - all_pos[il,idim,iint]
                    ddx[idim] = ddx[idim] - all_pos_d[il,idim,iint]
                    
                dx2 = dx[0]*dx[0]
                dxtddx = dx[0]*ddx[0]
                for idim in range(1,geodim):
                    dx2 = dx2 + dx[idim]*dx[idim]
                    dxtddx = dxtddx + dx[idim]*ddx[idim]

                pot,potp,potpp = Cpt_interbody_pot(dx2)
                
                a = (2*ProdMassSumAll[il,ibi]*potp)
                b = (4*ProdMassSumAll[il,ibi]*potpp*dxtddx)
        
                for idim in range(geodim):
                    ddf[idim] = b*dx[idim]+a*ddx[idim]

                for idim in range(geodim):
                    
                    c = SpaceRotsBin[il,ibi,0,idim]*ddf[0]
                    for jdim in range(1,geodim):
                        c = c+SpaceRotsBin[il,ibi,jdim,idim]*ddf[jdim]
                    
                    hess_pot_all_d[il ,idim,shift_i] += c
                    hess_pot_all_d[il ,idim,iint] -= ddf[idim]

    return hess_pot_all_d

def Compute_action_Numba_2D_serial(
    nloop             ,
    ncoeff            ,
    nint              ,
    mass              ,
    loopnb            ,
    Targets           ,
    MassSum           ,
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
    all_coeffs        ,
    all_pos 
):
    # This function is probably the most important one.
    # Computes the action and its gradient with respect to the Fourier coefficients of the generator in each loop.
    
    Pot_en, grad_pot_all = Compute_action_Numba_time_loop_2D_serial(
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
    )

    grad_pot_fft = the_rfft(grad_pot_all,norm="forward")  #

    Kin_en, Action_grad = Compute_action_Numba_Kin_loop_nD_serial(
        nloop             ,
        ncoeff            ,
        MassSum           ,
        all_coeffs        ,
        grad_pot_fft 
    )

    Action = Kin_en-Pot_en/nint
    
    return Action, Action_grad

@numba.jit(**numba_kwargs)
def Compute_action_Numba_Kin_loop_nD_serial(
        nloop             ,
        ncoeff            ,
        MassSum           ,
        all_coeffs        ,
        grad_pot_fft 
):
    
    Kin_en = 0.
    Action_grad = np.empty((nloop,2,ncoeff,2),np.float64)

    for il in range(nloop):
        
        prod_fac = MassSum[il]*cfourpisq
        
        for idim in range(2): 

            Action_grad[il,idim,0,0] = - grad_pot_fft[il,idim,0].real
            Action_grad[il,idim,0,1] = 0  
            
            for k in range(1,ncoeff-1):
                
                k2 = k*k
                
                a = prod_fac*k2
                b = 2*a  

                Kin_en += a *((all_coeffs[il,idim,k,0]*all_coeffs[il,idim,k,0]) + (all_coeffs[il,idim,k,1]*all_coeffs[il,idim,k,1]))
                
                Action_grad[il,idim,k,0] = b*all_coeffs[il,idim,k,0] - 2*grad_pot_fft[il,idim,k].real
                Action_grad[il,idim,k,1] = b*all_coeffs[il,idim,k,1] - 2*grad_pot_fft[il,idim,k].imag
            

            k = ncoeff-1
            k2 = k*k
            
            a = prod_fac*k2
            b = 2*a  

            Kin_en += a *((all_coeffs[il,idim,k,0]*all_coeffs[il,idim,k,0]))
            Action_grad[il,idim,k,0] = b*all_coeffs[il,idim,k,0] - grad_pot_fft[il,idim,k].real

    return Kin_en, Action_grad

@numba.jit(**numba_kwargs)
def Compute_action_Numba_time_loop_2D_serial(
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
):
    
    Pot_en = 0.
    grad_pot_all = np.zeros((nloop,2,nint),dtype=np.float64)

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
                            
                        pot,potp,potpp = Cpt_interbody_pot(dx2)
                        
                        Pot_en += pot*prod_mass

                        a = (2*prod_mass*potp)

                        dx0 = a * dx0
                        dx1 = a * dx1

                        grad_pot_all[il ,0,shift_i ] += SpaceRotsUn[il ,ib ,0,0] * dx0 + SpaceRotsUn[il ,ib ,1,0] * dx1
                        grad_pot_all[ilp,0,shift_ip] -= SpaceRotsUn[ilp,ibp,0,0] * dx0 + SpaceRotsUn[ilp,ibp,1,0] * dx1
  
                        grad_pot_all[il ,1,shift_i ] += SpaceRotsUn[il ,ib ,0,1] * dx0 + SpaceRotsUn[il ,ib ,1,1] * dx1
                        grad_pot_all[ilp,1,shift_ip] -= SpaceRotsUn[ilp,ibp,0,1] * dx0 + SpaceRotsUn[ilp,ibp,1,1] * dx1

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

                pot,potp,potpp = Cpt_interbody_pot(dx2)
                
                Pot_en += pot*ProdMassSumAll[il,ibi]
                
                a = (2*ProdMassSumAll[il,ibi]*potp)

                dx0 = dx0*a
                dx1 = dx1*a

                b = SpaceRotsBin[il,ibi,0,0]*dx0 + SpaceRotsBin[il,ibi,1,0]*dx1

                grad_pot_all[il ,0,shift_i] += b
                grad_pot_all[il ,0,iint   ] -= dx0

                b = SpaceRotsBin[il,ibi,0,1]*dx0 + SpaceRotsBin[il,ibi,1,1]*dx1

                grad_pot_all[il ,1,shift_i] += b
                grad_pot_all[il ,1,iint   ] -= dx1

    return Pot_en, grad_pot_all

def Compute_action_hess_mul_Numba_2D_serial(
    nloop             ,
    ncoeff            ,
    nint              ,
    mass              ,
    loopnb            ,
    Targets           ,
    MassSum           ,
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
    all_coeffs        ,
    all_coeffs_d      , 
    all_pos 
):
    # Computes the matrix vector product H*dx where H is the Hessian of the action.
    # Useful to guide the root finding / optimisation process and to better understand the topography of the action (critical points / Morse theory).

    c_coeffs_d = all_coeffs_d.view(dtype=np.complex128)[...,0]
    all_pos_d = the_irfft(c_coeffs_d,norm="forward")

    hess_pot_all_d = Compute_action_hess_mul_Numba_time_loop_2D_serial(
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

    hess_dx_pot_fft = the_rfft(hess_pot_all_d,norm="forward")

    return Compute_action_hess_mul_Numba_Kin_loop_nD_serial(
        nloop             ,
        ncoeff            ,
        MassSum           ,
        all_coeffs_d      ,
        hess_dx_pot_fft 
    )

@numba.jit(**numba_kwargs)
def Compute_action_hess_mul_Numba_Kin_loop_nD_serial(
        nloop             ,
        ncoeff            ,
        MassSum           ,
        all_coeffs_d      ,
        hess_dx_pot_fft 
):

    Action_hess_dx = np.empty((nloop,2,ncoeff,2),np.float64)

    for il in range(nloop):
        
        prod_fac = MassSum[il]*cfourpisq
        
        for idim in range(2):
            
            Action_hess_dx[il,idim,0,0] = - hess_dx_pot_fft[il,idim,0].real
            Action_hess_dx[il,idim,0,1] = 0.            

            for k in range(1,ncoeff-1):
                
                k2 = k*k
                a = 2*prod_fac*k2

                Action_hess_dx[il,idim,k,0] = a*all_coeffs_d[il,idim,k,0] - 2*hess_dx_pot_fft[il,idim,k].real
                Action_hess_dx[il,idim,k,1] = a*all_coeffs_d[il,idim,k,1] - 2*hess_dx_pot_fft[il,idim,k].imag

            k = ncoeff-1
            k2 = k*k
            a = 2*prod_fac*k2

            Action_hess_dx[il,idim,k,0] = a*all_coeffs_d[il,idim,k,0] - hess_dx_pot_fft[il,idim,k].real

    return Action_hess_dx

@numba.jit(**numba_kwargs)
def Compute_action_hess_mul_Numba_time_loop_2D_serial(
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
    all_pos_d         ,
):

    hess_pot_all_d = np.zeros((nloop,2,nint),dtype=np.float64)

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
                            
                        pot,potp,potpp = Cpt_interbody_pot(dx2)

                        a = (2*prod_mass*potp)
                        b = (4*prod_mass*potpp*dxtddx)
                        
                        ddf0 = b*dx0+a*ddx0
                        ddf1 = b*dx1+a*ddx1
                            
                        hess_pot_all_d[il ,0,shift_i ] += SpaceRotsUn[il ,ib ,0,0] * ddf0 + SpaceRotsUn[il ,ib ,1,0] * ddf1
                        hess_pot_all_d[ilp,0,shift_ip] -= SpaceRotsUn[ilp,ibp,0,0] * ddf0 + SpaceRotsUn[ilp,ibp,1,0] * ddf1

                        hess_pot_all_d[il ,1,shift_i ] += SpaceRotsUn[il ,ib ,0,1] * ddf0 + SpaceRotsUn[il ,ib ,1,1] * ddf1
                        hess_pot_all_d[ilp,1,shift_ip] -= SpaceRotsUn[ilp,ibp,0,1] * ddf0 + SpaceRotsUn[ilp,ibp,1,1] * ddf1


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

                pot,potp,potpp = Cpt_interbody_pot(dx2)
                
                a = (2*ProdMassSumAll[il,ibi]*potp)
                b = (4*ProdMassSumAll[il,ibi]*potpp*dxtddx)
        
                ddf0 = b*dx0+a*ddx0
                ddf1 = b*dx1+a*ddx1

                hess_pot_all_d[il ,0,shift_i] += SpaceRotsBin[il,ibi,0,0]*ddf0 + SpaceRotsBin[il,ibi,1,0]*ddf1
                hess_pot_all_d[il ,0,iint   ] -= ddf0

                hess_pot_all_d[il ,1,shift_i] += SpaceRotsBin[il,ibi,0,1]*ddf0 + SpaceRotsBin[il,ibi,1,1]*ddf1
                hess_pot_all_d[il ,1,iint   ] -= ddf1

    return hess_pot_all_d

def Compute_action_Numba_nD_parallel(
    nloop             ,
    ncoeff            ,
    nint              ,
    mass              ,
    loopnb            ,
    Targets           ,
    MassSum           ,
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
    all_coeffs        ,
    all_pos 
):
    # This function is probably the most important one.
    # Computes the action and its gradient with respect to the Fourier coefficients of the generator in each loop.
    
    Pot_en, grad_pot_all = Compute_action_Numba_time_loop_nD_parallel(
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

    grad_pot_fft = the_rfft(grad_pot_all,norm="forward") 

    Kin_en, Action_grad = Compute_action_Numba_Kin_loop_nD_parallel(
        nloop             ,
        ncoeff            ,
        MassSum           ,
        all_coeffs        ,
        grad_pot_fft 
    )

    Action = Kin_en-Pot_en/nint
    
    return Action,Action_grad

@numba.jit(parallel=True,**numba_kwargs)
def Compute_action_Numba_Kin_loop_nD_parallel(
        nloop             ,
        ncoeff            ,
        MassSum           ,
        all_coeffs        ,
        grad_pot_fft 
    ):

    geodim = all_coeffs.shape[1]
    
    Kin_en = 0.
    Action_grad = np.empty((nloop,geodim,ncoeff,2),np.float64)

    for il in numba.prange(nloop):
        
        prod_fac = MassSum[il]*cfourpisq
        
        for idim in range(geodim): 

            Action_grad[il,idim,0,0] = - grad_pot_fft[il,idim,0].real
            Action_grad[il,idim,0,1] = 0  

            for k in range(1,ncoeff-1):
                
                k2 = k*k

                a = prod_fac*k2
                b=2*a  

                Kin_en += a *((all_coeffs[il,idim,k,0]*all_coeffs[il,idim,k,0]) + (all_coeffs[il,idim,k,1]*all_coeffs[il,idim,k,1]))
                
                Action_grad[il,idim,k,0] = b*all_coeffs[il,idim,k,0] - 2*grad_pot_fft[il,idim,k].real
                Action_grad[il,idim,k,1] = b*all_coeffs[il,idim,k,1] - 2*grad_pot_fft[il,idim,k].imag


            k = ncoeff-1
            k2 = k*k
            
            a = prod_fac*k2
            b=2*a  

            Kin_en += a *((all_coeffs[il,idim,k,0]*all_coeffs[il,idim,k,0]))
            Action_grad[il,idim,k,0] = b*all_coeffs[il,idim,k,0] - grad_pot_fft[il,idim,k].real

    return Kin_en, Action_grad 
            
@numba.jit(parallel=True,**numba_kwargs)
def Compute_action_Numba_time_loop_nD_parallel(
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
):

    geodim = all_pos.shape[1]

    # tot_rk = numba.get_num_threads()
    tot_rk = max_num_threads

    Pot_en = np.zeros((tot_rk),dtype=np.float64)

    grad_pot_all = np.zeros((tot_rk,nloop,2,nint),dtype=np.float64)

    for iint in numba.prange(nint):

        dx  = np.empty((geodim),dtype=np.float64) # hoisting ?

        rk = numba.get_thread_id()

        # Different loops
        for il in range(nloop):
            for ilp in range(il+1,nloop):

                for ib in range(loopnb[il]):
                    for ibp in range(loopnb[ilp]):

                        shift_i  = (((((iint - ((nint*TimeShiftNumUn[il ,ib ]) // TimeShiftDenUn[il ,ib ])) * TimeRevsUn[il ,ib ]) % nint) + nint) % nint)
                        shift_ip = (((((iint - ((nint*TimeShiftNumUn[ilp,ibp]) // TimeShiftDenUn[ilp,ibp])) * TimeRevsUn[ilp,ibp]) % nint) + nint) % nint)
                        
                        prod_mass = mass[Targets[il,ib]]*mass[Targets[ilp,ibp]]

                        for idim in range(geodim):
                            dx[idim] = SpaceRotsUn[il,ib,idim,0]*all_pos[il,0,shift_i] - SpaceRotsUn[ilp,ibp,idim,0]*all_pos[ilp,0,shift_ip]
                            for jdim in range(1,geodim):
                                dx[idim] = dx[idim] + SpaceRotsUn[il,ib,idim,jdim]*all_pos[il,jdim,shift_i] - SpaceRotsUn[ilp,ibp,idim,jdim]*all_pos[ilp,jdim,shift_ip]

                        dx2 = dx[0]*dx[0]
                        for idim in range(1,geodim):
                            dx2 = dx2 + dx[idim]*dx[idim]
                            
                        pot,potp,potpp = Cpt_interbody_pot(dx2)
                        
                        Pot_en[rk] += pot*prod_mass

                        a = (2*prod_mass*potp)

                        for idim in range(geodim):
                            dx[idim] = dx[idim] * a

                        for idim in range(geodim):
                            
                            b = SpaceRotsUn[il,ib,0,idim]*dx[0]
                            for jdim in range(1,geodim):
                                b=b+SpaceRotsUn[il,ib,jdim,idim]*dx[jdim]
                            
                            grad_pot_all[rk, il ,idim,shift_i] += b
                            
                            b = SpaceRotsUn[ilp,ibp,0,idim]*dx[0]
                            for jdim in range(1,geodim):
                                b=b+SpaceRotsUn[ilp,ibp,jdim,idim]*dx[jdim]
                            
                            grad_pot_all[rk, ilp,idim,shift_ip] -= b

        # Same loop + symmetry
        for il in range(nloop):

            for ibi in range(loopnbi[il]):

                shift_i = (((((iint - ((nint*TimeShiftNumBin[il ,ibi]) // TimeShiftDenBin[il ,ibi])) * TimeRevsBin[il ,ibi]) % nint) + nint) % nint)
                
                for idim in range(geodim):
                    dx[idim] = SpaceRotsBin[il,ibi,idim,0]*all_pos[il,0,shift_i]
                    for jdim in range(1,geodim):
                        dx[idim] = dx[idim] + SpaceRotsBin[il,ibi,idim,jdim]*all_pos[il,jdim,shift_i]
                    
                    dx[idim] = dx[idim] - all_pos[il,idim,iint]
                    
                dx2 = dx[0]*dx[0]
                for idim in range(1,geodim):
                    dx2 = dx2 +dx[idim]*dx[idim]

                pot,potp,potpp = Cpt_interbody_pot(dx2)
                
                Pot_en[rk] += pot*ProdMassSumAll[il,ibi]
                
                a = (2*ProdMassSumAll[il,ibi]*potp)

                for idim in range(geodim):
                    dx[idim] = a*dx[idim]

                for idim in range(geodim):
                    
                    b = SpaceRotsBin[il,ibi,0,idim]*dx[0]
                    for jdim in range(1,geodim):
                        b= b + SpaceRotsBin[il,ibi,jdim,idim]*dx[jdim]
                    
                    grad_pot_all[rk, il ,idim, shift_i] += b
                    grad_pot_all[rk, il ,idim, iint] -= dx[idim]

    return Pot_en.sum(), grad_pot_all.sum(axis=0)

def Compute_action_hess_mul_Numba_nD_parallel(
    nloop             ,
    ncoeff            ,
    nint              ,
    mass              ,
    loopnb            ,
    Targets           ,
    MassSum           ,
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
    all_coeffs        ,
    all_coeffs_d      , 
    all_pos 
):

    c_coeffs_d = all_coeffs_d.view(dtype=np.complex128)[...,0]
    all_pos_d = the_irfft(c_coeffs_d,n=nint,axis=2,norm="forward")

    hess_pot_all_d = Compute_action_hess_mul_Numba_time_loop_nD_parallel(
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

    hess_dx_pot_fft = the_rfft(hess_pot_all_d,norm="forward")

    return Compute_action_hess_mul_Numba_Kin_loop_nD_parallel(
        nloop             ,
        ncoeff            ,
        MassSum           ,
        all_coeffs_d      ,
        hess_dx_pot_fft 
    )

@numba.jit(parallel=True,**numba_kwargs)
def Compute_action_hess_mul_Numba_Kin_loop_nD_parallel(
        nloop             ,
        ncoeff            ,
        MassSum           ,
        all_coeffs_d      ,
        hess_dx_pot_fft 
    ):

    geodim = all_coeffs_d.shape[1]

    Action_hess_dx = np.empty((nloop,geodim,ncoeff,2),np.float64)

    for il in numba.prange(nloop):
        
        prod_fac = MassSum[il]*cfourpisq
        
        for idim in range(geodim):
            
            Action_hess_dx[il,idim,0,0] = -hess_dx_pot_fft[il,idim,0].real
            Action_hess_dx[il,idim,0,1] = 0 

            for k in range(1,ncoeff-1):
                
                k2 = k*k

                a = 2*prod_fac*k2
                
                Action_hess_dx[il,idim,k,0] = a*all_coeffs_d[il,idim,k,0] - 2*hess_dx_pot_fft[il,idim,k].real
                Action_hess_dx[il,idim,k,1] = a*all_coeffs_d[il,idim,k,1] - 2*hess_dx_pot_fft[il,idim,k].imag

            k = ncoeff-1
            k2 = k*k
            a = 2*prod_fac*k2

            Action_hess_dx[il,idim,k,0] = a*all_coeffs_d[il,idim,k,0] - hess_dx_pot_fft[il,idim,k].real

    return Action_hess_dx

@numba.jit(parallel=True,**numba_kwargs)
def Compute_action_hess_mul_Numba_time_loop_nD_parallel(
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
    all_pos_d         ,
):

    geodim = all_pos.shape[1]

    # tot_rk = numba.get_num_threads()
    tot_rk = max_num_threads

    hess_pot_all_d = np.zeros((tot_rk,nloop,2,nint),dtype=np.float64)

    for iint in numba.prange(nint):

        dx  = np.empty((geodim),dtype=np.float64) # hoisting ?
        ddx = np.empty((geodim),dtype=np.float64) # hoisting ?
        ddf = np.empty((geodim),dtype=np.float64) # hoisting ?

        rk = numba.get_thread_id()

        # Different loops
        for il in range(nloop):
            for ilp in range(il+1,nloop):

                for ib in range(loopnb[il]):
                    for ibp in range(loopnb[ilp]):

                        shift_i  = (((((iint - ((nint*TimeShiftNumUn[il ,ib ]) // TimeShiftDenUn[il ,ib ])) * TimeRevsUn[il ,ib ]) % nint) + nint) % nint)
                        shift_ip = (((((iint - ((nint*TimeShiftNumUn[ilp,ibp]) // TimeShiftDenUn[ilp,ibp])) * TimeRevsUn[ilp,ibp]) % nint) + nint) % nint)
                        
                        prod_mass = mass[Targets[il,ib]]*mass[Targets[ilp,ibp]]

                        for idim in range(geodim):
                            dx[idim] = SpaceRotsUn[il,ib,idim,0]*all_pos[il,0,shift_i] - SpaceRotsUn[ilp,ibp,idim,0]*all_pos[ilp,0,shift_ip]
                            ddx[idim] = SpaceRotsUn[il,ib,idim,0]*all_pos_d[il,0,shift_i] - SpaceRotsUn[ilp,ibp,idim,0]*all_pos_d[ilp,0,shift_ip]
                            for jdim in range(1,geodim):
                                dx[idim] = dx[idim] + SpaceRotsUn[il,ib,idim,jdim]*all_pos[il,jdim,shift_i] - SpaceRotsUn[ilp,ibp,idim,jdim]*all_pos[ilp,jdim,shift_ip]
                                ddx[idim] = ddx[idim] + SpaceRotsUn[il,ib,idim,jdim]*all_pos_d[il,jdim,shift_i] - SpaceRotsUn[ilp,ibp,idim,jdim]*all_pos_d[ilp,jdim,shift_ip]

                        dx2 = dx[0]*dx[0]
                        dxtddx = dx[0]*ddx[0]
                        for idim in range(1,geodim):
                            dx2 = dx2 + dx[idim]*dx[idim]
                            dxtddx = dxtddx + dx[idim]*ddx[idim]
                            
                        pot,potp,potpp = Cpt_interbody_pot(dx2)

                        a = (2*prod_mass*potp)
                        b = (4*prod_mass*potpp*dxtddx)
                        
                        for idim in range(geodim):
                            ddf[idim] = b*dx[idim]+a*ddx[idim]
                            
                        for idim in range(geodim):
                            
                            c = SpaceRotsUn[il,ib,0,idim]*ddf[0]
                            for jdim in range(1,geodim):
                                c = c+SpaceRotsUn[il,ib,jdim,idim]*ddf[jdim]
                            
                            hess_pot_all_d[rk, il ,idim,shift_i] += c
                            
                            c = SpaceRotsUn[ilp,ibp,0,idim]*ddf[0]
                            for jdim in range(1,geodim):
                                c = c+SpaceRotsUn[ilp,ibp,jdim,idim]*ddf[jdim]
                            
                            hess_pot_all_d[rk, ilp,idim,shift_ip] -= c

        # Same loop + symmetry
        for il in range(nloop):

            for ibi in range(loopnbi[il]):

                shift_i = (((((iint - ((nint*TimeShiftNumBin[il ,ibi]) // TimeShiftDenBin[il ,ibi])) * TimeRevsBin[il ,ibi]) % nint) + nint) % nint)
                
                for idim in range(geodim):
                    dx[idim]  = SpaceRotsBin[il,ibi,idim,0]*all_pos[il,0,shift_i]
                    ddx[idim] = SpaceRotsBin[il,ibi,idim,0]*all_pos_d[il,0,shift_i]
                    for jdim in range(1,geodim):
                        dx[idim]  = dx[idim] + SpaceRotsBin[il,ibi,idim,jdim]*all_pos[il,jdim,shift_i]
                        ddx[idim] = ddx[idim] + SpaceRotsBin[il,ibi,idim,jdim]*all_pos_d[il,jdim,shift_i]
                    
                    dx[idim]  = dx[idim] - all_pos[il,idim,iint]
                    ddx[idim] = ddx[idim] - all_pos_d[il,idim,iint]
                    
                dx2 = dx[0]*dx[0]
                dxtddx = dx[0]*ddx[0]
                for idim in range(1,geodim):
                    dx2 = dx2 + dx[idim]*dx[idim]
                    dxtddx = dxtddx + dx[idim]*ddx[idim]

                pot,potp,potpp = Cpt_interbody_pot(dx2)
                
                a = (2*ProdMassSumAll[il,ibi]*potp)
                b = (4*ProdMassSumAll[il,ibi]*potpp*dxtddx)
        
                for idim in range(geodim):
                    ddf[idim] = b*dx[idim]+a*ddx[idim]

                for idim in range(geodim):
                    
                    c = SpaceRotsBin[il,ibi,0,idim]*ddf[0]
                    for jdim in range(1,geodim):
                        c = c+SpaceRotsBin[il,ibi,jdim,idim]*ddf[jdim]
                    
                    hess_pot_all_d[rk, il ,idim,shift_i] += c
                    hess_pot_all_d[rk, il ,idim,iint] -= ddf[idim]

    return hess_pot_all_d.sum(axis=0)

def Compute_action_Numba_2D_parallel(
    nloop             ,
    ncoeff            ,
    nint              ,
    mass              ,
    loopnb            ,
    Targets           ,
    MassSum           ,
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
    all_coeffs        ,
    all_pos 
):
    # This function is probably the most important one.
    # Computes the action and its gradient with respect to the Fourier coefficients of the generator in each loop.
    
    Pot_en, grad_pot_all = Compute_action_Numba_time_loop_2D_parallel(
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
    )

    grad_pot_fft = the_rfft(grad_pot_all,norm="forward")  #

    Kin_en, Action_grad = Compute_action_Numba_Kin_loop_nD_parallel(
        nloop             ,
        ncoeff            ,
        MassSum           ,
        all_coeffs        ,
        grad_pot_fft 
    )

    Action = Kin_en-Pot_en/nint
    
    return Action, Action_grad

@numba.jit(parallel=True,**numba_kwargs)
def Compute_action_Numba_Kin_loop_nD_parallel(
        nloop             ,
        ncoeff            ,
        MassSum           ,
        all_coeffs        ,
        grad_pot_fft 
):
    
    Kin_en = 0.
    Action_grad = np.empty((nloop,2,ncoeff,2),np.float64)

    for il in numba.prange(nloop):
        
        prod_fac = MassSum[il]*cfourpisq
        
        for idim in range(2): 

            Action_grad[il,idim,0,0] = - grad_pot_fft[il,idim,0].real
            Action_grad[il,idim,0,1] = 0  
            
            for k in range(1,ncoeff-1):
                
                k2 = k*k
                
                a = prod_fac*k2
                b = 2*a  

                Kin_en += a *((all_coeffs[il,idim,k,0]*all_coeffs[il,idim,k,0]) + (all_coeffs[il,idim,k,1]*all_coeffs[il,idim,k,1]))
                
                Action_grad[il,idim,k,0] = b*all_coeffs[il,idim,k,0] - 2*grad_pot_fft[il,idim,k].real
                Action_grad[il,idim,k,1] = b*all_coeffs[il,idim,k,1] - 2*grad_pot_fft[il,idim,k].imag
            

            k = ncoeff-1
            k2 = k*k
            
            a = prod_fac*k2
            b = 2*a  

            Kin_en += a *((all_coeffs[il,idim,k,0]*all_coeffs[il,idim,k,0]))
            Action_grad[il,idim,k,0] = b*all_coeffs[il,idim,k,0] - grad_pot_fft[il,idim,k].real

    return Kin_en, Action_grad

@numba.jit(parallel=True,**numba_kwargs)
def Compute_action_Numba_time_loop_2D_parallel(
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
):
    
    # tot_rk = numba.get_num_threads()
    tot_rk = max_num_threads

    Pot_en = np.zeros((tot_rk),dtype=np.float64)
    grad_pot_all = np.zeros((tot_rk,nloop,2,nint),dtype=np.float64)

    for iint in numba.prange(nint):

        rk = numba.get_thread_id()

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
                            
                        pot,potp,potpp = Cpt_interbody_pot(dx2)
                        
                        Pot_en[rk] += pot*prod_mass

                        a = (2*prod_mass*potp)

                        dx0 = a * dx0
                        dx1 = a * dx1

                        grad_pot_all[rk, il ,0,shift_i ] += SpaceRotsUn[il ,ib ,0,0] * dx0 + SpaceRotsUn[il ,ib ,1,0] * dx1
                        grad_pot_all[rk, ilp,0,shift_ip] -= SpaceRotsUn[ilp,ibp,0,0] * dx0 + SpaceRotsUn[ilp,ibp,1,0] * dx1
  
                        grad_pot_all[rk, il ,1,shift_i ] += SpaceRotsUn[il ,ib ,0,1] * dx0 + SpaceRotsUn[il ,ib ,1,1] * dx1
                        grad_pot_all[rk, ilp,1,shift_ip] -= SpaceRotsUn[ilp,ibp,0,1] * dx0 + SpaceRotsUn[ilp,ibp,1,1] * dx1

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

                pot,potp,potpp = Cpt_interbody_pot(dx2)
                
                Pot_en[rk] += pot*ProdMassSumAll[il,ibi]
                
                a = (2*ProdMassSumAll[il,ibi]*potp)

                dx0 = dx0*a
                dx1 = dx1*a

                b = SpaceRotsBin[il,ibi,0,0]*dx0 + SpaceRotsBin[il,ibi,1,0]*dx1

                grad_pot_all[rk, il ,0,shift_i] += b
                grad_pot_all[rk, il ,0,iint   ] -= dx0

                b = SpaceRotsBin[il,ibi,0,1]*dx0 + SpaceRotsBin[il,ibi,1,1]*dx1

                grad_pot_all[rk, il ,1,shift_i] += b
                grad_pot_all[rk, il ,1,iint   ] -= dx1

    return Pot_en.sum(), grad_pot_all.sum(axis=0)

def Compute_action_hess_mul_Numba_2D_parallel(
    nloop             ,
    ncoeff            ,
    nint              ,
    mass              ,
    loopnb            ,
    Targets           ,
    MassSum           ,
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
    all_coeffs        ,
    all_coeffs_d      , 
    all_pos 
):
    # Computes the matrix vector product H*dx where H is the Hessian of the action.
    # Useful to guide the root finding / optimisation process and to better understand the topography of the action (critical points / Morse theory).

    c_coeffs_d = all_coeffs_d.view(dtype=np.complex128)[...,0]
    all_pos_d = the_irfft(c_coeffs_d,norm="forward")

    hess_pot_all_d = Compute_action_hess_mul_Numba_time_loop_2D_parallel(
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

    hess_dx_pot_fft = the_rfft(hess_pot_all_d,norm="forward")

    return Compute_action_hess_mul_Numba_Kin_loop_2D_parallel(
        nloop             ,
        ncoeff            ,
        MassSum           ,
        all_coeffs_d      ,
        hess_dx_pot_fft 
    )

@numba.jit(parallel=True,**numba_kwargs)
def Compute_action_hess_mul_Numba_Kin_loop_2D_parallel(
        nloop             ,
        ncoeff            ,
        MassSum           ,
        all_coeffs_d      ,
        hess_dx_pot_fft 
):

    Action_hess_dx = np.empty((nloop,2,ncoeff,2),np.float64)

    for il in numba.prange(nloop):
        
        prod_fac = MassSum[il]*cfourpisq
        
        for idim in range(2):
            
            Action_hess_dx[il,idim,0,0] = - hess_dx_pot_fft[il,idim,0].real
            Action_hess_dx[il,idim,0,1] = 0.            

            for k in range(1,ncoeff-1):
                
                k2 = k*k
                a = 2*prod_fac*k2

                Action_hess_dx[il,idim,k,0] = a*all_coeffs_d[il,idim,k,0] - 2*hess_dx_pot_fft[il,idim,k].real
                Action_hess_dx[il,idim,k,1] = a*all_coeffs_d[il,idim,k,1] - 2*hess_dx_pot_fft[il,idim,k].imag

            k = ncoeff-1
            k2 = k*k
            a = 2*prod_fac*k2

            Action_hess_dx[il,idim,k,0] = a*all_coeffs_d[il,idim,k,0] - hess_dx_pot_fft[il,idim,k].real

    return Action_hess_dx

@numba.jit(parallel={'prange':True,'reduction':False},**numba_kwargs)
def Compute_action_hess_mul_Numba_time_loop_2D_parallel(
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
    all_pos_d         ,
):
    
    # tot_rk = numba.get_num_threads()
    tot_rk = max_num_threads

    hess_pot_all_d = np.zeros((tot_rk,nloop,2,nint),dtype=np.float64)

    for iint in numba.prange(nint):

        rk = numba.get_thread_id()

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
                            
                        pot,potp,potpp = Cpt_interbody_pot(dx2)

                        a = (2*prod_mass*potp)
                        b = (4*prod_mass*potpp*dxtddx)
                        
                        ddf0 = b*dx0+a*ddx0
                        ddf1 = b*dx1+a*ddx1
                            
                        hess_pot_all_d[rk, il ,0,shift_i ] += SpaceRotsUn[il ,ib ,0,0] * ddf0 + SpaceRotsUn[il ,ib ,1,0] * ddf1
                        hess_pot_all_d[rk, ilp,0,shift_ip] -= SpaceRotsUn[ilp,ibp,0,0] * ddf0 + SpaceRotsUn[ilp,ibp,1,0] * ddf1

                        hess_pot_all_d[rk, il ,1,shift_i ] += SpaceRotsUn[il ,ib ,0,1] * ddf0 + SpaceRotsUn[il ,ib ,1,1] * ddf1
                        hess_pot_all_d[rk, ilp,1,shift_ip] -= SpaceRotsUn[ilp,ibp,0,1] * ddf0 + SpaceRotsUn[ilp,ibp,1,1] * ddf1


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

                pot,potp,potpp = Cpt_interbody_pot(dx2)
                
                a = (2*ProdMassSumAll[il,ibi]*potp)
                b = (4*ProdMassSumAll[il,ibi]*potpp*dxtddx)
        
                ddf0 = b*dx0+a*ddx0
                ddf1 = b*dx1+a*ddx1

                hess_pot_all_d[rk, il ,0,shift_i] += SpaceRotsBin[il,ibi,0,0]*ddf0 + SpaceRotsBin[il,ibi,1,0]*ddf1
                hess_pot_all_d[rk, il ,0,iint   ] -= ddf0

                hess_pot_all_d[rk, il ,1,shift_i] += SpaceRotsBin[il,ibi,0,1]*ddf0 + SpaceRotsBin[il,ibi,1,1]*ddf1
                hess_pot_all_d[rk, il ,1,iint   ] -= ddf1

    return hess_pot_all_d.sum(axis=0)

