import os
import sys

__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(__PROJECT_ROOT__)

import pytest
from test_config import *

import numpy as np
import scipy
import fractions
import json
import choreo

def test_create_destroy(AllConfigNames):
    
    for name in AllConfigNames:
        
        print(name)
        
        NBS = load_from_config_file(name)
        
        NBS.nint_fac = 2
        NBS.nint_fac = 3
        NBS.nint_fac = 5
        NBS.nint_fac = 2
        
        del NBS
        
        print()
        
def test_all_pos_to_segmpos(AllNBS, float64_tols):
    
    for name, NBS in AllNBS.items():
        
        print(f"Config name : {name}")     

        NBS.nint_fac = 10
        params_buf = np.random.random((NBS.nparams))

        # Unoptimized version
        all_coeffs = NBS.params_to_all_coeffs_noopt(params_buf)        
        all_pos = scipy.fft.irfft(all_coeffs, axis=1, norm='forward')
        
        NBS.AssertAllSegmGenConstraintsAreRespected(all_pos, pos=True)
        NBS.AssertAllBodyConstraintAreRespected(all_pos, pos=True)
        
        segmpos_noopt = NBS.all_to_segm_noopt(all_pos, pos=True)
        
        # Optimized version
        segmpos_cy = NBS.params_to_segmpos(params_buf)
        
        print(np.linalg.norm(segmpos_noopt - segmpos_cy))
        assert np.allclose(segmpos_noopt, segmpos_cy, rtol = float64_tols.rtol, atol = float64_tols.atol)     

def test_all_vel_to_segmvel(AllNBS, float64_tols):
    
    for name, NBS in AllNBS.items():
        
        print(f"Config name : {name}")     

        # NBS.nint_fac = 10
        NBS.nint_fac = 1
        params_buf = np.random.random((NBS.nparams))

        # Unoptimized version
        all_coeffs = NBS.params_to_all_coeffs_noopt(params_buf)        
        NBS.all_coeffs_pos_to_vel_inplace(all_coeffs)
        all_vel = scipy.fft.irfft(all_coeffs, axis=1, norm='forward')
        
        NBS.AssertAllSegmGenConstraintsAreRespected(all_vel, pos=False)
        NBS.AssertAllBodyConstraintAreRespected(all_vel, pos=False)
        
        segmvel_noopt = NBS.all_to_segm_noopt(all_vel, pos=False)
        
        # Optimized version
        segmvel_cy = NBS.params_to_segmvel(params_buf)
        
        for isegm in range(NBS.nsegm):
            print(isegm)
            print(segmvel_noopt[isegm,:,:] - segmvel_cy[isegm,:,:])
        
        print(np.linalg.norm(segmvel_noopt - segmvel_cy))
        assert np.allclose(segmvel_noopt, segmvel_cy, rtol = float64_tols.rtol, atol = float64_tols.atol)  
     
def test_segmpos_to_all_pos(AllNBS, float64_tols):
    
    for name, NBS in AllNBS.items():
        
        print(f"Config name : {name}")     
        
        NBS.nint_fac = 10
        params_buf = np.random.random((NBS.nparams))
        segmpos = NBS.params_to_segmpos(params_buf)

        all_pos = NBS.segmpos_to_all_noopt(segmpos, pos=True)
        NBS.AssertAllSegmGenConstraintsAreRespected(all_pos, pos=True)
        NBS.AssertAllBodyConstraintAreRespected(all_pos, pos=True)
        
        all_coeffs = scipy.fft.rfft(all_pos, axis=1,norm='forward')
        params = NBS.all_coeffs_to_params_noopt(all_coeffs)
        
        print(np.linalg.norm(params_buf-params))
        assert np.allclose(params_buf, params, rtol = float64_tols.rtol, atol = float64_tols.atol) 
        
def test_capture_co(AllNBS):
    
    for name, NBS in AllNBS.items():
        
        print(f"Config name : {name}")     
        NBS.nint_fac = 10

        eps = 1e-12
        nnz = [[] for il in range(NBS.nloop)]
        for il in range(NBS.nloop):
            
            nnz_k = NBS.nnz_k(il)
            params_basis_pos = NBS.params_basis_pos(il)
            
            if nnz_k.shape[0] > 0:
                if nnz_k[0] == 0:

                    for iparam in range(params_basis_pos.shape[2]):
                        
                        if np.linalg.norm(params_basis_pos[:,0,iparam].imag) > eps:
                            
                            nnz[il].append(iparam)

        for il in range(NBS.nloop):
            co_in = NBS.co_in(il)
            for iparam in range(co_in.shape[0]):            
                assert not(co_in[iparam]) == (iparam in nnz[il])

@ProbabilisticTest()
def test_round_trips_pos(AllNBS, float64_tols):
    
    for name, NBS in AllNBS.items():
        
        print(f"Config name : {name}")     
        
        NBS.nint_fac = 5 # Else it will sometime fail for huge symmetries
        params_buf = np.random.random((NBS.nparams))
        all_coeffs = NBS.params_to_all_coeffs_noopt(params_buf)  
        all_pos = scipy.fft.irfft(all_coeffs, axis=1, norm='forward')
        segmpos = NBS.all_to_segm_noopt(all_pos, pos=True)
        
        all_pos_rt = NBS.segmpos_to_all_noopt(segmpos, pos=True)
        print(np.linalg.norm(all_pos_rt - all_pos))
        assert np.allclose(all_pos, all_pos_rt, rtol = float64_tols.rtol, atol = float64_tols.atol) 
                
        all_coeffs_rt = scipy.fft.rfft(all_pos, axis=1,norm='forward')
        print(np.linalg.norm(all_coeffs_rt - all_coeffs))
        assert np.allclose(all_coeffs, all_coeffs_rt, rtol = float64_tols.rtol, atol = float64_tols.atol) 

        params_buf_rt = NBS.all_coeffs_to_params_noopt(all_coeffs)
        print(np.linalg.norm(params_buf - params_buf_rt))
        assert np.allclose(params_buf, params_buf_rt, rtol = float64_tols.rtol, atol = float64_tols.atol) 
        
        segmpos_cy = NBS.params_to_segmpos(params_buf)
        params_buf_rt = NBS.segmpos_to_params(segmpos_cy)
        print(np.linalg.norm(params_buf - params_buf_rt))
        assert np.allclose(params_buf, params_buf_rt, rtol = float64_tols.rtol, atol = float64_tols.atol) 
        
        print()
        
@ProbabilisticTest()
def test_round_trips_vel(AllNBS, float64_tols):
    
    for name, NBS in AllNBS.items():
        
        print(f"Config name : {name}")     
        
        NBS.nint_fac = 5 # Else it will sometime fail for huge symmetries
        params_buf = np.random.random((NBS.nparams))
        all_coeffs = NBS.params_to_all_coeffs_noopt(params_buf)  
        NBS.all_coeffs_pos_to_vel_inplace(all_coeffs)
        all_vel = scipy.fft.irfft(all_coeffs, axis=1, norm='forward')
        segmvel = NBS.all_to_segm_noopt(all_vel, pos=False)
        
        all_vel_rt = NBS.segmpos_to_all_noopt(segmvel, pos=False)
        print(np.linalg.norm(all_vel_rt - all_vel))
        assert np.allclose(all_vel, all_vel_rt, rtol = float64_tols.rtol, atol = float64_tols.atol) 
                
        all_coeffs_rt = scipy.fft.rfft(all_vel, axis=1,norm='forward')
        print(np.linalg.norm(all_coeffs_rt - all_coeffs))
        assert np.allclose(all_coeffs, all_coeffs_rt, rtol = float64_tols.rtol, atol = float64_tols.atol) 
        
        # segmvel_cy = NBS.params_to_segmvel(params_buf)
        # params_buf_rt = NBS.segmvel_to_params(segmvel_cy)
        # print(np.linalg.norm(params_buf - params_buf_rt))
        # assert np.allclose(params_buf, params_buf_rt, rtol = float64_tols.rtol, atol = float64_tols.atol) 
        
        print()
        
def test_changevars(AllNBS, float64_tols):
    
    for name, NBS in AllNBS.items():
        
        print(f"Config name : {name}")        
        NBS.nint_fac = 10
        
        params_mom_buf = np.random.random((NBS.nparams))
        
        params_pos_buf = NBS.params_changevar(params_mom_buf, inv=False, transpose=False)        
        params_mom_buf_rt = NBS.params_changevar(params_pos_buf, inv=True, transpose=False)        

        print(np.linalg.norm(params_mom_buf - params_mom_buf_rt))
        assert np.allclose(params_mom_buf, params_mom_buf_rt, rtol = float64_tols.rtol, atol = float64_tols.atol)     
        
        params_pos_buf_dual = np.random.random((NBS.nparams_incl_o))
        params_mom_buf_dual = NBS.params_changevar(params_pos_buf_dual, inv=False, transpose=True)
        
        dot_mom = np.dot(params_mom_buf, params_mom_buf_dual)
        dot_pos = np.dot(params_pos_buf, params_pos_buf_dual)
        
        print(abs(dot_mom - dot_pos))
        assert abs(dot_mom - dot_pos) < float64_tols.atol
        assert 2*abs(dot_mom - dot_pos) / (dot_mom + dot_pos) < float64_tols.rtol        
        
        # ##################################################################################################
        
        params_mom_buf = NBS.params_changevar(params_pos_buf, inv=True, transpose=False)        
        params_pos_buf_rt = NBS.params_changevar(params_mom_buf, inv=False, transpose=False)   
        print(np.linalg.norm(params_pos_buf - params_pos_buf_rt))
        assert np.allclose(params_pos_buf, params_pos_buf_rt, rtol = float64_tols.rtol, atol = float64_tols.atol)     
        
        params_mom_buf_dual = np.random.random((NBS.nparams))
        params_pos_buf_dual = NBS.params_changevar(params_mom_buf_dual, inv=True, transpose=True)
        
        dot_mom = np.dot(params_mom_buf, params_mom_buf_dual)
        dot_pos = np.dot(params_pos_buf, params_pos_buf_dual)
        
        print(abs(dot_mom - dot_pos))
        assert abs(dot_mom - dot_pos) < float64_tols.atol
        assert 2*abs(dot_mom - dot_pos) / (dot_mom + dot_pos) < float64_tols.rtol        
        
        print()
        
def test_params_segmpos_dual(AllNBS, float64_tols):
    
    for name, NBS in AllNBS.items():
        
        print(f"Config name : {name}")        
        NBS.nint_fac = 10
        
        params_buf = np.random.random((NBS.nparams))
        segmpos = NBS.params_to_segmpos(params_buf)
        
        segmpos_dual = np.random.random((NBS.nsegm,NBS.segm_store,NBS.geodim))
        params_buf_dual = NBS.segmpos_to_params_T(segmpos_dual)
        
        dot_params = np.dot(params_buf, params_buf_dual)
        dot_segmpos = np.dot(segmpos_dual.reshape(-1), segmpos.reshape(-1))
        
        print(abs(dot_params - dot_segmpos))
        assert abs(dot_params - dot_segmpos) < float64_tols.atol
        assert 2*abs(dot_params - dot_segmpos) / (dot_params + dot_segmpos) < float64_tols.rtol
        print()
    
def test_kin(AllNBS, float64_tols):
    
    for name, NBS in AllNBS.items():
        
        print(f"Config name : {name}")        
        NBS.nint_fac = 10
        params_buf = np.random.random((NBS.nparams))
        
        all_coeffs = NBS.params_to_all_coeffs_noopt(params_buf) 
        kin = NBS.all_coeffs_to_kin_nrg(all_coeffs)
        kin_opt = NBS.params_to_kin_nrg(params_buf)

        assert abs(kin - kin_opt) < float64_tols.atol
        assert 2*abs(kin - kin_opt) / (kin + kin_opt) < float64_tols.rtol
        
        kin_grad_params = NBS.params_to_kin_nrg_grad(params_buf)
        
        kin_grad_coeffs = NBS.all_coeffs_to_kin_nrg_grad(all_coeffs)
        kin_grad_params_2 = NBS.all_coeffs_to_params_noopt(kin_grad_coeffs, transpose=True)
        
        print(np.linalg.norm(kin_grad_params - kin_grad_params_2))
        assert np.allclose(kin_grad_params, kin_grad_params_2, rtol = float64_tols.rtol, atol = float64_tols.atol) 
        
        def grad(x, dx):
            return np.dot(NBS.params_to_kin_nrg_grad(x), dx)
        
        err = choreo.scipy_plus.test.compare_FD_and_exact_grad(
            NBS.params_to_kin_nrg   ,
            grad                    ,
            params_buf              ,
            dx=None                 ,
            epslist=None            ,
            order=2                 ,
            vectorize=False         ,
        )
        
        assert err.min() < float64_tols.rtol
        print()
        
@ProbabilisticTest(RepeatOnFail=2)
def test_pot(AllNBS):
    
    for name, NBS in AllNBS.items():
        
        print(f"Config name : {name}")        
        NBS.nint_fac = 10
        params_buf = np.random.random((NBS.nparams))
        
        def grad(x,dx):
            return np.dot(NBS.params_to_pot_nrg_grad(x), dx)
        
        dx = np.random.random((NBS.nparams))

        err = choreo.scipy_plus.test.compare_FD_and_exact_grad(
            NBS.params_to_pot_nrg   ,
            grad                    ,
            params_buf              ,
            dx=dx                   ,
            epslist=None            ,
            order=2                 ,
            vectorize=False         ,
        )

        print(err.min())
        assert (err.min() <  1e-7)
        
        err = choreo.scipy_plus.test.compare_FD_and_exact_grad(
            NBS.params_to_pot_nrg_grad  ,
            NBS.params_to_pot_nrg_hess  ,
            params_buf                  ,
            dx=dx                       ,
            epslist=None                ,
            order=2                     ,
            vectorize=False             ,
        )

        print(err.min())
        assert (err.min() <  1e-7)
        print()
        
@ProbabilisticTest(RepeatOnFail=2)
def test_action(AllNBS):
    
    for name, NBS in AllNBS.items():
        
        print(f"Config name : {name}")        
        NBS.nint_fac = 10
        params_buf = np.random.random((NBS.nparams))
        
        def grad(x,dx):
            return np.dot(NBS.params_to_action_grad(x), dx)
        
        dx = np.random.random((NBS.nparams))

        err = choreo.scipy_plus.test.compare_FD_and_exact_grad(
            NBS.params_to_action    ,
            grad                    ,
            params_buf              ,
            dx=dx                   ,
            epslist=None            ,
            order=2                 ,
            vectorize=False         ,
        )

        print(err.min())
        assert (err.min() <  1e-7)
        
        err = choreo.scipy_plus.test.compare_FD_and_exact_grad(
            NBS.params_to_action_grad   ,
            NBS.params_to_action_hess   ,
            params_buf                  ,
            dx=dx                       ,
            epslist=None                ,
            order=2                     ,
            vectorize=False             ,
        )

        print(err.min())
        assert (err.min() <  1e-7)
        
        print()
    
def test_resize(AllNBS, float64_tols):
    
    for name, NBS in AllNBS.items():
        
        print(f"Config name : {name}")        
        NBS.nint_fac = 10
        small_segm_size = NBS.segm_size
        params_buf = np.random.random((NBS.nparams))
        segmpos = NBS.params_to_segmpos(params_buf)
        
        all_coeffs = NBS.params_to_all_coeffs_noopt(params_buf)  
        all_pos = scipy.fft.irfft(all_coeffs, axis=1, norm='forward')
        segmpos_noopt = NBS.all_to_segm_noopt(all_pos, pos=True)
        
        print(np.linalg.norm(segmpos - segmpos_noopt))
        assert np.allclose(segmpos, segmpos_noopt, rtol = float64_tols.rtol, atol = float64_tols.atol) 
        
        fac = 4
        new_nint_fac = fac * NBS.nint_fac
        
        params_buf_long = NBS.params_resize(params_buf, new_nint_fac) 
        NBS.nint_fac = new_nint_fac
        long_segm_size = NBS.segm_size
        segmpos_long = NBS.params_to_segmpos(params_buf_long)
        
        all_coeffs = NBS.params_to_all_coeffs_noopt(params_buf_long)  
        all_pos = scipy.fft.irfft(all_coeffs, axis=1, norm='forward')
        segmpos_long_noopt = NBS.all_to_segm_noopt(all_pos, pos=True)
        
        print(np.linalg.norm(segmpos_long - segmpos_long_noopt))
        assert np.allclose(segmpos_long, segmpos_long_noopt, rtol = float64_tols.rtol, atol = float64_tols.atol) 
        
        print(np.linalg.norm(segmpos[:,:small_segm_size,:] - segmpos_long[:,:long_segm_size:fac,:]))
        assert np.allclose(segmpos[:,:small_segm_size,:], segmpos_long[:,:long_segm_size:fac,:], rtol = float64_tols.rtol, atol = float64_tols.atol) 
        print()
        
@ProbabilisticTest()
def test_pot_indep_resize(AllNBS):
    
    for name, NBS in AllNBS.items():
        
        print(f"Config name : {name}")   
        
        Passed_any = False
        
        ntries = 100
        
        err = np.zeros((ntries,4))
        
        for itry in range(ntries): 
            
            Passed = True
            
            nint_fac_short = 5
            nint_fac_mid = 200
            nint_fac_big = nint_fac_mid*2
            
            NBS.nint_fac = nint_fac_short
            params_buf_short = np.random.random((NBS.nparams))
            
            params_buf_mid = NBS.params_resize(params_buf_short, nint_fac_mid) 
            params_buf_big = NBS.params_resize(params_buf_short, nint_fac_big) 

            NBS.nint_fac = nint_fac_mid
            kin_nrg = NBS.params_to_kin_nrg(params_buf_mid)
            pot_nrg = NBS.params_to_pot_nrg(params_buf_mid)
            
            NBS.nint_fac = nint_fac_big
            kin_nrg_big= NBS.params_to_kin_nrg(params_buf_big)
            pot_nrg_big= NBS.params_to_pot_nrg(params_buf_big)
            
            err[itry,0] = abs(kin_nrg - kin_nrg_big)
            err[itry,1] = 2*abs(kin_nrg - kin_nrg_big) / abs(kin_nrg + kin_nrg_big)
            err[itry,2] = abs(pot_nrg - pot_nrg_big)
            err[itry,3] = 2*abs(pot_nrg - pot_nrg_big) / abs(pot_nrg + pot_nrg_big) 

            Passed = Passed and err[itry,0] < 1e-7
            Passed = Passed and err[itry,1] < 1e-7    
            
            Passed = Passed and err[itry,2] < 1e-7
            Passed = Passed and err[itry,3] < 1e-7
            
            Passed_any = Passed_any or Passed
            
            if Passed_any:
                break
            
        if not(Passed_any):
            print(err)
        
        assert Passed_any
        
def test_repeatability(AllNBS, float64_tols):
    
    for name, NBS in AllNBS.items():
        
        print(f"Config name : {name}")   
        
        params_buf = np.random.random((NBS.nparams))
        params_buf_cp = params_buf.copy()
        segmpos = NBS.params_to_segmpos(params_buf)
        
        print(np.linalg.norm(params_buf - params_buf_cp))
        assert np.allclose(params_buf, params_buf_cp, rtol = float64_tols.rtol, atol = float64_tols.atol) 
        
        segmpos_2 = NBS.params_to_segmpos(params_buf)
        
        print(np.linalg.norm(segmpos - segmpos_2))
        assert np.allclose(segmpos, segmpos_2, rtol = float64_tols.rtol, atol = float64_tols.atol) 

def test_fft_backends(AllNBS, float64_tols):
    
    for name, NBS in AllNBS.items():
        
        print(f"Config name : {name}")   
        
        NBS.fftw_planner_effort = 'FFTW_MEASURE'
        NBS.fftw_wisdom_only = False
        NBS.fftw_nthreads = 1
        
        params_buf = np.random.random((NBS.nparams))
        params_buf_cp = params_buf.copy()
        segmpos_ref = NBS.params_to_segmpos(params_buf)
        
        for backend in ["scipy", "mkl", "fftw"]:
            
            print(backend)
            
            NBS.fft_backend = backend
            
            segmpos = NBS.params_to_segmpos(params_buf_cp)
                
            print(np.linalg.norm(segmpos - segmpos_ref))
            assert np.allclose(segmpos, segmpos_ref, rtol = float64_tols.rtol, atol = float64_tols.atol) 
            
            params_buf_rt = NBS.segmpos_to_params(segmpos)

            print(np.linalg.norm(params_buf - params_buf_rt))
            assert np.allclose(params_buf, params_buf_rt, rtol = float64_tols.rtol, atol = float64_tols.atol) 

        print()