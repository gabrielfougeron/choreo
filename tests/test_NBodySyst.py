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

def test_all_pos_to_segmpos(AllNBS, float64_tols):
    
    for name, NBS in AllNBS.items():
        
        print(f"Config name : {name}")     

        NBS.nint_fac = 10
        params_buf = np.random.random((NBS.nparams))

        # Unoptimized version
        all_coeffs = NBS.params_to_all_coeffs_noopt(params_buf)        
        all_pos = scipy.fft.irfft(all_coeffs, axis=1)
        
        NBS.AssertAllSegmGenConstraintsAreRespected(all_pos)
        NBS.AssertAllBodyConstraintAreRespected(all_pos)
        
        segmpos_noopt = NBS.all_pos_to_segmpos_noopt(all_pos)
        
        # Optimized version
        segmpos_cy = NBS.params_to_segmpos(params_buf)
        
        assert np.allclose(segmpos_noopt, segmpos_cy, rtol = float64_tols.rtol, atol = float64_tols.atol) 
     
def test_segmpos_to_all_pos(AllNBS, float64_tols):
    
    for name, NBS in AllNBS.items():
        
        print(f"Config name : {name}")     
        
        NBS.nint_fac = 10
        params_buf = np.random.random((NBS.nparams))
        segmpos = NBS.params_to_segmpos(params_buf)


        all_pos = NBS.segmpos_to_all_pos_noopt(segmpos)
        NBS.AssertAllSegmGenConstraintsAreRespected(all_pos)
        NBS.AssertAllBodyConstraintAreRespected(all_pos)
        
        all_coeffs = scipy.fft.rfft(all_pos, axis=1)
        params = NBS.all_coeffs_to_params_noopt(all_coeffs)

        assert np.allclose(params_buf, params, rtol = float64_tols.rtol, atol = float64_tols.atol) 
        
def test_capture_co(AllNBS):
    
    for name, NBS in AllNBS.items():
        
        print(f"Config name : {name}")     
        NBS.nint_fac = 10

        eps = 1e-12
        nnz = [[] for il in range(NBS.nloop)]
        for il in range(NBS.nloop):
            
            nnz_k = NBS.nnz_k(il)
            params_basis = NBS.params_basis(il)
            
            if nnz_k.shape[0] > 0:
                if nnz_k[0] == 0:

                    for iparam in range(params_basis.shape[2]):
                        
                        if np.linalg.norm(params_basis[:,0,iparam].imag) > eps:
                            
                            nnz[il].append(iparam)

        for il in range(NBS.nloop):
            co_in = NBS.co_in(il)
            for iparam in range(co_in.shape[0]):            
                assert not(co_in[iparam]) == (iparam in nnz[il])

@ProbabilisticTest()
def test_round_trips(AllNBS, float64_tols):
    
    for name, NBS in AllNBS.items():
        
        print(f"Config name : {name}")     
        
        NBS.nint_fac = 5 # Else it will sometime fail for huge symmetries
        params_buf = np.random.random((NBS.nparams))
        all_coeffs = NBS.params_to_all_coeffs_noopt(params_buf)  
        all_pos = scipy.fft.irfft(all_coeffs, axis=1)
        segmpos = NBS.all_pos_to_segmpos_noopt(all_pos)
        
        all_pos_rt = NBS.segmpos_to_all_pos_noopt(segmpos)
        print(np.linalg.norm(all_pos_rt - all_pos))
        assert np.allclose(all_pos, all_pos_rt, rtol = float64_tols.rtol, atol = float64_tols.atol) 
                
        all_coeffs_rt = scipy.fft.rfft(all_pos_rt, axis=1)
        print(np.linalg.norm(all_coeffs_rt - all_coeffs))
        assert np.allclose(all_coeffs, all_coeffs_rt, rtol = float64_tols.rtol, atol = float64_tols.atol) 

        params_buf_rt = NBS.all_coeffs_to_params_noopt(all_coeffs_rt)
        print(np.linalg.norm(params_buf - params_buf_rt))
        assert np.allclose(params_buf, params_buf_rt, rtol = float64_tols.rtol, atol = float64_tols.atol) 
        
        segmpos_cy = NBS.params_to_segmpos(params_buf)
        params_buf_rt = NBS.segmpos_to_params(segmpos_cy)
        print(np.linalg.norm(params_buf - params_buf_rt))
        assert np.allclose(params_buf, params_buf_rt, rtol = float64_tols.rtol, atol = float64_tols.atol) 
        
        print()
        
def test_changevars(AllNBS, float64_tols):
    
    for name, NBS in AllNBS.items():
        
        print(f"Config name : {name}")        
        NBS.nint_fac = 1
        
        params_mom_buf = np.random.random((NBS.nparams))
        
        params_pos_buf = NBS.params_changevar(params_mom_buf, inv=False, transpose=False)        
        params_mom_buf_rt = NBS.params_changevar(params_pos_buf, inv=True, transpose=False)        
        print(np.linalg.norm(params_mom_buf - params_mom_buf_rt))
        assert np.allclose(params_mom_buf, params_mom_buf_rt, rtol = float64_tols.rtol, atol = float64_tols.atol)     
        
        params_pos_buf_dual = np.random.random((NBS.nparams_incl_o))
        params_mom_buf_dual = NBS.params_changevar(params_pos_buf_dual, inv=False, transpose=True)
        
        dot_mom = np.dot(params_mom_buf, params_mom_buf_dual)
        dot_pos = np.dot(params_pos_buf, params_pos_buf_dual)
        
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
        
        assert abs(dot_mom - dot_pos) < float64_tols.atol
        assert 2*abs(dot_mom - dot_pos) / (dot_mom + dot_pos) < float64_tols.rtol
    
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