""" Docstring for tests_NBody_Syst

.. autosummary::
    :toctree: _generated/

    test_create_destroy
    test_all_pos_to_segmpos
    test_all_vel_to_segmvel
    test_segmpos_to_all_pos
    test_capture_co
    test_round_trips_pos
    test_round_trips_vel
    test_changevars
    test_params_segmpos_dual
    test_kin
    test_pot
    test_action
    test_resize
    test_pot_indep_resize
    test_repeatability
    test_ForceGeneralSym
    test_ForceGreaterNstore
    test_fft_backends
    test_action_cst_sym_pairs
    test_custom_inter_law
    test_periodicity_default
    test_ODE_vs_spectral

"""

import pytest
from .test_config import *
import numpy as np
import scipy
import choreo

@pytest.mark.parametrize("name", AllConfigNames_list)
def test_create_destroy(name):
    """ This is a docstring for the function test_create_destroy
    
    """

    NBS = load_from_config_file(name)
    
    NBS.nint_fac = 2
    NBS.nint_fac = 3
    NBS.nint_fac = 5
    NBS.nint_fac = 2
    
    del NBS
        
@pytest.mark.parametrize("NBS", [pytest.param(NBS, id=name) for name, NBS in NBS_dict.items()])
def test_all_pos_to_segmpos(NBS, float64_tols):

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
    
    for Sym in NBS.Sym_list:
        assert NBS.ComputeSymDefault(segmpos_cy, Sym) < float64_tols.atol
        
@pytest.mark.parametrize("NBS", [pytest.param(NBS, id=name) for name, NBS in NBS_dict.items()])
def test_all_vel_to_segmvel(NBS, float64_tols):

    NBS.nint_fac = 10
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
    
    print(np.linalg.norm(segmvel_noopt - segmvel_cy))
    assert np.allclose(segmvel_noopt, segmvel_cy, rtol = float64_tols.rtol, atol = float64_tols.atol)  
    
@pytest.mark.parametrize("NBS", [pytest.param(NBS, id=name) for name, NBS in NBS_dict.items()])
def test_segmpos_to_all_pos(NBS, float64_tols):
    
    NBS.nint_fac = 3
    params_buf = np.random.random((NBS.nparams))
    segmpos = NBS.params_to_segmpos(params_buf)

    all_pos = NBS.segmpos_to_all_noopt(segmpos, pos=True)
    NBS.AssertAllSegmGenConstraintsAreRespected(all_pos, pos=True)
    NBS.AssertAllBodyConstraintAreRespected(all_pos, pos=True)
    
    all_coeffs = scipy.fft.rfft(all_pos, axis=1,norm='forward')
    params = NBS.all_coeffs_to_params_noopt(all_coeffs)
    
    print(np.linalg.norm(params_buf-params))
    assert np.allclose(params_buf, params, rtol = float64_tols.rtol, atol = float64_tols.atol)      

@pytest.mark.parametrize("NBS", [pytest.param(NBS, id=name) for name, NBS in NBS_dict.items()])
def test_capture_co(NBS):

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
@pytest.mark.parametrize("NBS", [pytest.param(NBS, id=name) for name, NBS in NBS_dict.items()])
def test_round_trips_pos(NBS, float64_tols):

    NBS.nint_fac = 10
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

    dt = np.random.random()
    all_coeffs = NBS.params_to_all_coeffs_noopt(params_buf, dt=dt)  
    params_buf_rt = NBS.all_coeffs_to_params_noopt(all_coeffs, dt=dt)
    print(np.linalg.norm(params_buf - params_buf_rt))
    assert np.allclose(params_buf, params_buf_rt, rtol = float64_tols.rtol, atol = float64_tols.atol)     
    
    IsReflexionInvariant = False
    for Sym in NBS.Sym_list:
        IsReflexionInvariant = IsReflexionInvariant or (Sym.TimeRev == -1)
    
    if not IsReflexionInvariant:
    
        all_coeffs = NBS.params_to_all_coeffs_noopt(params_buf)  
        dt = np.random.random()
        params_buf_rt = NBS.all_coeffs_to_params_noopt(all_coeffs, dt=dt)
        all_coeffs_rt = NBS.params_to_all_coeffs_noopt(params_buf_rt, dt=dt)  
        print(np.linalg.norm(all_coeffs - all_coeffs_rt))
        assert np.allclose(all_coeffs, all_coeffs_rt, rtol = float64_tols.rtol, atol = float64_tols.atol) 
        
@ProbabilisticTest()
@pytest.mark.parametrize("NBS", [pytest.param(NBS, id=name) for name, NBS in NBS_dict.items()])
def test_round_trips_vel(NBS, float64_tols):

    NBS.nint_fac = 10 # Else it will sometime fail for huge symmetries
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
    
@pytest.mark.parametrize("NBS", [pytest.param(NBS, id=name) for name, NBS in NBS_dict.items()])
def test_changevars(NBS, float64_tols):
    
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
        
@pytest.mark.parametrize("NBS", [pytest.param(NBS, id=name) for name, NBS in NBS_dict.items()])
def test_params_segmpos_dual(NBS, float64_tols):
        
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
    
@pytest.mark.parametrize("NBS", [pytest.param(NBS, id=name) for name, NBS in NBS_dict.items()])
def test_kin(NBS, float64_tols):
        
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

    assert np.allclose(kin_grad_params, kin_grad_params_2, rtol = float64_tols.rtol, atol = float64_tols.atol) 
    
    def grad(x, dx):
        return np.dot(NBS.params_to_kin_nrg_grad(x), dx)
    
    err = compare_FD_and_exact_grad(
        NBS.params_to_kin_nrg   ,
        grad                    ,
        params_buf              ,
        dx=None                 ,
        epslist=None            ,
        order=2                 ,
        vectorize=False         ,
    )
    
    assert err.min() < float64_tols.rtol
        
@ProbabilisticTest(RepeatOnFail=2)
@pytest.mark.parametrize("NBS", [pytest.param(NBS, id=name) for name, NBS in NBS_nozerodiv_dict.items()])
def test_pot(NBS):
    
    NBS.nint_fac = 10
    params_buf = np.random.random((NBS.nparams))
    
    def grad(x,dx):
        return np.dot(NBS.params_to_pot_nrg_grad(x), dx)
    
    dx = np.random.random((NBS.nparams))

    err = compare_FD_and_exact_grad(
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
    
    err = compare_FD_and_exact_grad(
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
        
@ProbabilisticTest(RepeatOnFail=2)
@pytest.mark.parametrize("NBS", [pytest.param(NBS, id=name) for name, NBS in NBS_nozerodiv_dict.items()])
def test_action(NBS):

    NBS.nint_fac = 10
    params_buf = np.random.random((NBS.nparams))
    
    def grad(x,dx):
        return np.dot(NBS.params_to_action_grad(x), dx)
    
    dx = np.random.random((NBS.nparams))

    err = compare_FD_and_exact_grad(
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
    
    err = compare_FD_and_exact_grad(
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
    
@pytest.mark.parametrize("NBS", [pytest.param(NBS, id=name) for name, NBS in NBS_nozerodiv_dict.items()])
def test_resize(NBS, float64_tols):

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

@pytest.mark.parametrize("NBS", [pytest.param(NBS, id=name) for name, NBS in NBS_nozerodiv_dict.items()])
def test_pot_indep_resize(NBS):

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
    
@pytest.mark.parametrize("NBS", [pytest.param(NBS, id=name) for name, NBS in NBS_dict.items()])
def test_repeatability(NBS, float64_tols):

    NBS.nint_fac = 10
    
    params_buf = np.random.random((NBS.nparams))
    params_buf_cp = params_buf.copy()
    segmpos = NBS.params_to_segmpos(params_buf)
    
    print(np.linalg.norm(params_buf - params_buf_cp))
    assert np.allclose(params_buf, params_buf_cp, rtol = float64_tols.rtol, atol = float64_tols.atol) 
    
    segmpos_2 = NBS.params_to_segmpos(params_buf)
    
    print(np.linalg.norm(segmpos - segmpos_2))
    assert np.allclose(segmpos, segmpos_2, rtol = float64_tols.rtol, atol = float64_tols.atol)         
        
@pytest.mark.parametrize("NBS", [pytest.param(NBS, id=name) for name, NBS in NBS_dict.items()])
def test_ForceGeneralSym(NBS, float64_tols):

    NBS.nint_fac = 10
    params_buf = np.random.random((NBS.nparams))
    
    NBS.ForceGeneralSym = False
    segmpos = NBS.params_to_segmpos(params_buf)
    segmvel = NBS.params_to_segmvel(params_buf)
    params = NBS.segmpos_to_params(segmpos)
    params_T = NBS.segmpos_to_params_T(segmpos)
    
    NBS.ForceGeneralSym = True
    segmpos_f = NBS.params_to_segmpos(params_buf)
    segmvel_f = NBS.params_to_segmvel(params_buf)
    params_f = NBS.segmpos_to_params(segmpos)
    params_T_f = NBS.segmpos_to_params_T(segmpos)
    
    print(np.linalg.norm(segmpos - segmpos_f))
    assert np.allclose(segmpos, segmpos_f, rtol = float64_tols.rtol, atol = float64_tols.atol)             
    
    print(np.linalg.norm(segmvel - segmvel_f))
    assert np.allclose(segmvel, segmvel_f, rtol = float64_tols.rtol, atol = float64_tols.atol)    
        
    print(np.linalg.norm(params - params_f))
    assert np.allclose(params, params_f, rtol = float64_tols.rtol, atol = float64_tols.atol)            
        
    print(np.linalg.norm(params_T - params_T_f))
    assert np.allclose(params_T, params_T_f, rtol = float64_tols.rtol, atol = float64_tols.atol)     
    
@ProbabilisticTest(RepeatOnFail=2)
@pytest.mark.parametrize("NBS", [pytest.param(NBS, id=name) for name, NBS in NBS_dict.items()])
def test_ForceGreaterNstore(NBS, float64_tols):

    NBS.ForceGreaterNStore = False

    NBS.nint_fac = 10
    params_buf  = np.random.random((NBS.nparams))
    dparams_buf = np.random.random((NBS.nparams))
    
    segm_store_ini = NBS.segm_store 
    
    segmpos = NBS.params_to_segmpos(params_buf)
    action_grad = NBS.params_to_action_grad(params_buf)
    action_hess = NBS.params_to_action_hess(params_buf, dparams_buf)

    NBS.ForceGreaterNStore = True
    assert NBS.segm_store == NBS.segm_size + 1
    assert NBS.GreaterNStore
    
    segmpos_f = NBS.params_to_segmpos(params_buf)
    action_grad_f = NBS.params_to_action_grad(params_buf)
    action_hess_f = NBS.params_to_action_hess(params_buf, dparams_buf)
    
    for isegm in range(NBS.nsegm):
        print(isegm, np.linalg.norm(segmpos[isegm,:NBS.segm_size,:] - segmpos_f[isegm,:NBS.segm_size,:]))
        assert np.allclose(segmpos[isegm,:NBS.segm_size,:], segmpos_f[isegm,:NBS.segm_size,:], rtol = float64_tols.rtol, atol = float64_tols.atol)  
    
    print(np.linalg.norm(action_grad - action_grad_f))
    assert np.allclose(action_grad, action_grad_f, rtol = float64_tols.rtol, atol = float64_tols.atol)  
    print(np.linalg.norm(action_hess - action_hess_f))
    assert np.allclose(action_hess, action_hess_f, rtol = float64_tols.rtol, atol = float64_tols.atol)  
    
    NBS.ForceGreaterNStore = False
    assert segm_store_ini == NBS.segm_store 
    
    segmpos_nf = NBS.params_to_segmpos(params_buf)
    action_grad_nf = NBS.params_to_action_grad(params_buf)
    action_hess_nf = NBS.params_to_action_hess(params_buf, dparams_buf)
    
    print(np.linalg.norm(segmpos - segmpos_nf))
    assert np.allclose(segmpos, segmpos_nf, rtol = float64_tols.rtol, atol = float64_tols.atol)  
    print(np.linalg.norm(action_grad - action_grad_nf))
    assert np.allclose(action_grad, action_grad_nf, rtol = float64_tols.rtol, atol = float64_tols.atol)  
    print(np.linalg.norm(action_hess - action_hess_nf))
    assert np.allclose(action_hess, action_hess_nf, rtol = float64_tols.rtol, atol = float64_tols.atol)  
    
# @pytest.mark.skip(reason="PYFFTW install currently broken")
@ProbabilisticTest(RepeatOnFail=2)
@pytest.mark.parametrize("NBS", [pytest.param(NBS, id=name) for name, NBS in NBS_dict.items()])
@pytest.mark.parametrize("backend", ["scipy", "mkl", "fftw", "ducc"])
@pytest.mark.parametrize("ForceGeneralSym", [True, False])
def test_fft_backends(float64_tols, ForceGeneralSym, backend, NBS):

    NBS.nint_fac = 10
    
    NBS.fft_backend = "scipy"
    NBS.fftw_planner_effort = 'FFTW_ESTIMATE'
    NBS.fftw_wisdom_only = False
    NBS.fftw_nthreads = 1
    
    params_buf = np.random.random((NBS.nparams))
    segmpos_ref = NBS.params_to_segmpos(params_buf)

    print(f'{backend = }, {ForceGeneralSym = }')

    NBS.ForceGeneralSym = ForceGeneralSym
    NBS.fft_backend = backend
    
    params_buf_cp = params_buf.copy()
    segmpos = NBS.params_to_segmpos(params_buf_cp)
        
    print(np.linalg.norm(segmpos - segmpos_ref))
    assert np.allclose(segmpos, segmpos_ref, rtol = float64_tols.rtol, atol = float64_tols.atol) 
    
    params_buf_rt = NBS.segmpos_to_params(segmpos)

    print(np.linalg.norm(params_buf - params_buf_rt))
    assert np.allclose(params_buf, params_buf_rt, rtol = float64_tols.rtol, atol = float64_tols.atol) 

    NBS.ForceGeneralSym = False
    NBS.fft_backend = "scipy"
    
@ProbabilisticTest(RepeatOnFail=2)
@pytest.mark.parametrize("NBS_pair", [pytest.param(NBS_pair, id=name) for name, NBS_pair in NBS_pairs_dict.items()])
def test_action_cst_sym_pairs(NBS_pair, float64_tols):
    
    NBS_m, NBS_l = NBS_pair
    
    # m => more symmetry. l => less symmetry

    assert NBS_m.nint_min > NBS_l.nint_min
    assert NBS_m.nint_min % NBS_l.nint_min == 0
    
    nint_fac = 10
    NBS_m.nint_fac = nint_fac
    NBS_l.nint = NBS_m.nint
    
    params_buf_m = np.random.random((NBS_m.nparams))        
    all_coeffs = NBS_m.params_to_all_coeffs_noopt(params_buf_m)  
    params_buf_l = NBS_l.all_coeffs_to_params_noopt(all_coeffs)

    kin_m = NBS_m.all_coeffs_to_kin_nrg(all_coeffs)
    kin_l = NBS_l.all_coeffs_to_kin_nrg(all_coeffs)
    
    print(abs(kin_m - kin_l))
    assert abs(kin_m - kin_l) < float64_tols.atol

    kin_m = NBS_m.params_to_kin_nrg(params_buf_m)
    kin_l = NBS_l.params_to_kin_nrg(params_buf_l)
    
    print(abs(kin_m - kin_l))
    assert abs(kin_m - kin_l) < float64_tols.atol

    pot_m = NBS_m.params_to_pot_nrg(params_buf_m)
    pot_l = NBS_l.params_to_pot_nrg(params_buf_l)
    
    print(abs(pot_m - pot_l))
    assert abs(pot_m - pot_l) < float64_tols.atol
    
    act_m = NBS_m.params_to_action(params_buf_m)
    act_l = NBS_l.params_to_action(params_buf_l)
    
    print(abs(act_m - act_l))
    assert abs(act_m - act_l) < float64_tols.atol
    
    segmpos_m = NBS_m.params_to_segmpos(params_buf_m)
    segmvel_m = NBS_m.params_to_segmvel(params_buf_m)
    segmpos_l = NBS_l.params_to_segmpos(params_buf_l)
    segmvel_l = NBS_l.params_to_segmvel(params_buf_l)
    
    loop_len_m, bin_dx_min_m = NBS_m.segm_to_path_stats(segmpos_m, segmvel_m)
    loop_len_l, bin_dx_min_l = NBS_m.segm_to_path_stats(segmpos_l, segmvel_l)
    
    for il in range(NBS_m.nloop):
        
        print(abs(loop_len_m[il] - loop_len_l[il]))
        assert abs(loop_len_m[il] - loop_len_l[il]) < float64_tols.atol     
    
    bin_dx_min_m.sort()
    bin_dx_min_l.sort()
                    
    for ibin in range(NBS_m.nbin_segm_unique):

        print(abs(bin_dx_min_m[ibin] - bin_dx_min_l[ibin]))
        assert abs(bin_dx_min_m[ibin] - bin_dx_min_l[ibin]) < float64_tols.atol        

def test_custom_inter_law(float64_tols):
        
    geodim = 2
    nbody = 3
    bodymass = np.array([1., 2., 3.])
    bodycharge = np.array([4., 5., 6.])
    Sym_list = []
    
    # Classical gravity_pot
    inter_law = scipy.LowLevelCallable.from_cython(choreo.cython._NBodySyst, "gravity_pot")
    NBS = choreo.NBodySyst(geodim, nbody, bodymass, bodycharge, Sym_list, inter_law)
    params_buf = np.random.random((NBS.nparams))   
    action_grad_grav = NBS.params_to_action_grad(params_buf)

    # Parametrized power_law_pot
    inter_law_str = "power_law_pot"
    inter_law_param_dict = {'n':-1., 'alpha':1.}
    NBS = choreo.NBodySyst(geodim, nbody, bodymass, bodycharge, Sym_list, None, inter_law_str, inter_law_param_dict)
    action_grad = NBS.params_to_action_grad(params_buf)
    
    print(np.linalg.norm(action_grad_grav-action_grad))
    assert np.allclose(action_grad_grav, action_grad, rtol = float64_tols.rtol, atol = float64_tols.atol)  
    
    # Python pot_fun
    def inter_law(ptr, xsq, res):
            
        a = xsq ** (-2.5)
        b = xsq*a

        res[0] = -xsq*b
        res[1]= 0.5*b
        res[2] = (-0.75)*a
            
    NBS = choreo.NBodySyst(geodim, nbody, bodymass, bodycharge, Sym_list, inter_law)
    action_grad = NBS.params_to_action_grad(params_buf)
    
    print(np.linalg.norm(action_grad_grav-action_grad))
    assert np.allclose(action_grad_grav, action_grad, rtol = float64_tols.rtol, atol = float64_tols.atol)  
    
    inter_law_str = """
def inter_law(ptr, xsq, res):
        
    a = xsq ** (-2.5)
    b = xsq*a

    res[0] = -xsq*b
    res[1]= 0.5*b
    res[2] = (-0.75)*a
    """

    NBS = choreo.NBodySyst(geodim, nbody, bodymass, bodycharge, Sym_list, None, inter_law_str)
    action_grad = NBS.params_to_action_grad(params_buf)
    
    print(np.linalg.norm(action_grad_grav-action_grad))
    assert np.allclose(action_grad_grav, action_grad, rtol = float64_tols.rtol, atol = float64_tols.atol)  
        
@pytest.mark.parametrize("NBS,params_buf", [pytest.param(NBS, params_buf, id=name) for name, (NBS, params_buf) in Sols_dict.items()])
def test_periodicity_default(float64_tols, NBS, params_buf):
        
    NBS.ForceGreaterNStore = True
    segmpos = NBS.params_to_segmpos(params_buf)

    xo = np.ascontiguousarray(segmpos[:,0 ,:].reshape(-1))
    xf = np.ascontiguousarray(segmpos[:,-1,:].reshape(-1))
    
    dx = NBS.Compute_periodicity_default(xo, xf)
    ndof = dx.shape[0]
    
    assert np.allclose(dx, np.zeros((ndof), dtype=np.float64), rtol = float64_tols.rtol, atol = float64_tols.atol)  
        
@pytest.mark.parametrize("NoSymIfPossible", [True, False])
@pytest.mark.parametrize("LowLevel", [True, False])
@pytest.mark.parametrize("vector_calls", [True, False])
@pytest.mark.parametrize("NBS,params_buf", [pytest.param(NBS, params_buf, id=name) for name, (NBS, params_buf) in Sols_dict.items()])
def test_ODE_vs_spectral(NBS, params_buf, vector_calls, LowLevel, NoSymIfPossible):
        
    NBS.ForceGreaterNStore = True
    segmpos = NBS.params_to_segmpos(params_buf)
    
    action_grad = NBS.segmpos_params_to_action_grad(segmpos, params_buf)
    action_grad_norm = np.linalg.norm(action_grad, ord = np.inf)
    tol = 100 * action_grad_norm
    
    ODE_Syst = NBS.Get_ODE_def(params_buf, vector_calls = vector_calls, LowLevel = LowLevel, NoSymIfPossible = NoSymIfPossible)
    
    nsteps = 10
    keep_freq = 1
    nint_ODE = (NBS.segm_store-1) * keep_freq
    method = "Gauss"
    
    rk = choreo.scipy_plus.multiprec_tables.ComputeImplicitRKTable_Gauss(nsteps, method=method)
    
    segmpos_ODE, segmvel_ODE = choreo.scipy_plus.ODE.SymplecticIVP(
        rk = rk                 ,
        keep_freq = keep_freq   ,
        nint = nint_ODE         ,
        keep_init = True        ,
        **ODE_Syst              ,
    )

    segmpos_ODE = np.ascontiguousarray(segmpos_ODE.reshape((NBS.segm_store, NBS.nsegm, NBS.geodim)).swapaxes(0, 1))

    print(np.linalg.norm(segmpos - segmpos_ODE))
    assert np.allclose(segmpos, segmpos_ODE, rtol = tol, atol = tol)