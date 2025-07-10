""" Tests properties of :class:`choreo.NBodySyst`.

.. autosummary::
    :toctree: _generated/
    :template: tests-formatting/base.rst
    :nosignatures:

    test_bad_init_args
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
    test_action_indep_resize
    test_repeatability
    test_ForceGeneralSym
    test_ForceGreaterNstore
    test_fft_backends
    test_action_cst_sym_pairs
    test_custom_inter_law
    test_periodicity_default
    test_RK_vs_spectral
    test_RK_vs_spectral_reset
    test_segmpos_param
    test_grad_fun_FD
    test_ODE_grad_vs_FD_Implicit
    test_ODE_grad_vs_FD_Explicit
    test_remove_all_syms_nrg
    test_remove_all_syms_ODE
    test_ODE_grad_period_noopt
    test_ODE_grad_period_opt
    test_Kepler
    test_RK_vs_spectral_periodicity_default
    test_center_of_mass
    test_perdef
    test_initposmom
    test_params_to_periodicity_default_grad_vs_FD

"""

import pytest
from .test_config import *
import numpy as np
import scipy
import choreo

@ParametrizeDocstrings
def test_bad_init_args():
    """ Tests that mismatched init arguments are properly handled during initialization of a :class:`choreo.NBodySyst`.
    """
    
    NBS = choreo.NBodySyst()
    
    geodim = 3
    nbody = 3
    NBS = choreo.NBodySyst(geodim, nbody)
    
    with pytest.raises(ValueError):    
        NBS = choreo.NBodySyst(-1, nbody)
    with pytest.raises(ValueError):    
        NBS = choreo.NBodySyst(geodim, -1)
    
@ParametrizeDocstrings
@pytest.mark.parametrize("name", AllConfigNames_list)
def test_create_destroy(name):
    """ Tests initialization and destruction of NBodySyst.
    """

    NBS = load_from_config_file(name)
    
    NBS.nint_fac = 2
    NBS.nint_fac = 3
    NBS.nint_fac = 5
    NBS.nint_fac = 2
    
    del NBS

@ParametrizeDocstrings        
@pytest.mark.parametrize("NBS", [pytest.param(NBS, id=name) for name, NBS in NBS_dict.items()])
def test_all_pos_to_segmpos(NBS, float64_tols):
    """ Tests whether ``all_pos`` and ``segmpos`` agree.

Tests:   
 
* Whether the unoptimized track ``params_buf`` => ``all_pos`` => ``segmpos_noopt`` and the optimized track ``params_buf`` => ``segmpos_cy`` agree.
* That ``all_pos`` and ``segmpos`` respects all symmetry constraints.

    """

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
        
@ParametrizeDocstrings
@pytest.mark.parametrize("NBS", [pytest.param(NBS, id=name) for name, NBS in NBS_dict.items()])
def test_all_vel_to_segmvel(NBS, float64_tols):
    """ Tests whether ``all_vel`` and ``segmvel`` agree.

Tests:  
  
* Whether the unoptimized track ``params_buf`` => ``all_vel`` => ``segmvel_noopt`` and the optimized track ``params_buf`` => ``segmvel_cy`` agree.
* That ``all_vel`` and ``segmvel`` respects all symmetry constraints.

    """

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
    
    for Sym in NBS.Sym_list:
        assert NBS.ComputeSymDefault(segmvel_cy, Sym.TimeDerivative(), pos = False) < float64_tols.atol
    
@ParametrizeDocstrings
@pytest.mark.parametrize("NBS", [pytest.param(NBS, id=name) for name, NBS in NBS_dict.items()])
def test_segmpos_to_all_pos(NBS, float64_tols):
    """ Tests going from ``segmpos`` to ``all_pos`` and back.

Tests:  
  
* Whether the track ``params_buf`` => ``segmpos`` => ``all_pos`` => ``all_coeffs`` => ``params_buf`` is the identity.

    """
    
    NBS.nint_fac = 10
    params_buf = np.random.random((NBS.nparams))
    segmpos = NBS.params_to_segmpos(params_buf)
    all_pos = NBS.segmpos_to_all_noopt(segmpos, pos=True)
    all_coeffs = scipy.fft.rfft(all_pos, axis=1, norm='forward')
    params = NBS.all_coeffs_to_params_noopt(all_coeffs)
    
    print(np.linalg.norm(params_buf-params))
    assert np.allclose(params_buf, params, rtol = float64_tols.rtol, atol = float64_tols.atol)      

@ParametrizeDocstrings
@pytest.mark.parametrize("NBS", [pytest.param(NBS, id=name) for name, NBS in NBS_dict.items()])
def test_capture_co(float64_tols, NBS):
    """ Checks consistency of parameters definition for :math:`c_o`.

Tests whether the parameter corresponding to the imaginary part of the coefficient of index 0 in the Fourier expansion is included only if it needs to be.
    """

    NBS.nint_fac = 10

    nnz = [[] for il in range(NBS.nloop)]
    for il in range(NBS.nloop):
        
        nnz_k = NBS.nnz_k(il)
        params_basis_pos = NBS.params_basis_pos(il)
        
        if nnz_k.shape[0] > 0:
            if nnz_k[0] == 0:

                for iparam in range(params_basis_pos.shape[2]):
                    
                    if np.linalg.norm(params_basis_pos[:,0,iparam].imag) > float64_tols.atol:
                        
                        nnz[il].append(iparam)

    for il in range(NBS.nloop):
        co_in = NBS.co_in(il)
        for iparam in range(co_in.shape[0]):            
            assert not(co_in[iparam]) == (iparam in nnz[il])

@ParametrizeDocstrings
@ProbabilisticTest()
@pytest.mark.parametrize("NBS", [pytest.param(NBS, id=name) for name, NBS in NBS_dict.items()])
def test_round_trips_pos(float64_tols, NBS):
    """ Tests whether several ways of going back and forth in the chain ``param_buf`` => ``segmpos`` indeed give the same result.

Tests:  
  
* That ``all_pos`` => ``segmpos`` => ``all_pos`` is the identity.
* That ``all_coeffs`` => ``all_pos`` => ``all_coeffs`` is the identity.
* That ``params_buf`` => ``all_coeffs`` => ``params_buf`` is the identity.
* That ``params_buf`` => shifted ``all_coeffs`` => ``params_buf`` is the identity.
* That ``all_coeffs`` => shifted ``params_buf`` => ``all_coeffs`` is the identity in cases where it makes sense (i.e. no invariance wrt space orientation reversing transformations).

    """

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
        
@ParametrizeDocstrings
@ProbabilisticTest()
@pytest.mark.parametrize("NBS", [pytest.param(NBS, id=name) for name, NBS in NBS_dict.items()])
def test_round_trips_vel(float64_tols, NBS):
    """ Tests whether several ways of going back and forth in the chain ``param_buf`` => ``segmvel`` indeed give the same result.

Tests:  
  
* That ``all_vel`` => ``segmvel`` => ``all_vel`` is the identity.
* That ``all_coeffs`` => ``all_vel`` => ``all_coeffs`` is the identity.
    """

    NBS.nint_fac = 10 # Else it will sometime fail for huge symmetries
    params_buf = np.random.random((NBS.nparams))
    all_coeffs = NBS.params_to_all_coeffs_noopt(params_buf)  
    NBS.all_coeffs_pos_to_vel_inplace(all_coeffs)
    all_vel = scipy.fft.irfft(all_coeffs, axis=1, norm='forward')
    segmvel = NBS.all_to_segm_noopt(all_vel, pos=False)
    
    all_vel_rt = NBS.segmpos_to_all_noopt(segmvel, pos=False)
    print(np.linalg.norm(all_vel_rt - all_vel))
    assert np.allclose(all_vel, all_vel_rt, rtol = float64_tols.rtol, atol = float64_tols.atol) 
            
    all_coeffs_rt = scipy.fft.rfft(all_vel, axis=1, norm='forward')
    print(np.linalg.norm(all_coeffs_rt - all_coeffs))
    assert np.allclose(all_coeffs, all_coeffs_rt, rtol = float64_tols.rtol, atol = float64_tols.atol) 
    
@ParametrizeDocstrings
@pytest.mark.parametrize("NBS", [pytest.param(NBS, id=name) for name, NBS in NBS_dict.items()])
def test_changevars(float64_tols, NBS):
    """ Tests properties of change of variables of parameters.

Tests:  
  
* That ``params_mom_buf`` => ``params_pos_buf`` => ``params_mom_buf`` is the identity.
* That ``params_pos_buf`` => ``params_mom_buf`` => ``params_pos_buf`` is the identity.
* That scalar poducts are conserved by the (dual) change of variables.
    """
    
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
        
@ParametrizeDocstrings
@pytest.mark.parametrize("NBS", [pytest.param(NBS, id=name) for name, NBS in NBS_dict.items()])
def test_params_segmpos_dual(float64_tols, NBS):
    """ Tests invariance of dot product  by the transformation ``params`` => ``segmpos``.
    """
        
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
    
@ParametrizeDocstrings
@pytest.mark.parametrize("NBS", [pytest.param(NBS, id=name) for name, NBS in NBS_dict.items()])
def test_kin(float64_tols, NBS):
    """ Tests computation of kinetic energy.
    
Tests:  
  
* That optimized and non optimized computations of kinetic energy give the same result.
* Idem for the gradient of the kinetic energy.
* That the gradient of the kinetic energy agrees with its finite difference approximation.
    
    """
        
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
        order=2                 ,
        vectorize=False         ,
    )
    
    assert err.min() < float64_tols.rtol
        
@ParametrizeDocstrings
@ProbabilisticTest(RepeatOnFail=2)
@pytest.mark.parametrize("NBS", [pytest.param(NBS, id=name) for name, NBS in NBS_nozerodiv_dict.items()])
def test_pot(float64_tols_loose, NBS):
    """ Tests computation of potential energy.
    
Tests:  
  
* That the gradient of the potential energy agrees with its finite difference approximation.
* That the hessian of the potential energy agrees with its finite difference approximation.
    
    """
    
    NBS.nint_fac = 10
    params_buf = np.random.random((NBS.nparams))
    
    def grad(x,dx):
        return np.dot(NBS.params_to_pot_nrg_grad(x), dx)

    err = compare_FD_and_exact_grad(
        NBS.params_to_pot_nrg   ,
        grad                    ,
        params_buf              ,
        order=2                 ,
        vectorize=False         ,
    )

    print(err.min())
    assert (err.min() < float64_tols_loose.rtol)
    
    err = compare_FD_and_exact_grad(
        NBS.params_to_pot_nrg_grad  ,
        NBS.params_to_pot_nrg_hess  ,
        params_buf                  ,
        order=2                     ,
        vectorize=False             ,
    )

    print(err.min())
    assert (err.min() < float64_tols_loose.rtol)
        
@ParametrizeDocstrings
@ProbabilisticTest(RepeatOnFail=2)
@pytest.mark.parametrize("NBS", [pytest.param(NBS, id=name) for name, NBS in NBS_nozerodiv_dict.items()])
def test_action(float64_tols_loose, NBS):
    """ Tests computation of the action.
    
Tests:  
  
* That the gradient of the action agrees with its finite difference approximation.
* That the hessian of the action agrees with its finite difference approximation.
    """

    NBS.nint_fac = 10
    params_buf = np.random.random((NBS.nparams))
    
    def grad(x,dx):
        return np.dot(NBS.params_to_action_grad(x), dx)
    
    dx = np.random.random((NBS.nparams))

    err = compare_FD_and_exact_grad(
        NBS.params_to_action    ,
        grad                    ,
        params_buf              ,
        order=2                 ,
        vectorize=False         ,
    )

    print(err.min())
    assert (err.min() < float64_tols_loose.rtol)
    
    err = compare_FD_and_exact_grad(
        NBS.params_to_action_grad   ,
        NBS.params_to_action_hess   ,
        params_buf                  ,
        order=2                     ,
        vectorize=False             ,
    )

    print(err.min())
    assert (err.min() < float64_tols_loose.rtol)
    
@ParametrizeDocstrings
@pytest.mark.parametrize("NBS", [pytest.param(NBS, id=name) for name, NBS in NBS_nozerodiv_dict.items()])
def test_resize(float64_tols, NBS):
    """ Tests nested properties of Fourier spaces on positions.
    """

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

@ParametrizeDocstrings
@RetryTest(n=100)
@pytest.mark.parametrize("NBS", [pytest.param(NBS, id=name) for name, NBS in NBS_nozerodiv_dict.items()])
def test_action_indep_resize(float64_tols_loose, NBS):
    """ Tests that action is left **similar** by resizing.

.. note::
    This test is very susceptible to false positives. It should be replaced by a better test.
    
    """

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
    
    assert abs(kin_nrg - kin_nrg_big) < float64_tols_loose.rtol
    assert 2*abs(kin_nrg - kin_nrg_big) / abs(kin_nrg + kin_nrg_big) < float64_tols_loose.rtol
    assert abs(pot_nrg - pot_nrg_big) < float64_tols_loose.rtol
    assert 2*abs(pot_nrg - pot_nrg_big) / abs(pot_nrg + pot_nrg_big) < float64_tols_loose.rtol

@ParametrizeDocstrings
@pytest.mark.parametrize("NBS", [pytest.param(NBS, id=name) for name, NBS in NBS_dict.items()])
def test_repeatability(float64_tols, NBS):
    """ Tests that computing ``params_buf`` => ``segmpos`` twice give the same result.
    """

    NBS.nint_fac = 10
    
    params_buf = np.random.random((NBS.nparams))
    params_buf_cp = params_buf.copy()
    
    print(np.linalg.norm(params_buf - params_buf_cp))
    assert np.allclose(params_buf, params_buf_cp, rtol = float64_tols.rtol, atol = float64_tols.atol) 
    
    segmpos = NBS.params_to_segmpos(params_buf)
    segmpos_2 = NBS.params_to_segmpos(params_buf)
    
    print(np.linalg.norm(segmpos - segmpos_2))
    assert np.allclose(segmpos, segmpos_2, rtol = float64_tols.rtol, atol = float64_tols.atol)         
        
@ParametrizeDocstrings
@pytest.mark.parametrize("NBS", [pytest.param(NBS, id=name) for name, NBS in NBS_dict.items()])
def test_ForceGeneralSym(float64_tols, NBS):
    """ Tests that computations results are independent of :meth:`choreo.NBodySyst.ForceGeneralSym`.
    
Tests that the following computations are independent of :meth:`choreo.NBodySyst.ForceGeneralSym`:

* ``params`` => ``segmpos``
* ``params`` => ``segmvel``
* ``segmpos`` => ``params``
* ``segmpos`` => ``params_T``

    """

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
    
@ParametrizeDocstrings
@ProbabilisticTest(RepeatOnFail=2)
@pytest.mark.parametrize("NBS", [pytest.param(NBS, id=name) for name, NBS in NBS_dict.items()])
def test_ForceGreaterNstore(float64_tols, NBS):
    """ Tests that computations results are independent of :meth:`choreo.NBodySyst.ForceGreaterNStore`.
    
Tests that the following computations are independent of :meth:`choreo.NBodySyst.ForceGreaterNStore`:

* ``params`` => ``segmpos``
* ``params`` => ``action_grad``
* ``params`` => ``action_hess``
    """

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
    
@ParametrizeDocstrings
@ProbabilisticTest(RepeatOnFail=2)
@pytest.mark.parametrize("NBS", [pytest.param(NBS, id=name) for name, NBS in NBS_dict.items()])
@pytest.mark.parametrize("backend", ["scipy", "mkl", "fftw", "ducc"])
@pytest.mark.parametrize("ForceGeneralSym", [True, False])
def test_fft_backends(float64_tols, ForceGeneralSym, backend, NBS):
    """ Tests that computations results are independent of :meth:`choreo.NBodySyst.fft_backend`.
    
Tests that the following computations are independent of :meth:`choreo.NBodySyst.fft_backend`:

* ``params`` => ``segmpos``
* ``segmpos`` => ``params``
    """

    NBS.nint_fac = 10
    
    NBS.fft_backend = "scipy"
    NBS.fftw_planner_effort = 'FFTW_ESTIMATE'
    NBS.fftw_wisdom_only = False
    NBS.fftw_nthreads = 1
    
    params_buf = np.random.random((NBS.nparams))
    segmpos_ref = NBS.params_to_segmpos(params_buf)

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
    
@ParametrizeDocstrings
@ProbabilisticTest(RepeatOnFail=2)
@pytest.mark.parametrize("NBS_pair", [pytest.param(NBS_pair, id=name) for name, NBS_pair in NBS_pairs_dict.items()])
def test_action_cst_sym_pairs(float64_tols, NBS_pair):
    """ Tests that computations results are independent of the prescribed symmetries.
    
Tests that the following computations results are independent of the prescribed symmetries:

* ``all_coeffs`` => ``kinetic energy``
* ``params`` => ``kinetic energy``
* ``params`` => ``potential energy``
* ``params`` => ``action``
* ``params`` => ``path_stats``

    """
    
    NBS_m, NBS_l = NBS_pair
    
    # m => more symmetry. l => less symmetry

    assert NBS_m.nint_min >= NBS_l.nint_min
    assert NBS_m.nint_min % NBS_l.nint_min == 0
    
    if NBS_m.nloop != NBS_l.nloop:
        pytest.skip("Test not suitable for this case")
        
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

@ParametrizeDocstrings
def test_custom_inter_law(float64_tols):
    """ Tests that custom interaction laws are correctly handled.
    """
        
    geodim = 2
    nbody = 3
    bodymass = np.array([1., 2., 3.])
    bodycharge = np.array([4., 5., 6.])
    Sym_list = []
    
    # Classical gravity_pot
    inter_law = scipy.LowLevelCallable.from_cython(choreo.cython._NBodySyst_ann, "gravity_pot")
    NBS = choreo.NBodySyst(geodim, nbody, bodymass, bodycharge, Sym_list, inter_law)
    params_buf = np.random.random((NBS.nparams))   
    action_grad_grav = NBS.params_to_action_grad(params_buf)

    # Parametrized power_law_pot
    inter_law_str = "power_law_pot"
    inter_law_param_dict = {'n':-1.}
    NBS = choreo.NBodySyst(geodim, nbody, bodymass, bodycharge, Sym_list, None, inter_law_str, inter_law_param_dict)
    action_grad = NBS.params_to_action_grad(params_buf)
    
    print(np.linalg.norm(action_grad_grav-action_grad))
    assert np.allclose(action_grad_grav, action_grad, rtol = float64_tols.rtol, atol = float64_tols.atol)  
    
    # Python pot_fun
    def inter_law(xsq, res, ptr):
            
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
def inter_law(xsq, res, ptr):
        
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
        
@ParametrizeDocstrings
@pytest.mark.parametrize("NBS", [pytest.param(NBS, id=name) for name, NBS in NBS_dict.items()])
def test_periodicity_default(float64_tols, NBS):
    """ Tests that Fourier periodic trajectories have zero periodicity default and satisfy initial constraints.
    """
        
    NBS.nint_fac = 10
    params_buf = np.random.random((NBS.nparams))
        
    NBS.ForceGreaterNStore = True
    segmpos = NBS.params_to_segmpos(params_buf)
    segmvel = NBS.params_to_segmvel(params_buf)

    xo = np.ascontiguousarray(segmpos[:,0 ,:].reshape(-1))
    xf = np.ascontiguousarray(segmpos[:,-1,:].reshape(-1))
    
    vo = np.ascontiguousarray(segmvel[:,0 ,:].reshape(-1))
    vf = np.ascontiguousarray(segmvel[:,-1,:].reshape(-1))
    
    dx = NBS.Compute_periodicity_default_pos(xo, xf)
    dv = NBS.Compute_periodicity_default_vel(vo, vf)
    ndof = dx.shape[0]
    
    assert np.allclose(dx, np.zeros((ndof), dtype=np.float64), rtol = float64_tols.rtol, atol = float64_tols.atol)  
    assert np.allclose(dv, np.zeros((ndof), dtype=np.float64), rtol = float64_tols.rtol, atol = float64_tols.atol)         
     
    dx = NBS.Compute_initial_constraint_default_pos(xo)
    dv = NBS.Compute_initial_constraint_default_vel(vo)
    
    assert np.allclose(dx, np.zeros((ndof), dtype=np.float64), rtol = float64_tols.rtol, atol = float64_tols.atol)  
    assert np.allclose(dv, np.zeros((ndof), dtype=np.float64), rtol = float64_tols.rtol, atol = float64_tols.atol)
    
    perdef = NBS.endposmom_to_perdef(xo, vo, xf, vf)
    assert np.allclose(perdef, np.zeros((NBS.n_ODEperdef_eqproj), dtype=np.float64), rtol = float64_tols.rtol, atol = float64_tols.atol)  

    xf = xf.reshape(1,-1)
    vf = vf.reshape(1,-1)
    
    perdef_bulk = NBS.endposmom_to_perdef_bulk(xo, vo, xf, vf)
    assert np.allclose(perdef_bulk, np.zeros((1,NBS.n_ODEperdef_eqproj), dtype=np.float64), rtol = float64_tols.rtol, atol = float64_tols.atol)  

@ParametrizeDocstrings
@pytest.mark.parametrize("NoSymIfPossible", [True, False])
@pytest.mark.parametrize("LowLevel", [True, False])
@pytest.mark.parametrize("vector_calls", [True, False])
@pytest.mark.parametrize(("NBS", "params_buf"), [pytest.param(NBS, params_buf, id=name) for name, (NBS, params_buf) in Sols_dict.items()])
def test_RK_vs_spectral(NBS, params_buf, vector_calls, LowLevel, NoSymIfPossible):
    """ Tests that the Fourier periodic spectral solver agrees with the time forward Runge-Kutta solver.
    """
        
    NBS.ForceGreaterNStore = True
    segmpos = NBS.params_to_segmpos(params_buf)
    segmmom = NBS.params_to_segmmom(params_buf)
    
    action_grad = NBS.segmpos_params_to_action_grad(segmpos, params_buf)
    action_grad_norm = np.linalg.norm(action_grad, ord = np.inf)
    
    ODE_Syst = NBS.Get_ODE_def(params_buf, vector_calls = vector_calls, LowLevel = LowLevel, NoSymIfPossible = NoSymIfPossible)
    
    nsteps = 10
    keep_freq = 1
    nint_ODE = (NBS.segm_store-1) * keep_freq
    method = "Gauss"
    
    rk = choreo.segm.multiprec_tables.ComputeImplicitRKTable(nsteps, method=method)
    
    segmpos_ODE, segmmom_ODE = choreo.segm.ODE.SymplecticIVP(
        rk = rk                 ,
        keep_freq = keep_freq   ,
        nint = nint_ODE         ,
        keep_init = True        ,
        **ODE_Syst              ,
    )

    segmpos_ODE = np.ascontiguousarray(segmpos_ODE.reshape((NBS.segm_store, NBS.nsegm, NBS.geodim)).swapaxes(0, 1))

    tol = 100 * action_grad_norm
    print(np.linalg.norm(segmpos - segmpos_ODE))
    assert np.allclose(segmpos, segmpos_ODE, rtol = tol, atol = tol)  
    
    segmmom_ODE = np.ascontiguousarray(segmmom_ODE.reshape((NBS.segm_store, NBS.nsegm, NBS.geodim)).swapaxes(0, 1))
    
    tol = 20000 * action_grad_norm 
    print(np.linalg.norm(segmmom - segmmom_ODE))
    assert np.allclose(segmmom, segmmom_ODE, rtol = tol, atol = tol)        
    
@ParametrizeDocstrings
@pytest.mark.parametrize("NoSymIfPossible", [True, False])
@pytest.mark.parametrize("LowLevel", [True, False])
@pytest.mark.parametrize("vector_calls", [True, False])
@pytest.mark.parametrize(("NBS", "params_buf"), [pytest.param(NBS, params_buf, id=name) for name, (NBS, params_buf) in Sols_dict.items()])
def test_RK_vs_spectral_reset(NBS, params_buf, vector_calls, LowLevel, NoSymIfPossible):
    """ Tests that the Fourier periodic spectral solver agrees with the time forward Runge-Kutta solver.
    """
        
    NBS.ForceGreaterNStore = True
    segmpos = NBS.params_to_segmpos(params_buf)
    segmmom = NBS.params_to_segmmom(params_buf)

    reg_xo = np.ascontiguousarray(segmpos.swapaxes(0, 1).reshape(NBS.segm_store,-1))
    reg_vo = np.ascontiguousarray(segmmom.swapaxes(0, 1).reshape(NBS.segm_store,-1))
    
    action_grad = NBS.segmpos_params_to_action_grad(segmpos, params_buf)
    action_grad_norm = np.linalg.norm(action_grad, ord = np.inf)
    
    ODE_Syst = NBS.Get_ODE_def(vector_calls = vector_calls, LowLevel = LowLevel, NoSymIfPossible = NoSymIfPossible)
    
    nsteps = 10
    keep_freq = 1
    nint_ODE = (NBS.segm_store-1) * keep_freq
    method = "Gauss"
    reg_init_freq = keep_freq
    
    rk = choreo.segm.multiprec_tables.ComputeImplicitRKTable(nsteps, method=method)
    
    segmpos_ODE, segmmom_ODE = choreo.segm.ODE.ImplicitSymplecticIVP(
        rk_x = rk, rk_v = rk            ,
        keep_freq = keep_freq           ,
        nint = nint_ODE                 ,
        keep_init = True                ,
        reg_xo = reg_xo                 ,
        reg_vo = reg_vo                 ,
        reg_init_freq = reg_init_freq   ,
        **ODE_Syst                      ,
    )

    segmpos_ODE = np.ascontiguousarray(segmpos_ODE.reshape((NBS.segm_store, NBS.nsegm, NBS.geodim)).swapaxes(0, 1))
    segmmom_ODE = np.ascontiguousarray(segmmom_ODE.reshape((NBS.segm_store, NBS.nsegm, NBS.geodim)).swapaxes(0, 1))
    
    tol = 0.1 * action_grad_norm # !!!!
    print(np.linalg.norm(segmpos - segmpos_ODE))
    assert np.allclose(segmpos, segmpos_ODE, rtol = tol, atol = tol)   

    tol = 100 * action_grad_norm
    print(np.linalg.norm(segmmom - segmmom_ODE))
    assert np.allclose(segmmom, segmmom_ODE, rtol = tol, atol = tol)        
    
@ParametrizeDocstrings
@pytest.mark.parametrize("NBS", [pytest.param(NBS, id=name) for name, NBS in NBS_dict.items()])
def test_segmpos_param(float64_tols_strict, NBS):
    """ Tests whether functions taking advantage of ``segmpos`` caching give the same result as those that do not.
    """

    NBS.nint_fac = 10
    params_buf = np.random.random((NBS.nparams))
    dparams_buf = np.random.random((NBS.nparams))
    segmpos = NBS.params_to_segmpos(params_buf)
    
    action_ref = NBS.params_to_action(params_buf)
    action_grad_ref = NBS.params_to_action_grad(params_buf)
    action_hess_ref = NBS.params_to_action_hess(params_buf, dparams_buf)
    
    action_opt = NBS.segmpos_params_to_action(segmpos, params_buf)
    action_grad_opt = NBS.segmpos_params_to_action_grad(segmpos, params_buf)
    action_hess_opt = NBS.segmpos_dparams_to_action_hess(segmpos, dparams_buf)
    
    print(abs(action_ref - action_opt))
    assert abs(action_ref - action_opt) < float64_tols_strict.atol

    print(np.linalg.norm(action_grad_ref - action_grad_opt))
    assert np.allclose(action_grad_ref, action_grad_opt, rtol = float64_tols_strict.rtol, atol = float64_tols_strict.atol)
      
    print(np.linalg.norm(action_hess_ref - action_hess_opt))
    assert np.allclose(action_hess_ref, action_hess_opt, rtol = float64_tols_strict.rtol, atol = float64_tols_strict.atol)  

@ParametrizeDocstrings
@pytest.mark.parametrize("NoSymIfPossible", [True, False])
@pytest.mark.parametrize(("NBS", "params_buf"), [pytest.param(NBS, params_buf, id=name) for name, (NBS, params_buf) in Sols_dict.items()])
def test_grad_fun_FD(float64_tols_loose, NBS, params_buf, NoSymIfPossible):
    """ Tests that the gradient of velocities and forces agree with their finite difference estimations.
    """
        
    NBS.ForceGreaterNStore = True
    
    ODE_Syst = NBS.Get_ODE_def(params_buf, vector_calls = False, LowLevel = False, NoSymIfPossible = NoSymIfPossible, grad = True)
    
    ndof = NBS.nsegm * NBS.geodim
    xo = ODE_Syst["xo"]
    dx = np.random.random((ndof))
    
    fun = lambda x : ODE_Syst["fun"](0., x)
    grad_fun = lambda x, dx : ODE_Syst["grad_fun"](0., x, dx)
    
    err = compare_FD_and_exact_grad(
        fun             ,
        grad_fun        ,
        xo              ,
        dx=dx           ,
        order=2         ,
        vectorize=True  ,
    )

    print(err.min())
    assert (err.min() < float64_tols_loose.rtol)
    
    po = ODE_Syst["vo"]
    
    gun = lambda p : ODE_Syst["gun"](0., p)
    grad_gun = lambda p, dp : ODE_Syst["grad_gun"](0., p, dp)
    
    err = compare_FD_and_exact_grad(
        gun             ,
        grad_gun        ,
        po              ,
        order=2         ,
        vectorize=True  ,
    )

    print(err.min())
    assert (err.min() < float64_tols_loose.rtol)
        
@pytest.mark.slow(required_time = 10)
@ParametrizeDocstrings
@pytest.mark.parametrize("NoSymIfPossible", [True, False])
@pytest.mark.parametrize("vector_calls", [True, False])
@pytest.mark.parametrize("LowLevel", [True, False])
@pytest.mark.parametrize(("NBS", "params_buf"), [pytest.param(NBS, params_buf, id=name) for name, (NBS, params_buf) in Sols_dict.items()])
def test_ODE_grad_vs_FD_Implicit(float64_tols_loose, NBS, params_buf, vector_calls, LowLevel, NoSymIfPossible):
    """ Tests that the solution of the tangent system using an implicit integration agrees with its finite difference estimation.
    """
        
    NBS.ForceGreaterNStore = True
    
    ODE_Syst = NBS.Get_ODE_def(params_buf, vector_calls = vector_calls, LowLevel = LowLevel, NoSymIfPossible = NoSymIfPossible, grad=True)
    
    nint_ODE = (NBS.segm_store-1)

    nsteps = 10
    method = "Gauss"    
    rk = choreo.segm.multiprec_tables.ComputeImplicitRKTable(nsteps, method=method)

    t_span = ODE_Syst["t_span"]
    fun = ODE_Syst["fun"]
    gun = ODE_Syst["gun"]
    grad_fun = ODE_Syst["grad_fun"]
    grad_gun = ODE_Syst["grad_gun"]

    def fun_fd(x):
        
        nn = x.shape[0]
        assert nn % 2 == 0
        
        n = nn // 2
        
        xo = x[0:  n]
        vo = x[n:2*n]
        
        segmpos_ODE, segmvel_ODE = choreo.segm.ODE.ImplicitSymplecticIVP(
            fun = fun                   ,
            gun = gun                   ,
            xo = xo                     ,
            vo = vo                     ,
            rk_x = rk                   ,
            rk_v = rk                   ,
            nint = nint_ODE             ,
            t_span = t_span             ,
            vector_calls = vector_calls ,
        )
        
        res = np.empty((nn), dtype=np.float64)
        
        res[0:  n] = segmpos_ODE[-1,:]
        res[n:2*n] = segmvel_ODE[-1,:]

        return res

    def grad_fun_fd(x, dx):
        
        nn = x.shape[0]
        assert nn % 2 == 0
        
        n = nn // 2
        
        xo = x[0:  n]
        vo = x[n:2*n]
        
        grad_xo = dx[0:  n].reshape((n,1))
        grad_vo = dx[n:2*n].reshape((n,1))
        
        segmpos_ODE, segmvel_ODE, segmpos_grad_ODE, segmvel_grad_ODE = choreo.segm.ODE.ImplicitSymplecticIVP(
            fun = fun                   ,
            grad_fun = grad_fun         ,
            gun = gun                   ,
            grad_gun = grad_gun         ,
            xo = xo                     ,
            grad_xo = grad_xo           ,
            vo = vo                     ,
            grad_vo = grad_vo           ,
            rk_x = rk                   ,
            rk_v = rk                   ,
            nint = nint_ODE             ,
            t_span = t_span             ,
            vector_calls = vector_calls ,
        )
        
        res = np.empty((nn), dtype=np.float64)
        
        res[0:  n] = segmpos_grad_ODE[0,:,0]
        res[n:2*n] = segmvel_grad_ODE[0,:,0]

        return res
    
    n = ODE_Syst["xo"].shape[0]
    nn = 2*n
    
    xo = np.empty((nn), dtype=np.float64)
    xo[0:n ] = ODE_Syst["xo"]
    xo[n:nn] = ODE_Syst["vo"]
    
    err = compare_FD_and_exact_grad(
        fun_fd          ,
        grad_fun_fd     ,
        xo              ,
        order=2         ,
        vectorize=False ,
    )

    print(err.min())
    assert (err.min() < float64_tols_loose.rtol)
       
@pytest.mark.slow(required_time = 10)
@ParametrizeDocstrings
@pytest.mark.parametrize("NoSymIfPossible", [True, False])
@pytest.mark.parametrize("LowLevel", [True, False])
@pytest.mark.parametrize(("NBS", "params_buf"), [pytest.param(NBS, params_buf, id=name) for name, (NBS, params_buf) in Sols_dict.items()])
def test_ODE_grad_vs_FD_Explicit(float64_tols_loose, NBS, params_buf, LowLevel, NoSymIfPossible):
    """ Tests that the solution of the tangent system using an explicit integration agrees with its finite difference estimation.
    """
        
    NBS.ForceGreaterNStore = True
    
    ODE_Syst = NBS.Get_ODE_def(params_buf, vector_calls = False, LowLevel = LowLevel, NoSymIfPossible = NoSymIfPossible, grad=True)
    
    nint_ODE = (NBS.segm_store-1)
    keep_freq = nint_ODE
    
    rk = choreo.segm.precomputed_tables.SofSpa10

    t_span = ODE_Syst["t_span"]
    fun = ODE_Syst["fun"]
    gun = ODE_Syst["gun"]
    grad_fun = ODE_Syst["grad_fun"]
    grad_gun = ODE_Syst["grad_gun"]

    def fun_fd(x):
        
        nn = x.shape[0]
        assert nn % 2 == 0
        
        n = nn // 2
        
        xo = x[0:  n]
        vo = x[n:2*n]
        
        segmpos_ODE, segmvel_ODE = choreo.segm.ODE.SymplecticIVP(
            fun = fun                   ,
            gun = gun                   ,
            xo = xo                     ,
            vo = vo                     ,
            rk = rk                     ,
            nint = nint_ODE             ,
            t_span = t_span             ,
        )
        
        res = np.empty((nn), dtype=np.float64)
        
        res[0:  n] = segmpos_ODE[-1,:]
        res[n:2*n] = segmvel_ODE[-1,:]

        return res

    def grad_fun_fd(x, dx):
        
        nn = x.shape[0]
        assert nn % 2 == 0
        
        n = nn // 2
        
        xo = x[0:  n]
        vo = x[n:2*n]
        
        grad_xo = dx[0:  n].reshape((n,1))
        grad_vo = dx[n:2*n].reshape((n,1))
        
        segmpos_ODE, segmvel_ODE, segmpos_grad_ODE, segmvel_grad_ODE = choreo.segm.ODE.SymplecticIVP(
            fun = fun                   ,
            grad_fun = grad_fun         ,
            gun = gun                   ,
            grad_gun = grad_gun         ,
            xo = xo                     ,
            grad_xo = grad_xo           ,
            vo = vo                     ,
            grad_vo = grad_vo           ,
            rk = rk                     ,
            nint = nint_ODE             ,
            t_span = t_span             ,
        )
        
        res = np.empty((nn), dtype=np.float64)
        
        res[0:  n] = segmpos_grad_ODE[0,:,0]
        res[n:2*n] = segmvel_grad_ODE[0,:,0]

        return res
    
    n = ODE_Syst["xo"].shape[0]
    nn = 2*n
    
    xo = np.empty((nn), dtype=np.float64)
    xo[0:n ] = ODE_Syst["xo"]
    xo[n:nn] = ODE_Syst["vo"]
    
    err = compare_FD_and_exact_grad(
        fun_fd          ,
        grad_fun_fd     ,
        xo              ,
        order=2         ,
        vectorize=False ,
    )

    print(err.min())
    assert (err.min() < float64_tols_loose.rtol)

@pytest.mark.slow(required_time = 10)
@ParametrizeDocstrings
@pytest.mark.parametrize(("NBS", "params_buf"), [pytest.param(NBS, params_buf, id=name) for name, (NBS, params_buf) in Sols_dict.items()])
def test_ODE_grad_period_noopt(float64_tols, NBS, params_buf):
    """ Tests the integration of the tangent system on a minimum interval
    """
        
    NBS.ForceGreaterNStore = True
    
    ODE_Syst = NBS.Get_ODE_def(params_buf, grad=True, regular_init = True)
    
    nint_ODE = (NBS.segm_store-1)

    nsteps = 10
    method = "Gauss"    
    rk = choreo.segm.multiprec_tables.ComputeImplicitRKTable(nsteps, method=method)

    segmpos = NBS.params_to_segmpos(params_buf)
    segmpos_all = NBS.segmpos_to_allsegm_noopt(segmpos, pos=True)
    segmmom = NBS.params_to_segmmom(params_buf)
    segmmom_all = NBS.segmpos_to_allsegm_noopt(segmmom, pos=False)
    
    ODE_Syst["reg_xo"] = np.ascontiguousarray(segmpos_all.swapaxes(0, 1).reshape(NBS.nint,-1))
    ODE_Syst["reg_vo"] = np.ascontiguousarray(segmmom_all.swapaxes(0, 1).reshape(NBS.nint,-1))
        
    segmpos_ODE, segmmom_ODE, segmpos_grad_ODE, segmmom_grad_ODE = choreo.segm.ODE.ImplicitSymplecticIVP(
        nint = nint_ODE         ,
        rk_x = rk, rk_v = rk    ,
        reg_init_freq = 1       ,
        **ODE_Syst              ,
    )

    MonodromyMat_propagated = NBS.PropagateMonodromy_noopt(segmpos_grad_ODE, segmmom_grad_ODE, OnlyFinal = False)
    
    keep_freq = (NBS.segm_store-1)
    nint_ODE *= NBS.nint_min
    ODE_Syst["t_span"] = (0, 1.)
    
    segmpos_ODE_full, segmmom_ODE_full, segmpos_grad_ODE_full, segmmom_grad_ODE_full = choreo.segm.ODE.ImplicitSymplecticIVP(
        nint = nint_ODE         ,
        keep_freq = keep_freq   ,
        rk_x = rk, rk_v = rk    ,
        reg_init_freq = 1       ,
        **ODE_Syst              ,
    )
    
    MonodromyMat_direct = np.ascontiguousarray(np.concatenate((segmpos_grad_ODE_full,segmmom_grad_ODE_full),axis=1)).reshape(NBS.nint_min, 2, NBS.nsegm, NBS.geodim, 2, NBS.nsegm, NBS.geodim)
    
    for i in range(NBS.nint_min):
        print(i, np.linalg.norm(MonodromyMat_propagated[i,...] - MonodromyMat_direct[i,...]))
        assert np.allclose(MonodromyMat_propagated[i,...], MonodromyMat_direct[i,...], rtol = float64_tols.rtol, atol = float64_tols.atol)

@ParametrizeDocstrings
@pytest.mark.parametrize(("NBS", "params_buf"), [pytest.param(NBS, params_buf, id=name) for name, (NBS, params_buf) in Sols_dict.items()])
def test_ODE_grad_period_opt(float64_tols, NBS, params_buf):
    """ Tests the integration of the tangent system on a minimum interval
    """
        
    NBS.ForceGreaterNStore = True

    ODE_Syst = NBS.Get_ODE_def(params_buf, grad=True, regular_init = True)
    
    reg_init_freq = 1
    nint_ODE = (NBS.segm_store-1)*reg_init_freq

    nsteps = 10
    method = "Gauss"    
    rk = choreo.segm.multiprec_tables.ComputeImplicitRKTable(nsteps, method=method)
  
    segmpos_ODE, segmmom_ODE, segmpos_grad_ODE, segmmom_grad_ODE = choreo.segm.ODE.ImplicitSymplecticIVP(
        nint = nint_ODE                 ,
        rk_x = rk, rk_v = rk            ,
        reg_init_freq = reg_init_freq   ,
        **ODE_Syst                      ,
    )

    MonodromyMat_noopt = NBS.PropagateMonodromy_noopt(segmpos_grad_ODE, segmmom_grad_ODE)
    MonodromyMat_opt = NBS.PropagateMonodromy(segmpos_grad_ODE, segmmom_grad_ODE)

    print(np.linalg.norm(MonodromyMat_noopt - MonodromyMat_opt))
    assert np.allclose(MonodromyMat_noopt, MonodromyMat_opt, rtol = float64_tols.rtol, atol = float64_tols.atol)

@pytest.mark.slow(required_time = 10)
@ParametrizeDocstrings
@pytest.mark.parametrize(("NBS_in", "params_buf_in"), [pytest.param(NBS, params_buf, id=name) for name, (NBS, params_buf) in Sols_dict.items()])
def test_Monodromy(float64_tols, NBS_in, params_buf_in):
    """ Tests the properties of Monodromy matrix
    """
    
    if NBS_in.nsegm == NBS_in.nbody:
        NBS = NBS_in
        params_buf = params_buf_in
        
    else:
                
        NBS = NBS_in.copy_nosym()

        segmpos = NBS_in.params_to_segmpos(params_buf_in)
        all_bodypos = NBS_in.segmpos_to_allbody_noopt(segmpos, pos = True )
        params_buf = NBS.segmpos_to_params(all_bodypos)

    NBS.ForceGreaterNStore = True

    ODE_Syst = NBS.Get_ODE_def(params_buf, grad=True, regular_init = True)
    
    reg_init_freq = 1
    nint_ODE = (NBS.segm_store-1)*reg_init_freq

    nsteps = 20
    method = "Gauss"    
    rk = choreo.segm.multiprec_tables.ComputeImplicitRKTable(nsteps, method=method)
  
    segmpos_ODE, segmmom_ODE, segmpos_grad_ODE, segmmom_grad_ODE = choreo.segm.ODE.ImplicitSymplecticIVP(
        nint = nint_ODE                 ,
        rk_x = rk, rk_v = rk            ,
        reg_init_freq = reg_init_freq   ,
        **ODE_Syst                      ,
    )

    n = NBS.nsegm * NBS.geodim

    MonodromyMat = NBS.PropagateMonodromy(segmpos_grad_ODE, segmmom_grad_ODE)
    MonodromyMat = MonodromyMat.reshape(2*n,2*n)

    w = np.zeros((2*n,2*n),dtype=np.float64)
    w[0:n,n:2*n] = np.identity(n)
    w[n:2*n,0:n] = -np.identity(n)
    
    # Symplecticity error
    print(np.linalg.norm(w - np.dot(MonodromyMat.transpose(),np.dot(w,MonodromyMat))))
    cond = np.linalg.norm(MonodromyMat)**2
    rtol = float64_tols.rtol * cond
    atol = float64_tols.atol * cond
    assert np.allclose(w, np.dot(MonodromyMat.transpose(),np.dot(w,MonodromyMat)), rtol = rtol, atol = atol)   

    eigvals = scipy.linalg.eigvals(MonodromyMat)
    print('Max Eigenvalue of the Monodromy matrix :',np.abs(eigvals).max())
    
    cond = np.max(abs(eigvals)) / np.min(abs(eigvals)) * 10
    rtol = float64_tols.rtol * cond
    atol = float64_tols.atol * cond
    
    # Test Loxodromy
    for eigval in eigvals:
        assert inarray(np.conjugate(eigval)     , eigvals, rtol = rtol, atol = atol)
        assert inarray(1./eigval                , eigvals, rtol = rtol, atol = atol)
        assert inarray(1./np.conjugate(eigval)  , eigvals, rtol = rtol, atol = atol)
    
    xo = np.ascontiguousarray(ODE_Syst['reg_xo'][0,:])
    po = np.ascontiguousarray(ODE_Syst['reg_vo'][0,:])
    
    # Periodicity of the solution gives an eigenvector for free
    yo = NBS.Compute_velocities(0., po)
    qo = NBS.Compute_forces(0., xo)
    zo = np.ascontiguousarray(np.concatenate((yo,qo),axis=0).reshape(2*n))
    
    print(np.linalg.norm(np.dot(MonodromyMat, zo) - zo))
    cond = np.linalg.norm(MonodromyMat) * 100
    rtol = float64_tols.rtol * cond
    atol = float64_tols.atol * cond
    assert np.allclose(np.dot(MonodromyMat, zo), zo, rtol = rtol, atol = atol)   
    
@ParametrizeDocstrings
@pytest.mark.parametrize("NBS", [pytest.param(NBS, id=name) for name, NBS in NBS_dict.items()])
def test_remove_all_syms_nrg(float64_tols, NBS):
    """ Tests whether action gradient and hessian computation can be replicated without symmetries
    """
    
    NBS.nint_fac = 10
    params_buf = np.random.random((NBS.nparams))
    segmpos = NBS.params_to_segmpos(params_buf)

    NBS_nosym = NBS.copy_nosym()
    
    for ib in range(NBS_nosym.nbody):
        isegm = NBS_nosym.bodysegm[ib, 0]
        assert isegm == ib
    
    all_bodypos = NBS.segmpos_to_allbody_noopt(segmpos)
    NBS_nosym.nint = NBS.nint

    params_buf_nosym = NBS_nosym.segmpos_to_params(all_bodypos)
    
    kin_nrg = NBS.params_to_kin_nrg(params_buf)
    pot_nrg = NBS.params_to_pot_nrg(params_buf)
    
    kin_nrg_nosym = NBS_nosym.params_to_kin_nrg(params_buf_nosym)
    pot_nrg_nosym = NBS_nosym.params_to_pot_nrg(params_buf_nosym)

    assert 2 * abs(kin_nrg - kin_nrg_nosym) / (abs(kin_nrg) + abs(kin_nrg_nosym)) < float64_tols.rtol
    assert 2 * abs(pot_nrg - pot_nrg_nosym) / (abs(pot_nrg) + abs(pot_nrg_nosym)) < float64_tols.rtol

@ParametrizeDocstrings
@pytest.mark.parametrize(("NBS", "params_buf"), [pytest.param(NBS, params_buf, id=name) for name, (NBS, params_buf) in Sols_dict.items()])
@pytest.mark.parametrize("NoSymIfPossible", [True, False])
@pytest.mark.parametrize("vector_calls", [True, False])
@pytest.mark.parametrize("LowLevel", [True, False])
def test_remove_all_syms_ODE(NBS, params_buf, vector_calls, LowLevel, NoSymIfPossible):
    """ Tests whether ODE computations can be replicated without symmetries
    """

    NBS_nosym = NBS.copy_nosym()
    
    for ib in range(NBS_nosym.nbody):
        isegm = NBS_nosym.bodysegm[ib, 0]
        assert isegm == ib    

    segmpos = NBS.params_to_segmpos(params_buf)
    segmmom = NBS.params_to_segmmom(params_buf)
    
    action_grad = NBS.segmpos_params_to_action_grad(segmpos, params_buf)
    action_grad_norm = np.linalg.norm(action_grad, ord = np.inf)
    
    all_bodypos = NBS.segmpos_to_allbody_noopt(segmpos, pos = True )
    all_bodymom = NBS.segmpos_to_allbody_noopt(segmmom, pos = False)

    params_buf_nosym = NBS_nosym.segmpos_to_params(all_bodypos)

    reg_xo = np.ascontiguousarray(all_bodypos.swapaxes(0, 1).reshape(NBS_nosym.segm_store,-1))
    reg_vo = np.ascontiguousarray(all_bodymom.swapaxes(0, 1).reshape(NBS_nosym.segm_store,-1))
    
    ODE_Syst = NBS_nosym.Get_ODE_def(vector_calls = vector_calls, LowLevel = LowLevel, NoSymIfPossible = NoSymIfPossible)
    
    keep_freq = 1     
    reg_init_freq = keep_freq
    nint_ODE = (NBS_nosym.segm_store) * keep_freq
    
    nsteps = 10
    method = "Gauss"    
    rk = choreo.segm.multiprec_tables.ComputeImplicitRKTable(nsteps, method=method)

    all_bodypos_ODE, all_bodymom_ODE = choreo.segm.ODE.ImplicitSymplecticIVP(
        nint = nint_ODE                 ,
        keep_init = True                ,
        keep_freq = keep_freq           ,
        reg_xo = reg_xo                 ,
        reg_vo = reg_vo                 ,
        reg_init_freq = reg_init_freq   ,
        rk_x = rk, rk_v = rk            ,
        t_span = (0, 1.)                ,
        vector_calls = vector_calls     ,
        fun = ODE_Syst["fun"]           ,
        gun = ODE_Syst["gun"]           ,
    )
    
    # We need to drop the last timestep
    all_bodypos_ODE = np.ascontiguousarray(all_bodypos_ODE.reshape((NBS_nosym.segm_store+1, NBS_nosym.nsegm, NBS_nosym.geodim))[:NBS_nosym.segm_store,:,:].swapaxes(0, 1))    
    all_bodymom_ODE = np.ascontiguousarray(all_bodymom_ODE.reshape((NBS_nosym.segm_store+1, NBS_nosym.nsegm, NBS_nosym.geodim))[:NBS_nosym.segm_store,:,:].swapaxes(0, 1))

    tol = 0.1 * action_grad_norm 
    print(np.linalg.norm(all_bodypos - all_bodypos_ODE))
    assert np.allclose(all_bodypos, all_bodypos_ODE, rtol = tol, atol = tol)   
    
    tol = 100 * action_grad_norm
    print(np.linalg.norm(all_bodymom - all_bodymom_ODE))
    assert np.allclose(all_bodymom, all_bodymom_ODE, rtol = tol, atol = tol)      
    
@ParametrizeDocstrings
@pytest.mark.parametrize("nbody", [2,3,5,10,15])
@pytest.mark.parametrize("eccentricity", [0.,0.5,0.8,0.95])
def test_Kepler(float64_tols, float64_tols_loose, nbody, eccentricity):
    """ Tests on exact Kepler solutions
    """
    
    mass = 1. + np.random.random()
    nint_fac = 1024
    # nint_fac = 2048

    NBS, ODE_dict = choreo.NBodySyst.KeplerEllipse(nbody, nint_fac, mass, eccentricity)
    segmpos = ODE_dict["segmpos"]
    params_buf = NBS.segmpos_to_params(segmpos)
    action_grad = NBS.segmpos_params_to_action_grad(segmpos, params_buf)
    action_grad_norm = np.linalg.norm(action_grad, ord = np.inf)

    ndof = action_grad.shape[0]

    print(action_grad_norm)
    assert np.allclose(action_grad, np.zeros((ndof), dtype=np.float64), rtol = float64_tols.rtol, atol = float64_tols.atol*10)  

    segmmom = ODE_dict["segmmom"]
    segmmom_rt = NBS.params_to_segmmom(params_buf)
    
    print(np.linalg.norm(segmmom - segmmom_rt))
    assert np.allclose(segmmom, segmmom_rt, rtol = float64_tols_loose.rtol, atol = float64_tols_loose.atol)  

    reg_xo = ODE_dict["reg_xo"]
    reg_vo = ODE_dict["reg_vo"]

    nsteps = 10
    keep_freq = 1
    nint_ODE = (NBS.segm_store-1) * keep_freq
    method = "Gauss"
    reg_init_freq = keep_freq
    
    rk = choreo.segm.multiprec_tables.ComputeImplicitRKTable(nsteps, method=method)
    
    vector_calls = ODE_dict["vector_calls"]
    fun = ODE_dict["fun"]
    gun = ODE_dict["gun"]
    t_span = ODE_dict["t_span"]
    
    segmpos_ODE, segmmom_ODE = choreo.segm.ODE.ImplicitSymplecticIVP(
        fun = fun                       ,
        gun = gun                       ,
        t_span = t_span                 ,
        rk_x = rk, rk_v = rk            ,
        keep_freq = keep_freq           ,
        nint = nint_ODE                 ,
        keep_init = True                ,
        reg_xo = reg_xo                 ,
        reg_vo = reg_vo                 ,
        reg_init_freq = reg_init_freq   ,
        vector_calls = vector_calls     ,
    )

    segmpos_ODE = np.ascontiguousarray(segmpos_ODE.reshape((NBS.segm_store, NBS.nsegm, NBS.geodim)).swapaxes(0, 1))
    segmmom_ODE = np.ascontiguousarray(segmmom_ODE.reshape((NBS.segm_store, NBS.nsegm, NBS.geodim)).swapaxes(0, 1))
    
    print(np.linalg.norm(segmpos - segmpos_ODE))
    assert np.allclose(segmpos, segmpos_ODE, rtol = float64_tols.rtol, atol = float64_tols.atol)  

    print(np.linalg.norm(segmmom - segmmom_ODE))
    assert np.allclose(segmmom, segmmom_ODE, rtol = float64_tols_loose.rtol, atol = float64_tols_loose.atol)  
    
@ParametrizeDocstrings
@pytest.mark.parametrize("NoSymIfPossible", [True, False])
@pytest.mark.parametrize("LowLevel", [True, False])
@pytest.mark.parametrize("vector_calls", [True, False])
@pytest.mark.parametrize(("NBS", "params_buf"), [pytest.param(NBS, params_buf, id=name) for name, (NBS, params_buf) in Sols_dict.items()])
def test_RK_vs_spectral_periodicity_default(float64_tols, NBS, params_buf, vector_calls, LowLevel, NoSymIfPossible):
    """ Tests that the Fourier periodic spectral solver agrees with the time forward Runge-Kutta solver.
    Tests that Fourier periodic trajectories have zero periodicity default and satisfy initial constraints.
    """
        
    NBS.ForceGreaterNStore = True
    
    action_grad = NBS.params_to_action_grad(params_buf)
    action_grad_norm = np.linalg.norm(action_grad, ord = np.inf)
    
    ODE_Syst = NBS.Get_ODE_def(params_buf, vector_calls = vector_calls, LowLevel = LowLevel, NoSymIfPossible = NoSymIfPossible, regular_init=True)
    
    nsteps = 10
    keep_freq = 1
    nint_ODE = (NBS.segm_store-1) * keep_freq
    method = "Gauss"
    
    rk = choreo.segm.multiprec_tables.ComputeImplicitRKTable(nsteps, method=method)
    
    xf, vf = choreo.segm.ODE.ImplicitSymplecticIVP(
        rk_x = rk               ,
        rk_v = rk               ,
        keep_freq = keep_freq   ,
        nint = nint_ODE         ,
        keep_init = True        ,
        **ODE_Syst              ,
    )
    
    xo = ODE_Syst["reg_xo"].reshape(NBS.segm_store, NBS.nsegm, NBS.geodim)
    vo = ODE_Syst["reg_vo"].reshape(NBS.segm_store, NBS.nsegm, NBS.geodim)    
    xf = xf.reshape(NBS.segm_store, NBS.nsegm, NBS.geodim)
    vf = vf.reshape(NBS.segm_store, NBS.nsegm, NBS.geodim)
    
    tol = 100 * action_grad_norm
    dx = xf-xo
    print(np.linalg.norm(dx))
    assert np.allclose(dx, np.zeros_like(dx), rtol = tol, atol = tol)  
    
    tol = 20000 * action_grad_norm 
    dv = vf-vo
    print(np.linalg.norm(dv))
    assert np.allclose(dv, np.zeros_like(dv), rtol = tol, atol = tol)   
    
    n = (NBS.segm_store-1)
    
    dx = NBS.Compute_periodicity_default_pos(xo[0,:,:].reshape(-1), xo[n,:,:].reshape(-1))
    print(np.linalg.norm(dx))
    assert np.allclose(dx, np.zeros_like(dx), rtol = float64_tols.rtol, atol = float64_tols.atol)  
    
    tol = 100 * action_grad_norm
    dx = NBS.Compute_periodicity_default_pos(xf[0,:,:].reshape(-1), xf[n,:,:].reshape(-1))
    print(np.linalg.norm(dx))
    assert np.allclose(dx, np.zeros_like(dx), rtol = tol, atol = tol)   
      
    dv = NBS.Compute_periodicity_default_vel(vo[0,:,:].reshape(-1), vo[n,:,:].reshape(-1))
    print(np.linalg.norm(dv))
    assert np.allclose(dv, np.zeros_like(dv), rtol = float64_tols.rtol, atol = float64_tols.atol)  
    
    tol = 20000 * action_grad_norm 
    dv = NBS.Compute_periodicity_default_vel(vf[0,:,:].reshape(-1), vf[n,:,:].reshape(-1))
    print(np.linalg.norm(dv))
    assert np.allclose(dv, np.zeros_like(dv), rtol = tol, atol = tol) 

    dx, dv = NBS.Compute_ODE_default(xo, vo, xf, vf)

    tol = 100 * action_grad_norm
    print(np.linalg.norm(dx))
    assert np.allclose(dx, np.zeros_like(dx), rtol = tol, atol = tol)  
    
    tol = 20000 * action_grad_norm 
    print(np.linalg.norm(dv))
    assert np.allclose(dv, np.zeros_like(dv), rtol = tol, atol = tol)   

@ParametrizeDocstrings
@pytest.mark.parametrize("NBS", [pytest.param(NBS, id=name) for name, NBS in NBS_dict.items()])
def test_center_of_mass(float64_tols, NBS):
    """ Tests properties of center of mass
    """
        
    NBS.nint_fac = 10
    params_buf  = np.random.random((NBS.nparams))
    segmpos = NBS.params_to_segmpos(params_buf)
    
    cm = NBS.ComputeCenterOfMass(segmpos)

    for idim in range(NBS.geodim):
        segmpos[:,:,idim] -= cm[idim]
        
    cm = NBS.ComputeCenterOfMass(segmpos)

    print(np.linalg.norm(cm))
    assert np.linalg.norm(cm) < float64_tols.atol

@ParametrizeDocstrings
@pytest.mark.parametrize("NBS", [pytest.param(NBS, id=name) for name, NBS in NBS_dict.items()])
def test_perdef(float64_tols, NBS):
    """ A bunch of things that I use more or less implicitely. Them not being true would likely break something.
    """
        
    assert NBS.n_ODEperdef_eqproj == NBS.n_ODEinitparams
    assert NBS.RequiresGreaterNStore == (NBS.TimeRev < 0)
        
    for iint in range(NBS.nint_min):
        
        jint = (iint + 2) % NBS.nint_min
        
        for ib in range(NBS.nbody):
            
            Sym_i = NBS.intersegm_to_all[ib][iint]
            Sym_j = NBS.intersegm_to_all[ib][jint]
            
            assert Sym_i.TimeRev == Sym_j.TimeRev
    
    for iint in range(NBS.nint_min):
        # Is this always true ?
        for ib in range(NBS.nbody):
            
            if (iint % 2) == 0:
                TimeRev =  1
            else:
                TimeRev = NBS.TimeRev
                
            assert NBS.intersegm_to_all[ib][iint].TimeRev == TimeRev
            
@ParametrizeDocstrings
@pytest.mark.parametrize("NBS", [pytest.param(NBS, id=name) for name, NBS in NBS_dict.items()])
def test_initposmom(float64_tols, NBS):
    """ 
        Tests that ODEinitparams and (xo, vo) behave correctly
    """
    
    ODEinitparams = np.random.random((NBS.n_ODEinitparams))
    
    xo, vo = NBS.ODE_params_to_initposmom(ODEinitparams)
    
    dx = NBS.Compute_initial_constraint_default_pos(xo)
    dv = NBS.Compute_initial_constraint_default_vel(vo)
    ndof = NBS.nsegm * NBS.geodim
    
    print(np.linalg.norm(dx))
    assert np.allclose(dx, np.zeros((ndof), dtype=np.float64), rtol = float64_tols.rtol, atol = float64_tols.atol)  
    print(np.linalg.norm(dv))
    assert np.allclose(dv, np.zeros((ndof), dtype=np.float64), rtol = float64_tols.rtol, atol = float64_tols.atol)   
    
    ODEinitparams_rt = NBS.initposmom_to_ODE_params(xo, vo)
    
    print(np.linalg.norm(ODEinitparams-ODEinitparams_rt))
    assert np.allclose(ODEinitparams, ODEinitparams_rt, rtol = float64_tols.rtol, atol = float64_tols.atol)  
    
    
@pytest.mark.slow(required_time = 10)
@ParametrizeDocstrings
@RetryTest(n = 2)
@pytest.mark.parametrize("NBS", [pytest.param(NBS, id=name) for name, NBS in NBS_dict.items()])
def test_params_to_periodicity_default_grad_vs_FD(float64_tols_loose, NBS):
    """ Tests that the gradient of the RK periodicity default computation agrees with its FD approximation """
        
    rk_explicit = choreo.segm.precomputed_tables.SofSpa10
    
    nint_fac_ini = NBS.nint_fac
    
    NBS.nint_fac = 512    
    NBS.setup_params_to_periodicity_default(rk_explicit = rk_explicit)

    xo = np.random.random((NBS.n_ODEinitparams)) * 100
    
    err = compare_FD_and_exact_grad(
        NBS.params_to_periodicity_default       ,
        NBS.params_to_periodicity_default_grad  ,
        xo                                      ,
        order=2                                 ,
        vectorize=False                         ,
    )

    print(err.min())
    assert (err.min() < float64_tols_loose.rtol)

    NBS.nint_fac = nint_fac_ini