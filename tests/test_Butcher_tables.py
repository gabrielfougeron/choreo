import pytest
from test_config import *
import numpy as np
import scipy
import choreo


@pytest.mark.parametrize("method", ClassicalImplicitRKMethods)
@pytest.mark.parametrize("nsteps", Small_orders)
def test_ImplicitRKDefaultDPSIsEnough(float64_tols_strict, method, nsteps):

    dps_overkill = 1000

    rk = choreo.scipy_plus.multiprec_tables.ComputeImplicitRKTable_Gauss(nsteps, method=method)
    rk_overkill = choreo.scipy_plus.multiprec_tables.ComputeImplicitRKTable_Gauss(nsteps, dps=dps_overkill, method=method)

    print(np.linalg.norm(rk.a_table  - rk_overkill.a_table))
    print(np.linalg.norm(rk.b_table  - rk_overkill.b_table))
    print(np.linalg.norm(rk.c_table  - rk_overkill.c_table))
    print(np.linalg.norm(rk.beta_table  - rk_overkill.beta_table))
    print(np.linalg.norm(rk.gamma_table  - rk_overkill.gamma_table))

    assert  np.allclose(
        rk.a_table                      ,
        rk_overkill.a_table             ,
        atol = float64_tols_strict.atol ,
        rtol = float64_tols_strict.rtol ,
    )
    assert  np.allclose(
        rk.b_table                      ,
        rk_overkill.b_table             ,
        atol = float64_tols_strict.atol ,
        rtol = float64_tols_strict.rtol ,
    )
    assert  np.allclose(
        rk.c_table                      ,
        rk_overkill.c_table             ,
        atol = float64_tols_strict.atol ,
        rtol = float64_tols_strict.rtol ,
    )
    assert  np.allclose(
        rk.beta_table                   ,
        rk_overkill.beta_table          ,
        atol = float64_tols_strict.atol ,
        rtol = float64_tols_strict.rtol ,
    )
    assert  np.allclose(
        rk.gamma_table                  ,
        rk_overkill.gamma_table         ,
        atol = float64_tols_strict.atol ,
        rtol = float64_tols_strict.rtol ,
    )
    
    rk_ad = rk.symplectic_adjoint()
    print(rk.symplectic_default(rk_ad))
    assert rk.is_symplectic_pair(rk_ad, tol = float64_tols_strict.atol)
    
    rk_ad = rk.symmetric_adjoint()
    print(rk.symmetry_default(rk_ad))
    assert rk.is_symmetric_pair(rk_ad, tol = float64_tols_strict.atol)
    
@pytest.mark.parametrize("method_pair", SymplecticImplicitRKMethodPairs)
@pytest.mark.parametrize("nsteps", Small_orders)
def test_ImplicitSymplecticPairs(float64_tols_strict, method_pair, nsteps):

    rk      = choreo.scipy_plus.multiprec_tables.ComputeImplicitRKTable_Gauss(nsteps, method=method_pair[0])
    rk_ad   = choreo.scipy_plus.multiprec_tables.ComputeImplicitRKTable_Gauss(nsteps, method=method_pair[1])
    
    print(rk.symplectic_default(rk_ad))
    
    assert rk.is_symplectic_pair(rk_ad, tol = float64_tols_strict.atol)

@pytest.mark.parametrize("method_pair", SymmetricImplicitRKMethodPairs)
@pytest.mark.parametrize("nsteps", Small_orders)
def test_ImplicitSymmetricPairs(float64_tols_strict, method_pair, nsteps):

    rk      = choreo.scipy_plus.multiprec_tables.ComputeImplicitRKTable_Gauss(nsteps, method=method_pair[0])
    rk_ad   = choreo.scipy_plus.multiprec_tables.ComputeImplicitRKTable_Gauss(nsteps, method=method_pair[1])
    
    print(rk.symmetry_default(rk_ad))
    
    assert rk.is_symmetric_pair(rk_ad, tol = float64_tols_strict.atol)

@pytest.mark.parametrize("method_pair", SymmetricSymplecticImplicitRKMethodPairs)
@pytest.mark.parametrize("nsteps", Small_orders)
def test_ImplicitSymmetricSymplecticPairs(float64_tols_strict, method_pair, nsteps):

    rk      = choreo.scipy_plus.multiprec_tables.ComputeImplicitRKTable_Gauss(nsteps, method=method_pair[0])
    rk_ad   = choreo.scipy_plus.multiprec_tables.ComputeImplicitRKTable_Gauss(nsteps, method=method_pair[1]).symmetric_adjoint()
    
    print(rk.symplectic_default(rk_ad))
    
    assert rk.is_symplectic_pair(rk_ad, tol = float64_tols_strict.atol)





