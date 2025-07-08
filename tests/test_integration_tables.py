""" Tests related to the construction of integration tables.

.. autosummary::
    :toctree: _generated/
    :template: tests-formatting/base.rst
    :nosignatures:

    test_QuadDefaultDPSIsEnough
    test_ImplicitRKDefaultDPSIsEnough
    test_ImplicitSymplecticPairs
    test_ImplicitSymmetricPairs
    test_ImplicitSymmetricSymplecticPairs

"""

import pytest
from .test_config import *
import numpy as np
import scipy
import choreo

@ParametrizeDocstrings
@pytest.mark.parametrize("method", QuadMethods)
@pytest.mark.parametrize("nsteps", Small_orders)
def test_QuadDefaultDPSIsEnough(float64_tols_strict, method, nsteps):
    """ Tests whether the default precision is sufficient for computing quadrature methods.
    
Tests whether the default value of the dps parameter of :func:`choreo.segm.multiprec_tables.ComputeQuadrature` is large enough to ensure that that the quadrature tables are exact at least up to double precision.
    
    """

    dps_overkill = 1000

    quad = choreo.segm.multiprec_tables.ComputeQuadrature(nsteps, method=method)
    quad_overkill = choreo.segm.multiprec_tables.ComputeQuadrature(nsteps, dps=dps_overkill, method=method)

    print(np.linalg.norm(quad.x  - quad_overkill.x))
    print(np.linalg.norm(quad.w  - quad_overkill.w))
    print(np.linalg.norm(quad.wlag  - quad_overkill.wlag))

    assert  np.allclose(
        quad.x                          ,
        quad_overkill.x                 ,
        atol = float64_tols_strict.atol ,
        rtol = float64_tols_strict.rtol ,
    )
    assert  np.allclose(
        quad.w                          ,
        quad_overkill.w                 ,
        atol = float64_tols_strict.atol ,
        rtol = float64_tols_strict.rtol ,
    )
    assert  np.allclose(
        quad.wlag                       ,
        quad_overkill.wlag              ,
        atol = float64_tols_strict.atol ,
        rtol = float64_tols_strict.rtol ,
    )
    
    quad_ad = quad.symmetric_adjoint()
    print(quad.symmetry_default(quad_ad))
    assert quad.is_symmetric_pair(quad_ad, tol = float64_tols_strict.atol)
    
@ParametrizeDocstrings
@pytest.mark.parametrize("method", ClassicalImplicitRKMethods)
@pytest.mark.parametrize("nsteps", Small_orders)
def test_ImplicitRKDefaultDPSIsEnough(float64_tols_strict, float64_tols, method, nsteps):
    """ Tests whether the default precision is sufficient for computing implicit Runge-Kutta tables.
    
Tests whether the default value of the dps parameter of :func:`choreo.segm.multiprec_tables.ComputeImplicitRKTable` is large enough to ensure that that the Runge-Kutta tables are exact at least up to double precision.
    
    """

    dps_overkill = 1000

    rk = choreo.segm.multiprec_tables.ComputeImplicitRKTable(nsteps, method=method)
    rk_overkill = choreo.segm.multiprec_tables.ComputeImplicitRKTable(nsteps, dps=dps_overkill, method=method)

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
    assert rk.is_symplectic_pair(rk_ad, tol = float64_tols.atol)
    
    rk_ad = rk.symmetric_adjoint()
    print(rk.symmetry_default(rk_ad))
    assert rk.is_symmetric_pair(rk_ad, tol = float64_tols.atol)
    
@ParametrizeDocstrings
@pytest.mark.parametrize("method_pair", SymplecticImplicitRKMethodPairs)
@pytest.mark.parametrize("nsteps", Small_orders)
def test_ImplicitSymplecticPairs(float64_tols_strict, method_pair, nsteps):
    """ Tests whether symplectic pairs of implicit Runge-Kutta tables are indeed symplectic at least up to double precision accuracy.    
    """

    rk      = choreo.segm.multiprec_tables.ComputeImplicitRKTable(nsteps, method=method_pair[0])
    rk_ad   = choreo.segm.multiprec_tables.ComputeImplicitRKTable(nsteps, method=method_pair[1])
    
    print(rk.symplectic_default(rk_ad))
    
    assert rk.is_symplectic_pair(rk_ad, tol = float64_tols_strict.atol)

@ParametrizeDocstrings
@pytest.mark.parametrize("method_pair", SymmetricImplicitRKMethodPairs)
@pytest.mark.parametrize("nsteps", Small_orders)
def test_ImplicitSymmetricPairs(float64_tols_strict, method_pair, nsteps):
    """ Tests whether symmetric pairs of implicit Runge-Kutta tables are indeed symmetric at least up to double precision accuracy.
    """   
    
    rk      = choreo.segm.multiprec_tables.ComputeImplicitRKTable(nsteps, method=method_pair[0])
    rk_ad   = choreo.segm.multiprec_tables.ComputeImplicitRKTable(nsteps, method=method_pair[1])
    
    print(rk.symmetry_default(rk_ad))
    
    assert rk.is_symmetric_pair(rk_ad, tol = float64_tols_strict.atol)

@ParametrizeDocstrings
@pytest.mark.parametrize("method_pair", SymmetricSymplecticImplicitRKMethodPairs)
@pytest.mark.parametrize("nsteps", Small_orders)
def test_ImplicitSymmetricSymplecticPairs(float64_tols_strict, method_pair, nsteps):
    """ Tests whether symmetric-symplectic pairs of explicit Runge-Kutta tables are indeed symmetric-symplectic at least up to double precision accuracy.
    """   

    rk      = choreo.segm.multiprec_tables.ComputeImplicitRKTable(nsteps, method=method_pair[0])
    rk_ad   = choreo.segm.multiprec_tables.ComputeImplicitRKTable(nsteps, method=method_pair[1]).symmetric_adjoint()
    
    print(rk.symplectic_default(rk_ad))
    
    assert rk.is_symplectic_pair(rk_ad, tol = float64_tols_strict.atol)


@ParametrizeDocstrings
@pytest.mark.parametrize("method", ClassicalImplicitRKMethods)
@pytest.mark.parametrize("nsteps", Small_orders)
def test_ClassicalImplicitSymmetricPairs(float64_tols_strict, method, nsteps):
    """ Tests whether symplectic pairs of implicit Runge-Kutta tables can be generated at least up to double precision accuracy.
    """   
    
    rk, rk_ad = choreo.segm.multiprec_tables.ComputeImplicitSymplecticRKTablePair(nsteps, method=method)

    print(rk.symplectic_default(rk_ad))
    
    assert rk.is_symplectic_pair(rk_ad, tol = float64_tols_strict.atol)


