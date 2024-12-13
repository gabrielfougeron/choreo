import pytest
from test_config import *
import numpy as np
import choreo

define_ODE_ivp


@pytest.mark.parametrize("DoEFT", [True, False])
@pytest.mark.parametrize("vector_calls", [True, False])
@pytest.mark.parametrize("fun_type", all_fun_types)
@pytest.mark.parametrize("ivp", [pytest.param(define_ODE_ivp(eq_name), id=eq_name) for eq_name in all_eq_names])
@pytest.mark.parametrize("nsteps", Small_orders)
@pytest.mark.parametrize("method_x,method_v", [pytest.param(method_x, method_v) for (method_x, method_v) in SymplecticImplicitRKMethodPairs])
def test_Implicit_ODE(float64_tols, method_x, method_v, nsteps, ivp, fun_type, vector_calls, DoEFT):

    rk_x = choreo.scipy_plus.multiprec_tables.ComputeImplicitRKTable_Gauss(nsteps, method = method_x)
    rk_v = choreo.scipy_plus.multiprec_tables.ComputeImplicitRKTable_Gauss(nsteps, method = method_v)
   
    fgun = ivp["fgun"].get((fun_type, vector_calls))
    
    if fgun is None:
        return
    
    fun, gun = fgun
    
    t_span = ivp["t_span"]
    ex_sol_x = ivp["ex_sol_x"]
    ex_sol_v = ivp["ex_sol_v"]
    nint_OK = ivp["nint"](rk_x.th_cvg_rate)
    
    if nint_OK is None:
        nint = 1
    else:
        nint = nint_OK
    
    x0 = ex_sol_x(t_span[0])
    v0 = ex_sol_v(t_span[0])
    
    xf, vf = choreo.scipy_plus.ODE.ImplicitSymplecticIVP(
        fun                         ,
        gun                         ,
        t_span                      ,
        x0                          ,
        v0                          ,
        rk_x = rk_x                 ,
        rk_v = rk_v                 ,
        vector_calls = vector_calls ,
        nint = nint                 ,
        DoEFT = DoEFT               ,
    )
    
    xf_ex = ex_sol_x(t_span[1])
    vf_ex = ex_sol_v(t_span[1])
    
    print(np.linalg.norm(xf-xf_ex))
    if nint_OK is not None:
        assert np.allclose(xf, xf_ex, rtol = float64_tols.rtol, atol = float64_tols.atol) 
    
    print(np.linalg.norm(vf-vf_ex))
    if nint_OK is not None:
        assert np.allclose(vf, vf_ex, rtol = float64_tols.rtol, atol = float64_tols.atol) 
    
@pytest.mark.parametrize("DoEFT", [True, False])
@pytest.mark.parametrize("fun_type", all_fun_types)
@pytest.mark.parametrize("ivp", [pytest.param(define_ODE_ivp(eq_name), id=eq_name) for eq_name in all_eq_names])
@pytest.mark.parametrize("rk", [pytest.param(rk, id=name) for name, rk in Explicit_tables_dict.items()])
def test_Explicit_ODE(float64_tols, rk, ivp, fun_type, DoEFT):

    fgun = ivp["fgun"].get((fun_type, False))
    
    if fgun is None:
        return
    
    fun, gun = fgun
    
    t_span = ivp["t_span"]
    ex_sol_x = ivp["ex_sol_x"]
    ex_sol_v = ivp["ex_sol_v"]
    nint_OK = ivp["nint"](rk.th_cvg_rate)
    
    if nint_OK is None:
        nint = 1
    else:
        nint = nint_OK
        
    x0 = ex_sol_x(t_span[0])
    v0 = ex_sol_v(t_span[0])
    
    xf, vf = choreo.scipy_plus.ODE.ExplicitSymplecticIVP(
        fun             ,
        gun             ,
        t_span          ,
        x0              ,
        v0              ,
        rk = rk         ,
        nint = nint     ,
        DoEFT = DoEFT   ,
    )
    
    xf_ex = ex_sol_x(t_span[1])
    vf_ex = ex_sol_v(t_span[1])
    
    print(np.linalg.norm(xf-xf_ex))
    if nint_OK is not None:
        assert np.allclose(xf, xf_ex, rtol = float64_tols.rtol, atol = float64_tols.atol) 
    
    print(np.linalg.norm(vf-vf_ex))
    if nint_OK is not None:
        assert np.allclose(vf, vf_ex, rtol = float64_tols.rtol, atol = float64_tols.atol) 
    
    