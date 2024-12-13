import pytest
from test_config import *
import numpy as np
import scipy
import choreo

@pytest.mark.parametrize("nsteps", High_orders)
@pytest.mark.parametrize("method_x,method_v", [pytest.param(method_x, method_v) for (method_x, method_v) in SymplecticImplicitRKMethodPairs])
def test_Implicit_ODE(float64_tols_strict, method_x, method_v, nsteps):

    rk_x = choreo.scipy_plus.multiprec_tables.ComputeImplicitRKTable_Gauss(nsteps, method = method_x)
    rk_v = choreo.scipy_plus.multiprec_tables.ComputeImplicitRKTable_Gauss(nsteps, method = method_v)
   
