""" Tests the properties of quadratures and ODE solvers.

.. autosummary::
    :toctree: _generated/
    :template: tests-formatting/base.rst
    :nosignatures:

    test_quad
    test_Implicit_ODE
    test_Explicit_ODE

"""

import pytest
from .test_config import *
import numpy as np
import choreo

def test_elemental(float64_tols):
    """Tests the behavior of the most basic element-free transfromations.
    """

    a = 1e30
    x, y = choreo.segm.cython.eft_lib.Split_py(a)
    assert y != 0.
    
    

