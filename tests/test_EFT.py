""" Tests the properties of Error-Free Transformations for floating point.

These tests will fail if :mod:`choreo` is not compiled with the correct optimization flags.

.. autosummary::
    :toctree: _generated/
    :template: tests-formatting/base.rst
    :nosignatures:

    test_elemental

"""

import pytest
from .test_config import *
import numpy as np
import choreo

def test_elemental(float64_tols):
    """Tests the behavior of the most basic error-free transformations.
    """

    a = 1e30
    x, y = choreo.segm.cython.eft_lib.Split_py(a)
    assert y != 0.
    
    a = 0.1
    b = 0.2
    x, y = choreo.segm.cython.eft_lib.TwoSum_py(a, b)
    assert y != 0.    
    
    a = 0.1
    b = 0.2
    x, y = choreo.segm.cython.eft_lib.TwoProduct_py(a, b)
    assert y != 0.   
    
    

