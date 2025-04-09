""" Configuration module for tests.

This module defines several useful functions and :py:func:`pytest:pytest.fixture` for running and parameterizing tests.

Floating point precision
------------------------

.. autosummary::
    :toctree: _generated/
    :template: tests-formatting/base.rst
    :nosignatures:

    float64_tols_strict
    float64_tols
    float64_tols_loose
    float32_tols_strict
    float32_tols

"""

from .test_config_precision  import *
from .test_config_quad_ODE   import *
from .test_config_NBodySyst  import *
from .test_config_decorators import *
