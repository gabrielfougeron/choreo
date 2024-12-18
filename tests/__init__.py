""" This is a docstring for tests

.. autosummary::
    :toctree: _generated/
    :caption: Description of tests
    :template: only-explicit/module.rst

    test_config
    test_scipy_plus_linalg
    test_integration_tables
    test_ODE
    test_ActionSym
    test_NBodySyst

"""

from . import test_config

# For Sphinx only
from . import test_scipy_plus_linalg
from . import test_integration_tables
from . import test_ODE
from . import test_ActionSym
from . import test_NBodySyst
