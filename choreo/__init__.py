"""
======
choreo
======


===================================
Systems of interacting point masses
===================================

.. autosummary::
    :toctree: _generated/
    :caption: N-body system
    :nosignatures:

    NBodySyst
    
===================
Defining Symmetries
===================

.. autosummary::
    :toctree: _generated/
    :caption: Symmetries
    :nosignatures:

    ActionSym
      
===========
Entrypoints
===========

.. autoprogram:: choreo.run:CLI_search_parser
      
"""



from .metadata import *

try:
    from .numba_funs    import *
    NUMBA_AVAILABLE = True
except:
    NUMBA_AVAILABLE = False

from .cython            import NBodySyst, ActionSym
from .                  import scipy_plus
from .                  import run

from .find              import *




