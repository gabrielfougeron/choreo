"""
======
choreo
======


===================
Defining Symmetries
===================

.. autosummary::
   :toctree: _generated/
   :caption: Symmetries
   :nosignatures:

   NBodySyst
   ActionSym
      
===========
Entrypoints
===========

.. autosummary::
   :toctree: _generated/
   :caption: Symmetries
   :nosignatures:

   run.entrypoint_CLI_search
      
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




