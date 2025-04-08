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
Defining symmetries
===================

.. autosummary::
    :toctree: _generated/
    :caption: Symmetries
    :nosignatures:

    ActionSym    
    
==========================
Numerical tools on segment
==========================

.. autosummary::
    :toctree: _generated/
    :caption: Segment quadrature / ODE
    :nosignatures:

    segm.quad.QuadTable
    segm.multiprec_tables.ComputeQuadrature
    segm.quad.IntegrateOnSegment

      
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
from .                  import segm
from .                  import scipy_plus
from .                  import run

from .find              import *




