API
===

.. note::
    Even though the performance critical sections of :mod:`choreo` are coded in `Cython <https://cython.org/>`_, the actual underlying C types only appear in this API documentation as part of function prototypes. The rest of the API documentation uses `Python <https://www.python.org/>`_ types as expected from a user interacting with the code. This follows `Cython's automatic type conversions <https://cython.readthedocs.io/en/latest/src/userguide/language_basics.html#automatic-type-conversions>`_ and uses :class:`numpy:numpy.ndarray` with specified :class:`shape` and :class:`dtype` attributes when possible in lieu of `Cython typed memoryviews <https://cython.readthedocs.io/en/latest/src/userguide/memoryviews.html>`_ as present in the code.


    The following modules classes are concerned:
    
    * :class:`choreo.ActionSym`
    * :class:`choreo.NBodySyst`
    * :mod:`choreo.segm.ODE`


.. automodule:: choreo
.. automodule:: choreo_GUI

