'''
ODE.py : Defines ODE-related things I designed I feel ought to be in scipy.

'''

import numpy as np
import math as m
import mpmath
import functools

from choreo.scipy_plus.cython.ODE import ExplicitSymplecticIVP
from choreo.scipy_plus.cython.ODE import ExplicitSymplecticRKTable

from choreo.scipy_plus.cython.ODE import ImplicitSymplecticIVP
from choreo.scipy_plus.cython.ODE import ImplicitSymplecticRKTable


def SymplecticIVP(
    fun         ,
    gun         ,
    t_span      ,
    x0          ,
    v0          ,
    rk          ,
    **kwargs    ,
): 

    if isinstance(rk,ExplicitSymplecticRKTable):
        return ExplicitSymplecticIVP(
            fun         ,
            gun         ,
            t_span      ,
            x0          ,
            v0          ,
            rk          ,
            **kwargs    ,
        )
    
    elif isinstance(rk,ImplicitSymplecticRKTable):
        return ImplicitSymplecticIVP(
            fun         ,
            gun         ,
            t_span      ,
            x0          ,
            v0          ,
            rk_x = rk   ,
            rk_v = rk.symplectic_adjoint()   ,
            **kwargs    ,
        )

    else:
        raise ValueError(f'Unknown rk type : {type(rk)}')
