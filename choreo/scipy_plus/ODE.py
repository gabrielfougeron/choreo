'''
ODE.py : Defines ODE-related things I designed I feel ought to be in scipy.

'''
import scipy

try:
    import numba 
    UseNumba = True
except ImportError:
    UseNumba = False

from choreo.scipy_plus.cython.ODE import ExplicitSymplecticIVP
from choreo.scipy_plus.cython.ODE import ExplicitSymplecticRKTable

from choreo.scipy_plus.cython.ODE import ImplicitSymplecticIVP
from choreo.scipy_plus.cython.ODE import ImplicitRKTable

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
    
    elif isinstance(rk,ImplicitRKTable):
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


if UseNumba:
    # Define decorators to make scipy.LowLevelCallable from python functions using numba
    
    default_numba_kwargs = {
        'nopython':True     ,
        'cache':True        ,
        'fastmath':True     ,
        'nogil':True        ,
    }

    def nb_jit_inplace_double_array(integrand_function, numba_kwargs = default_numba_kwargs):
        jitted_function = numba.jit(integrand_function, **numba_kwargs)

        #func(double t, double *x, double *res)
        @numba.cfunc(numba.types.void(numba.types.float64, numba.types.CPointer(numba.types.float64), numba.types.CPointer(numba.types.float64)))
        def wrapped(t, x, res):   
            jitted_function(t, x, res)
        
        return scipy.LowLevelCallable(wrapped.ctypes)
