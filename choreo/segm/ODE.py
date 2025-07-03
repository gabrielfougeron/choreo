'''
ODE.py : Defines ODE-related things.

'''
import scipy
from choreo import NUMBA_AVAILABLE

if NUMBA_AVAILABLE:
    import numba

from choreo.segm.cython.ODE import ExplicitSymplecticRKTable
from choreo.segm.cython.ODE import ExplicitSymplecticIVP

from choreo.segm.cython.ODE import ImplicitRKTable
from choreo.segm.cython.ODE import ImplicitSymplecticIVP

def SymplecticIVP(
    fun         ,
    gun         ,
    t_span      ,
    xo          ,
    vo          ,
    rk          ,
    **kwargs    ,
): 

    if isinstance(rk, ExplicitSymplecticRKTable):
        return ExplicitSymplecticIVP(
            fun         ,
            gun         ,
            t_span      ,
            xo          ,
            vo          ,
            rk          ,
            **kwargs    ,
        )
    
    elif isinstance(rk, ImplicitRKTable):
        return ImplicitSymplecticIVP(
            fun                             ,
            gun                             ,
            t_span                          ,
            xo                              ,
            vo                              ,
            rk_x = rk                       ,
            rk_v = rk.symplectic_adjoint()  ,
            **kwargs                        ,
        )

    else:
        raise ValueError(f'Unknown rk type : {type(rk)}')

if NUMBA_AVAILABLE:
    # Define decorators to make scipy.LowLevelCallable from python functions using numba
    
    default_numba_kwargs = {
        'nopython':True     ,
        # 'cache':True        ,
        'fastmath':True     ,
        'nogil':True        ,
    }

    def nb_jit_c_fun_pointer(py_fun, numba_kwargs = default_numba_kwargs):

        #func(double t, double *x, double *res)
        sig = numba.types.void(numba.types.float64, numba.types.CPointer(numba.types.float64), numba.types.CPointer(numba.types.float64))

        cfunc_fun = numba.cfunc(sig, **numba_kwargs)(py_fun)
        
        return scipy.LowLevelCallable(cfunc_fun.ctypes)
        