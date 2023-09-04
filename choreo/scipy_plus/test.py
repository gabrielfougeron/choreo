import math as m
import numpy as np
import scipy
from choreo.scipy_plus.ODE import SymplecticIVP, ImplicitSymplecticIVP
from choreo.scipy_plus.SegmQuad import ComputeQuadrature
from choreo.scipy_plus.SegmQuad import IntegrateOnSegment


def Quad_cpte_error_on_test(
    fun_name,
    quad_method,
    quad_nsteps,
    nint,
):

    if fun_name == "exp" :
        # WOLFRAM
        # f(x) = y*exp(y*x)
        # F(x) = exp(y*x)

        test_ndim = 20

        fun = lambda x: np.array([y*m.exp(y*x) for y in range(test_ndim)])
        Fun = lambda x: np.array([m.exp(y*x) for y in range(test_ndim)])
        
        x_span = (0.,1.)
        exact = Fun(x_span[1]) - Fun(x_span[0])


    quad = ComputeQuadrature(quad_method, quad_nsteps)

    approx = IntegrateOnSegment(
        fun = fun       ,
        ndim = test_ndim,
        x_span = x_span ,
        quad = quad     ,
        nint = nint     ,
    )

    error = np.linalg.norm(approx-exact)/np.linalg.norm(exact)

    return error
 
def ODE_define_test(eq_name):
     
    if eq_name == "y'' = -y" :
        # WOLFRAM
        # y'' = - y
        # y(x) = A cos(x) + B sin(x)

        test_ndim = 2

        ex_sol = lambda t : np.array( [ np.cos(t) , np.sin(t),-np.sin(t), np.cos(t) ]  )

        fun = lambda t,y:   np.asarray(y)
        gun = lambda t,x:  -np.asarray(x)

    if eq_name == "y'' = - exp(y)" :
        # WOLFRAM
        # y'' = - exp(y)
        # y(x) = - 2 * ln( cosh(t / sqrt(2) ))

        test_ndim = 1

        invsqrt2 = 1./np.sqrt(2.)
        sqrt2 = np.sqrt(2.)
        ex_sol = lambda t : np.array( [ -2*np.log(np.cosh(invsqrt2*t)) , -sqrt2*np.tanh(invsqrt2*t) ]  )

        fun = lambda t,y:  np.array(y)
        gun = lambda t,x: -np.exp(x)

    if eq_name == "y'' = xy" :

        # Solutions: Airy functions
        # Nonautonomous linear test case

        test_ndim = 2

        def ex_sol(t):

            ai, aip, bi, bip = scipy.special.airy(t)

            return np.array([ai,bi,aip,bip])

        fun = lambda t,y: np.array(y)
        gun = lambda t,x: np.array([t*x[0],t*x[1]],dtype=np.float64)
        
    if eq_name == "y' = Az; z' = By" :

        test_ndim = 10

        A = np.diag(np.array(range(test_ndim)))
        B = np.identity(test_ndim)

        AB = np.zeros((2*test_ndim,2*test_ndim))
        AB[0:test_ndim,test_ndim:2*test_ndim] = A
        AB[test_ndim:2*test_ndim,0:test_ndim] = B

        yo = np.array(range(test_ndim))
        zo = np.array(range(test_ndim))

        yzo = np.zeros(2*test_ndim)
        yzo[0:test_ndim] = yo
        yzo[test_ndim:2*test_ndim] = zo

        def ex_sol(t):
            return scipy.linalg.expm(t*AB).dot(yzo)

        fun = lambda t,z: A.dot(z)
        gun = lambda t,y: B.dot(y)

    return fun, gun, ex_sol, test_ndim
 

def ODE_cpte_error_on_test(
    eq_name     ,
    rk_method   ,
    nint        ,
    **kwargs    ,
):

    fun, gun, ex_sol, test_ndim = ODE_define_test(eq_name)

    t_span = (0.,np.pi)

    ex_init  = ex_sol(t_span[0])
    ex_final = ex_sol(t_span[1])

    x0 = ex_init[0          :  test_ndim].copy()
    v0 = ex_init[test_ndim  :2*test_ndim].copy()

    xf,vf = SymplecticIVP(
        fun             ,
        gun             ,
        t_span          ,
        x0              ,
        v0              ,
        rk = rk_method  ,
        nint = nint     ,
        **kwargs        ,
    )
        
    sol = np.ascontiguousarray(np.concatenate((xf,vf),axis=0).reshape(2*test_ndim))
    error = np.linalg.norm(sol-ex_final)/np.linalg.norm(ex_final)

    return error

def ISPRK_ODE_cpte_error_on_test(
    eq_name     ,
    rk_x        ,
    rk_v        ,
    nint        ,
    **kwargs    ,
):

    fun, gun, ex_sol, test_ndim = ODE_define_test(eq_name)

    t_span = (0.,np.pi)

    ex_init  = ex_sol(t_span[0])
    ex_final = ex_sol(t_span[1])

    x0 = ex_init[0          :  test_ndim].copy()
    v0 = ex_init[test_ndim  :2*test_ndim].copy()

    xf,vf = ImplicitSymplecticIVP(
        fun             ,
        gun             ,
        t_span          ,
        x0              ,
        v0              ,
        rk_x = rk_x     ,
        rk_v = rk_v     ,
        nint = nint     ,
        **kwargs        ,
    )
                
    sol = np.ascontiguousarray(np.concatenate((xf,vf),axis=0).reshape(2*test_ndim))
    error = np.linalg.norm(sol-ex_final)/np.linalg.norm(ex_final)

    return error