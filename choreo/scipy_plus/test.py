import math as m
import numpy as np
import scipy
from choreo.scipy_plus.ODE import SymplecticIVP, ImplicitSymplecticIVP
from choreo.scipy_plus.multiprec_tables import ComputeQuadrature
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


    quad = ComputeQuadrature(quad_nsteps, method = quad_method)

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
        
        def fgun(t, xy):
            
            fxy = np.empty(2*test_ndim)
            fxy[0] =  xy[2]
            fxy[1] =  xy[3]
            fxy[2] = -xy[0]
            fxy[3] = -xy[1]
            
            return fxy

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
        
        def fgun(t, xy):
            
            fxy = np.empty(2*test_ndim)
            fxy[0] = xy[1]
            fxy[1] = -np.exp(xy[0])

            return fxy

    if eq_name == "y'' = xy" :

        # Solutions: Airy functions
        # Nonautonomous linear test case

        test_ndim = 2

        def ex_sol(t):

            ai, aip, bi, bip = scipy.special.airy(t)

            return np.array([ai,bi,aip,bip])

        fun = lambda t,y: np.array(y)
        gun = lambda t,x: np.array([t*x[0],t*x[1]],dtype=np.float64)
        
        def fgun(t, xy):
            
            fxy = np.empty(2*test_ndim)
            fxy[0] =  xy[2]
            fxy[1] =  xy[3]
            fxy[2] = t*xy[0]
            fxy[3] = t*xy[1]
            
            return fxy
        
        
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
        
        def fgun(t, xy):
            
            fxy = np.empty(2*test_ndim)
            fxy[:test_ndim] =  A.dot(xy[test_ndim:])
            fxy[test_ndim:] =  B.dot(xy[:test_ndim])

            return fxy

    return fun, gun, fgun, ex_sol, test_ndim
 
def ODE_cpte_error_on_test(
    eq_name     ,
    rk_method   ,
    nint        ,
    **kwargs    ,
):

    fun, gun, fgun, ex_sol, test_ndim = ODE_define_test(eq_name)

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

    fun, gun, fgun, ex_sol, test_ndim = ODE_define_test(eq_name)

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

def scipy_ODE_cpte_error_on_test(
    eq_name     ,
    method      ,
    nint        ,
    **kwargs    ,
):

    fun, gun, fgun, ex_sol, test_ndim = ODE_define_test(eq_name)

    t_span = (0.,np.pi)
    
    max_step = (t_span[1] - t_span[0]) / nint

    ex_init  = ex_sol(t_span[0])
    ex_final = ex_sol(t_span[1])

    bunch = scipy.integrate.solve_ivp(
        fun = fgun                      ,
        t_span = t_span                 ,
        y0 = ex_init                    ,
        method = method                 ,
        t_eval = np.array([t_span[1]])  ,
        first_step = max_step           ,
        max_step = max_step             ,
        atol = 1.             ,
        rtol = 1.             ,
    )

    error = np.linalg.norm(bunch.y[:,0]-ex_final)/np.linalg.norm(ex_final)

    return error

def compute_FD(fun,xo,dx,eps,fo=None,order=1):
    
    if fo is None:
        fo = fun(xo)
        
    if order == 1:
        
        xp = xo + eps*dx
        fp = fun(xp)
        dfdx = (fp-fo)/eps
        
    elif (order == 2):
        
        xp = xo + eps*dx
        fp = fun(xp)        
        xm = xo - eps*dx
        fm = fun(xm)
        dfdx = (fp-fm)/(2*eps)
        
    else:
        
        raise ValueError(f"Invalid order {order}")

    return dfdx

def compare_FD_and_exact_grad(fun, gradfun, xo, dx=None, epslist=None, order=1, vectorize=True, relative=True):
    
    if epslist is None:
        epslist = [10**(-i) for i in range(16)]
        
    if dx is None:
        dx = np.array(np.random.rand(*xo.shape), dtype= xo.dtype)
    
    fo = fun(xo)
    if vectorize:
        dfdx_exact = gradfun(xo,dx.reshape(-1,1)).reshape(-1)
    else:
        dfdx_exact = gradfun(xo,dx)
    dfdx_exact_magn = np.linalg.norm(dfdx_exact)
    
    error_list = []
    for eps in epslist:
        dfdx_FD = compute_FD(fun,xo,dx,eps,fo=fo,order=order)
        
        if relative:
            error = np.linalg.norm(dfdx_FD - dfdx_exact) / dfdx_exact_magn 
        else:
            error = np.linalg.norm(dfdx_FD - dfdx_exact)
            
        error_list.append(error)
    
    return np.array(error_list)
        