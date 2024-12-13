import attrs
import pytest
import numpy as np

@attrs.define
class float_tol:
    atol: float
    rtol: float

@pytest.fixture
def float64_tols_strict():
    return float_tol(
        atol = np.finfo(np.float64).eps,
        rtol = np.finfo(np.float64).eps,
    )
    
@pytest.fixture
def float64_tols():
    return float_tol(
        atol = 1e-12,
        rtol = 1e-10,
    )
    
@pytest.fixture
def float64_tols_loose():
    return float_tol(
        atol = 1e-9,
        rtol = 1e-7,
    )

@pytest.fixture
def float32_tols_strict():
    return float_tol(
        atol = np.finfo(np.float32).eps,
        rtol = np.finfo(np.float32).eps,
    )

@pytest.fixture
def float32_tols():
    return float_tol(
        atol = 1e-5,
        rtol = 1e-3,
    )

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
        