import attrs
import pytest
import numpy as np

@attrs.define
class float_tol:
    atol: float
    rtol: float

@pytest.fixture
def float64_tols_strict():
    """ Strict double-precision tolerance.
    
    To be used for testing whether two double-precision floating point numbers are equal within maximum accuracy.
    
    >>> atol = 2.220446049250313e-16
    >>> rtol = 2.220446049250313e-16
    
    """
    return float_tol(
        atol = np.finfo(np.float64).eps,
        rtol = np.finfo(np.float64).eps,
    )
    
@pytest.fixture
def float64_tols():
    """ Regular double-precision tolerance.
    
    To be used for testing whether two double-precision floating point numbers are almost equal when only small numerical errors are expected.
    
    >>> atol = 1e-12
    >>> rtol = 1e-10
    
    """
    return float_tol(
        atol = 1e-12,
        rtol = 1e-10,
    )
    
@pytest.fixture
def float64_tols_loose():
    """ Loose double-precision tolerance.
    
    To be used for testing whether two double-precision floating point numbers are almost equal when large numerical errors are expected.
    
    >>> atol = 1e-9
    >>> rtol = 1e-7
    
    """
    return float_tol(
        atol = 1e-9,
        rtol = 1e-7,
    )

@pytest.fixture
def float32_tols_strict():
    """ Strict simple-precision tolerance.
    
    To be used for testing whether two simple-precision floating point numbers are equal within maximum accuracy.
    
    >>> atol = 1.1920929e-07
    >>> rtol = 1.1920929e-07
    
    """
    return float_tol(
        atol = np.finfo(np.float32).eps,
        rtol = np.finfo(np.float32).eps,
    )

@pytest.fixture
def float32_tols():
    """ Regular simple-precision tolerance.
    
    To be used for testing whether two simple-precision floating point numbers are almost equal when numerical errors are expected.
    
    >>> atol = 1.1920929e-07
    >>> rtol = 1.1920929e-07
    
    """
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
        dx = np.array(np.random.rand(*xo.shape), dtype=xo.dtype)
    
    fo = fun(xo)
    if vectorize:
        dfdx_exact = gradfun(xo,dx.reshape(-1,1)).reshape(-1)
    else:
        dfdx_exact = gradfun(xo,dx)
    dfdx_exact_magn = np.linalg.norm(dfdx_exact)
    
    error_list = []
    for eps in epslist:
        dfdx_FD = compute_FD(fun,xo,dx,eps,fo=fo,order=order)
        
        # print()
        # print(f'{eps = }')
        # print(f'{dfdx_FD = }')
        # print(f'{dfdx_exact = }')
        # print(f'{dfdx_FD / dfdx_exact = }')
        
        if relative:
            error = 2 * np.linalg.norm(dfdx_FD - dfdx_exact) / (dfdx_exact_magn + np.linalg.norm(dfdx_FD))
        else:
            error = np.linalg.norm(dfdx_FD - dfdx_exact)
            
        error_list.append(error)
    
    return np.array(error_list)
        
def inarray(val, arr, rtol, atol):
     
    minval = np.min(np.abs(arr-val))
    nrm = np.linalg.norm(arr, ord = np.inf)
    
    return (minval < atol) and (minval < rtol * nrm)
    