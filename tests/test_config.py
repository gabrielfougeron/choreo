import attrs
import pytest
import inspect
import typing
import warnings
import functools
import numpy as np

@attrs.define
class float_tol:
    atol: float
    rtol: float

@attrs.define
class likelyhood():
    probable:       float
    not_unlikely:   float
    uncommon:       float
    unlikely:       float
    unbelievable:   float
    impossible:     float

@attrs.define
class dimension:
    all_geodims:    list[int]

@attrs.define
class nbody:
    all_nbodies:    list[int]
    
@attrs.define
class order:
    all_orders:    list[int]
    
@attrs.define
class ImplicitRKMethods:
    all_methods:    list[str]      
    
@attrs.define
class ImplicitRKMethodPairs:
    all_method_pairs:    list[tuple[str]]    

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

@pytest.fixture
def nonstiff_float64_likelyhood():
    return likelyhood(
        probable        = 1e-1 ,
        not_unlikely    = 1e-3 ,
        uncommon        = 1e-5 ,
        unlikely        = 1e-8 ,
        unbelievable    = 1e-12,
        impossible      = 1e-16,
    )

@pytest.fixture
def TwoD_only():
    return dimension(
        all_geodims = [2] ,
    )

@pytest.fixture
def Physical_dims():
    return dimension(
        all_geodims = [2, 3] ,
    )

@pytest.fixture
def Few_bodies():
    return nbody(
        all_nbodies = [2, 3, 4, 5] ,
    )

@pytest.fixture
def Dense_linalg_dims():
    return dimension(
        all_geodims = [2, 10, 20] ,
    )
    
@pytest.fixture
def Small_orders():
    return order(
        all_orders = list(range(2,11)) ,
    )
    
@pytest.fixture
def ClassicalImplicitRKMethods():
    return ImplicitRKMethods(
        all_methods = [
            "Gauss"         ,
            "Radau_IA"      ,
            "Radau_IIA"     ,
            "Radau_IB"      ,
            "Radau_IIB"     ,
            "Lobatto_IIIA"  ,
            "Lobatto_IIIB"  ,
            "Lobatto_IIIC"  ,
            "Lobatto_IIIC*" ,
            'Lobatto_IIID'  ,            
            'Lobatto_IIIS'  ,     
        ]
    )    
    
@pytest.fixture
def ClassicalImplicitRKMethodPairs():
    return ImplicitRKMethodPairs(
        all_method_pairs = [
            ("Gauss"        , "Gauss"           ),
            ("Radau_IB"     , "Radau_IB"        ),
            ("Radau_IIB"    , "Radau_IIB"       ),
            ("Lobatto_IIID" , "Lobatto_IIID"    ),
            ("Lobatto_IIIS" , "Lobatto_IIIS"    ),
            ("Lobatto_IIIA" , "Lobatto_IIIB"    ),
            ("Lobatto_IIIC" , "Lobatto_IIIC*"   ),
        ]   
    )

def ProbabilisticTest(RepeatOnFail = 10):

    def decorator(test):

        @functools.wraps(test)
        def wrapper(*args, **kwargs):
            
            try:
                return test(*args, **kwargs)

            except AssertionError:
                
                out_str = f"Probabilistic test failed. Running test again {RepeatOnFail} times."
                head_foot = '='*len(out_str)

                print('')
                print(head_foot)
                print(out_str)
                print(head_foot)
                print('')

                for i in range(RepeatOnFail):
                    res = test(*args, **kwargs)

                return res

        return wrapper
    
    return decorator

def RepeatTest(n = 10):

    def decorator(test):

        @functools.wraps(test)
        def wrapper(*args, **kwargs):
            
            for i in range(n):
                res = test(*args, **kwargs)

            return res

        return wrapper
    
    return decorator
