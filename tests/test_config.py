import os
import sys

__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(__PROJECT_ROOT__)

import attrs
import pytest
import inspect
import typing
import warnings
import functools
import numpy as np
import json
import choreo
import scipy

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
def SymplecticImplicitRKMethodPairs():
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
    
@pytest.fixture
def SymmetricImplicitRKMethodPairs():
    return ImplicitRKMethodPairs(
        all_method_pairs = [
            ("Gauss"        , "Gauss"           ),
            ("Lobatto_IIIA" , "Lobatto_IIIA"    ),
            ("Radau_IB"     , "Radau_IIB"       ),
            ("Lobatto_IIIS" , "Lobatto_IIIS"    ),
            ("Lobatto_IIID" , "Lobatto_IIID"    ),
        ]   
    )
    
@pytest.fixture
def SymmetricSymplecticImplicitRKMethodPairs():
    return ImplicitRKMethodPairs(
        all_method_pairs = [
            ("Radau_IA"     , "Radau_IIA"       ),
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

@pytest.fixture
def AllConfigNames():
    return [
        '3q'        , '3q3q'    , '3q3qD'   , '2q2q'    , '4q4q'    , '4q4qD'   ,
        '4q4qD3k'   , '1q2q'    , '5q5q'    , '6q6q'    , '2C3C'    , '2D3D'    ,
        '2C3C5k'    , '2D3D5k'  , '2D1'     , '4C5k'    , '4D3k'    , '4C'      ,
        '4D'        , '3C'      , '3D'      , '3D1'     , '3C2k'    , '3D2k'    ,
        '3Dp'       , '3C4k'    , '3D4k'    , '3C5k'    , '3D5k'    , '3C101k'  ,
        '3D101k'    , '3C7k2'   , '3D7k2'   , '6C'      , '6D'      , '6Ck5'    ,
        '6Dk5'      , '5Dq'     , '2C3C5C'  , '3C_3dim' , '2D1_3dim', '3C11k'   ,
        '5q'        , 'uneven_nnpr'         , '2D2D'    , '2D1D1D'  , '2D2D5k'  ,
    ]


def load_from_config_file(config_name):
    
    Workspace_folder = os.path.join(__PROJECT_ROOT__, 'tests', 'NewSym_data', config_name)
    params_filename = os.path.join(Workspace_folder, 'choreo_config.json')
    
    with open(params_filename) as jsonFile:
        params_dict = json.load(jsonFile)

    all_kwargs = choreo.find.ChoreoLoadFromDict(params_dict, Workspace_folder, args_list=["geodim", "nbody", "mass", "charge", "inter_pow", "inter_pm", "Sym_list"])
    
    geodim = all_kwargs["geodim"]
    nbody = all_kwargs["nbody"]
    mass = all_kwargs["mass"]
    charge = all_kwargs["charge"]
    Sym_list = all_kwargs["Sym_list"]
    
    inter_pow = all_kwargs["inter_pow"]
    inter_pm = all_kwargs["inter_pm"]
    
    inter_pow = all_kwargs["inter_pow"]
    inter_pm = all_kwargs["inter_pm"]
    
    if (inter_pow == -1.) and (inter_pm == 1) :
        inter_law = scipy.LowLevelCallable.from_cython(choreo.cython._NBodySyst, "gravity_pot")
    else:
        raise NotImplementedError

    return choreo.cython._NBodySyst.NBodySyst(geodim, nbody, mass, charge, Sym_list, inter_law)

@pytest.fixture
def AllNBS(AllConfigNames):
    return {config_name:load_from_config_file(config_name) for config_name in AllConfigNames}

