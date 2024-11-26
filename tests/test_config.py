import os
import sys
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

__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))

@attrs.define
class float_tol:
    atol: float
    rtol: float

@attrs.define
class likelihood():
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
def nonstiff_float64_likelihood():
    return likelihood(
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
        all_geodims = [2, 3, 4, 5] ,
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
def LoadPYFFTWWisdom():

    DP_Wisdom_file = os.path.join(__PROJECT_ROOT__, "PYFFTW_wisdom.txt")
    choreo.find.Load_wisdom_file(DP_Wisdom_file)
    
@pytest.fixture
def AllConfigNames(LoadPYFFTWWisdom):
    
    return [
        '3q'        , '3q3q'    , '3q3qD'   , '2q2q'    , '4q4q'    , '4q4qD'   ,
        '4q4qD3k'   , '1q2q'    , '5q5q'    , '6q6q'    , '2C3C'    , '2D3D'    ,
        '2C3C5k'    , '2D3D5k'  , '2D1'     , '4C5k'    , '4D3k'    , '4C'      ,
        '4D'        , '3C'      , '3D'      , '3D1'     , '3C2k'    , '3D2k'    ,
        '3Dp'       , '3C4k'    , '3D4k'    , '3C5k'    , '3D5k'    , '3C101k'  ,
        '3D101k'    , '3C7k2'   , '3D7k2'   , '6C'      , '6D'      , '6Ck5'    ,
        '6Dk5'      , '5Dq'     , '2C3C5C'  , '3C_3dim' , '2D1_3dim', '3C7k2'   ,
        '5q'        , 'uneven_nnpr'         , '2D2D'    , '2D1D1D'  , '2D2D5k'  ,
        'complex_mass_charge'   , 'non_gravity_2dim'    , 'non_gravity_3dim'    ,
        '2D1_non_gravity'       , '2D1_3dim_non_gravity', '1Dx3'    , '1D1D'    ,
        '2D3D4D'    , '3D7D'    , '1D1D1D'  , '2D3D4D'  , '3C4q4k'  , '3D4q4k'  ,
        '3DD'       , '5Dq_'    , '7D'      , '3D7D'    , '2D3D4D'  , '3Dp2'    ,
    ]    

@pytest.fixture
def AllConfigSymPairNames(LoadPYFFTWWisdom):
    """
    The first item of each pair has strictly more symmetries than the second.
    """
    
    return [
        ('3D'       , '3C'      ),
        ('3C2k'     , '3C'      ),
        ('3C4k'     , '3C'      ),
        ('3C5k'     , '3C'      ),
        ('3C101k'   , '3C'      ),
        ('3C7k2'    , '3C'      ),
        ('3C7k2'    , '3C'      ),
        ('4D'       , '4C'      ),
        ('4C5k'     , '4C'      ),
        ('6D'       , '6C'      ),
        ('6Ck5'     , '6C'      ),
        ('3D7k2'    , '3D'      ),
    ]

def params_from_config_file(config_name):
    
    Workspace_folder = os.path.join(__PROJECT_ROOT__, 'tests', 'NewSym_data', config_name)
    params_filename = os.path.join(Workspace_folder, 'choreo_config.json')
    
    with open(params_filename) as jsonFile:
        params_dict = json.load(jsonFile)

    return params_dict

def SymList_from_config_file(config_name):
    
    params_dict = params_from_config_file(config_name)
    
    return choreo.find.ChoreoLoadSymList(params_dict)

def load_from_config_file(config_name, override_args = {}):
    
    params_dict = params_from_config_file(config_name)
    
    Workspace_folder = os.path.join(__PROJECT_ROOT__, 'tests', 'NewSym_data', config_name)

    all_kwargs = choreo.find.ChoreoLoadFromDict(params_dict, Workspace_folder, args_list=["geodim", "nbody", "mass", "charge", "inter_pow", "inter_pm", "Sym_list"])

    for key, val in override_args.items():
        all_kwargs[key] = val
    
    geodim = all_kwargs["geodim"]
    nbody = all_kwargs["nbody"]
    mass = all_kwargs["mass"]
    charge = all_kwargs["charge"]
    Sym_list = all_kwargs["Sym_list"]
    
    inter_pow = all_kwargs["inter_pow"]
    inter_pm = all_kwargs["inter_pm"]
    
    if (inter_pow == -1.) and (inter_pm == 1) :
        inter_law_str = "gravity_pot"
        inter_law_param_dict = None
    else:
        inter_law_str = "power_law_pot"
        inter_law_param_dict = {'n': inter_pow, 'alpha': inter_pm}

    NBS = choreo.NBodySyst(
        geodim, nbody, mass, charge, Sym_list,
        inter_law_str = inter_law_str, inter_law_param_dict = inter_law_param_dict
    )
    
    return NBS

@pytest.fixture
def AllNBS(AllConfigNames):
    return {config_name:load_from_config_file(config_name) for config_name in AllConfigNames}

@pytest.fixture
def AllNBSPairs(AllConfigSymPairNames):
    return {config_name:(load_from_config_file(config_name[0]),load_from_config_file(config_name[1])) for config_name in AllConfigSymPairNames}

@pytest.fixture
def AllNBS_nozerodiv(AllConfigNames):
    override_args = {"inter_pow" : 1.}
    return {config_name:load_from_config_file(config_name, override_args) for config_name in AllConfigNames}

@pytest.fixture
def AllNBSPairs_nozerodiv(AllConfigSymPairNames):
    override_args = {"inter_pow" : 1.}
    return {config_name:(load_from_config_file(config_name[0], override_args),load_from_config_file(config_name[1], override_args)) for config_name in AllConfigSymPairNames}

@pytest.fixture
def AllSymList(AllConfigNames):
    return {config_name:SymList_from_config_file(config_name) for config_name in AllConfigNames}

