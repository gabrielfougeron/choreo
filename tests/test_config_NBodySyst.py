import os
import json
import choreo

__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
Wisdom_file = os.path.join(__PROJECT_ROOT__, "PYFFTW_wisdom.json")
choreo.find.Load_wisdom_file(Wisdom_file)

Physical_dims = [2, 3, 4, 5]
Few_bodies = [2, 3, 4, 5]
Dense_linalg_dims = [2, 10, 20] 

AllConfigNames_list = [
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

"""
The first item of each pair has strictly more symmetries than the second.
"""
AllConfigSymPairNames_list = [
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
    
    # NBS.fft_backend = 'scipy'
    # NBS.fft_backend = 'mkl'
    # NBS.fft_backend = 'ducc'

    # NBS.fft_backend = 'fftw'    
    # NBS.fftw_planner_effort = 'FFTW_ESTIMATE'
    # NBS.fftw_wisdom_only = False
    # NBS.fftw_nthreads = 1
    
    return NBS

NBS_dict =  {config_name:load_from_config_file(config_name) for config_name in AllConfigNames_list}
NBS_nozerodiv_dict = {config_name:load_from_config_file(config_name, {"inter_pow" : 1.}) for config_name in AllConfigNames_list}
NBS_pairs_dict = {f'{config_name[0]}-{config_name[1]}':(load_from_config_file(config_name[0]),load_from_config_file(config_name[1])) for config_name in AllConfigSymPairNames_list}
SymList_dict = {config_name:SymList_from_config_file(config_name) for config_name in AllConfigNames_list}

def load_from_solution_file(file_basename):
    
    Workspace_folder = os.path.join(__PROJECT_ROOT__, 'tests', 'NewSym_sols')
    
    full_in_file_basename = os.path.join(Workspace_folder, file_basename)
    
    NBS, segmpos = choreo.NBodySyst.FromSolutionFile(full_in_file_basename)
    params_buf = NBS.segmpos_to_params(segmpos)
    
    return NBS, params_buf

Sols_dict = {}
for thefile in os.listdir(os.path.join(__PROJECT_ROOT__, 'tests', 'NewSym_sols')):

    file_basename, ext = os.path.splitext(thefile)
    
    if ext == '.json':
        
        Sols_dict[file_basename] = load_from_solution_file(file_basename)