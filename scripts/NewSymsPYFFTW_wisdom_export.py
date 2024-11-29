import os
import sys
__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(__PROJECT_ROOT__)

import choreo 
import tqdm
import pyquickbench
import json
import scipy

def main():
        
    # all_tests = [
    #     '3q'        , '3q3q'    , '3q3qD'   , '2q2q'    , '4q4q'    , '4q4qD'   ,
    #     '4q4qD3k'   , '1q2q'    , '5q5q'    , '6q6q'    , '2C3C'    , '2D3D'    ,
    #     '2C3C5k'    , '2D3D5k'  , '2D1'     , '4C5k'    , '4D3k'    , '4C'      ,
    #     '4D'        , '3C'      , '3D'      , '3D1'     , '3C2k'    , '3D2k'    ,
    #     '3Dp'       , '3C4k'    , '3D4k'    , '3C5k'    , '3D5k'    , '3C101k'  ,
    #     '3D101k'    , '3C7k2'   , '3D7k2'   , '6C'      , '6D'      , '6Ck5'    ,
    #     '6Dk5'      , '5Dq'     , '2C3C5C'  , '3C_3dim' , '2D1_3dim', '3C7k2'   ,
    #     '5q'        , 'uneven_nnpr'         , '2D2D'    , '2D1D1D'  , '2D2D5k'  ,
    #     'complex_mass_charge'   , 'non_gravity_2dim'    , 'non_gravity_3dim'    ,
    #     '2D1_non_gravity'       , '2D1_3dim_non_gravity', '1Dx3'    , '1D1D'    ,
    #     '2D3D4D'    , '3D7D'    , '1D1D1D'  , '2D3D4D'  , '3C4q4k'  , '3D4q4k'  ,
    #     '3DD'       , '5Dq_'    , '7D'      , '3D7D'    , '2D3D4D'  , '3Dp2'    ,
    # ]            
    all_tests = [
        '3C'  
    ]    
        
    Wisdom_file = os.path.join(__PROJECT_ROOT__, "PYFFTW_wisdom.json")
    choreo.find.Load_wisdom_file(Wisdom_file)

    TT = pyquickbench.TimeTrain(
        include_locs = False    ,
        align_toc_names = True  ,
    )
    
    # n_refine = 7 + 6
    n_refine = 22

    for i_refine in tqdm.tqdm(range(n_refine)):
        
        for test in all_tests:
            Workspace_folder = os.path.join(__PROJECT_ROOT__, 'tests', 'NewSym_data', test)
            doit(Workspace_folder, i_refine)
            choreo.find.Write_wisdom_file(Wisdom_file)              
            
        # Workspace_folder = os.path.join(__PROJECT_ROOT__, 'Sniff_all_sym')
        # doit(Workspace_folder, i_refine)
        # choreo.find.Write_wisdom_file(Wisdom_file)     
            
        TT.toc(i_refine)

    print(TT)
    
    
def doit(Workspace_folder, i_refine):
        
    eps = 1e-14
    
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

    # NBS.fftw_planner_effort = 'FFTW_ESTIMATE'
    # NBS.fftw_planner_effort = 'FFTW_MEASURE'
    # NBS.fftw_planner_effort = 'FFTW_PATIENT'
    NBS.fftw_planner_effort = 'FFTW_EXHAUSTIVE'
    
    NBS.fftw_wisdom_only = False
    # NBS.fftw_wisdom_only = True
    
    NBS.fftw_nthreads = 1
    
    NBS.fft_backend = 'fftw'

    NBS.nint_fac = 2**i_refine


if __name__ == "__main__":
    main()
