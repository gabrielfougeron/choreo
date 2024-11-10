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
        
    all_tests = [
        '3q'        , '3q3q'    , '3q3qD'   , '2q2q'    , '4q4q'    , '4q4qD'   ,
        '4q4qD3k'   , '1q2q'    , '5q5q'    , '6q6q'    , '2C3C'    , '2D3D'    ,
        '2C3C5k'    , '2D3D5k'  , '2D1'     , '4C5k'    , '4D3k'    , '4C'      ,
        '4D'        , '3C'      , '3D'      , '3D1'     , '3C2k'    , '3D2k'    ,
        '3Dp'       , '3C4k'    , '3D4k'    , '3C5k'    , '3D5k'    , '3C101k'  ,
        '3D101k'    , '3C7k2'   , '3D7k2'   , '6C'      , '6D'      , '6Ck5'    ,
        '6Dk5'      , '5Dq'     , '2C3C5C'  , '3C_3dim' , '2D1_3dim', '3C7k2'   ,
        '5q'        , 'uneven_nnpr'         , '2D2D'    , '2D1D1D'  , '2D2D5k'  ,
        'complex_mass_charge'   , 'non_gravity_2dim'    , 'non_gravity_3dim'    ,
        '2D1_non_gravity'       ,'2D1_3dim_non_gravity' , '1Dx3'    , '1D1D'    ,
    ]
    
    DP_Wisdom_file = os.path.join(__PROJECT_ROOT__, "PYFFTW_wisdom.txt")
    choreo.find.Load_wisdom_file(DP_Wisdom_file)

    TT = pyquickbench.TimeTrain(
        include_locs = False    ,
        align_toc_names = True  ,
    )
    
    n_refine = 20

    for i_refine in tqdm.tqdm(range(n_refine)):
        
        for test in all_tests:
            doit(test, i_refine)
            
        TT.toc(i_refine)

        choreo.find.Write_wisdom_file(DP_Wisdom_file)

    print(TT)
    
    
def doit(config_name, i_refine):
        
    eps = 1e-14

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
    
    inter_pm = -1
    
    if (inter_pow == -1.) and (inter_pm == 1) :
        inter_law = scipy.LowLevelCallable.from_cython(choreo.cython._NBodySyst, "gravity_pot")
    else:
        inter_law = choreo.numba_funs.pow_inter_law(inter_pow/2, inter_pm)

    NBS = choreo.cython._NBodySyst.NBodySyst(geodim, nbody, mass, charge, Sym_list, inter_law)

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
