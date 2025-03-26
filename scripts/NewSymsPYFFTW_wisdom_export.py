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
    #     '3C'  
    # ]    
    
    all_tests = [name for name in os.listdir(os.path.join(__PROJECT_ROOT__, 'tests', 'NewSym_data'))]
        
    Wisdom_file = os.path.join(__PROJECT_ROOT__, "PYFFTW_wisdom.json")
    choreo.find.Load_wisdom_file(Wisdom_file)

    TT = pyquickbench.TimeTrain(
        include_locs = False    ,
        align_toc_names = True  ,
    )
    
    # n_refine = 7 + 6
    n_refine = 24

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
