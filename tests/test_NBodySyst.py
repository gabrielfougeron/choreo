import os
import sys

__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(__PROJECT_ROOT__)

import pytest
from test_config import *

import numpy as np
import scipy
import fractions
import json
import choreo
import choreo.cython._NBodySyst

def load_from_config_file(config_name):
    
    Workspace_folder = os.path.join(__PROJECT_ROOT__, 'tests', 'NewSym_data', config_name)
    params_filename = os.path.join(Workspace_folder, 'choreo_config.json')
    
    with open(params_filename) as jsonFile:
        params_dict = json.load(jsonFile)

    all_kwargs = choreo.find.ChoreoLoadFromDict(params_dict, Workspace_folder, args_list=["geodim", "nbody", "mass", "Sym_list"])
    
    geodim = all_kwargs["geodim"]
    nbody = all_kwargs["nbody"]
    mass = all_kwargs["mass"]
    Sym_list = all_kwargs["Sym_list"]
    
    return choreo.cython._NBodySyst.NBodySyst(geodim, nbody, mass, Sym_list)



def test_create_NBodySyst(AllConfigNames):
    
    for config_name in AllConfigNames:
        
        print(f"Config name : {config_name}")
        
        NBS = load_from_config_file(config_name)


def test_all_pos_to_segmpos(AllConfigNames, float64_tols):
    
    for config_name in AllConfigNames:
        
        print(f"Config name : {config_name}")
        
        NBS = load_from_config_file(config_name)
        
        NBS.nint_fac = 10
        params_buf = np.random.random((NBS.nparams))

        # Unoptimized version
        all_coeffs = NBS.params_to_all_coeffs_noopt(params_buf)        
        all_pos = scipy.fft.irfft(all_coeffs, axis=1)
        segmpos_noopt = NBS.all_pos_to_segmpos_noopt(all_pos)
        
        # Optimized version
        segmpos_cy = NBS.params_to_segmpos(params_buf)
        
        assert np.allclose(segmpos_noopt, segmpos_cy, rtol = float64_tols.rtol, atol = float64_tols.atol) 