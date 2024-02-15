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


def test_create_NBodySyst(AllTestSyms):
    
    for config_filename in AllTestSyms:
        
        print(f"Config file : {config_filename}")
        
        Workspace_folder = os.path.join(__PROJECT_ROOT__, 'tests', 'NewSym_data', config_filename)
        params_filename = os.path.join(Workspace_folder, 'choreo_config.json')
        
        with open(params_filename) as jsonFile:
            params_dict = json.load(jsonFile)
    
        all_kwargs = choreo.find.ChoreoLoadFromDict(params_dict, Workspace_folder, args_list=["geodim", "nbody", "mass", "Sym_list"])
        
        geodim = all_kwargs["geodim"]
        nbody = all_kwargs["nbody"]
        mass = all_kwargs["mass"]
        Sym_list = all_kwargs["Sym_list"]
        
        choreo.cython._NBodySyst.NBodySyst(geodim, nbody, mass, Sym_list)
