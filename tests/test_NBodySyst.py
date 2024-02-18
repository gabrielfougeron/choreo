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
        
        NBS.AssertAllSegmGenConstraintsAreRespected(all_pos)
        NBS.AssertAllBodyConstraintAreRespected(all_pos)
        
@ProbabilisticTest()
def test_capture_co(AllConfigNames):
    
    for config_name in AllConfigNames:
            
        print(f"Config name : {config_name}")

        NBS = load_from_config_file(config_name)

        NBS.nint_fac = 10
        params_buf = np.random.random((NBS.nparams))
        
    
    nnz = [[] for il in range(NBS.nloop)]
    for il in range(NBS.nloop):
        
        nnz_k = NBS.nnz_k(il)
        params_basis = NBS.params_basis(il)
        
        eps = 1e-12
        if nnz_k.shape[0] > 0:
            if nnz_k[0] == 0:
                
                for iparam in range(params_basis.shape[2]):
                    if np.linalg.norm(params_basis[:,0,iparam].imag) > eps:
                        nnz[il].append(iparam)

        for il in range(NBS.nloop):
            co_in = NBS.co_in(il)
            for iparam in range(co_in.shape[0]):            
                assert co_in[iparam] == (iparam in nnz[il])
