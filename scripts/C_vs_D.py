import os
import sys
__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(__PROJECT_ROOT__)

import choreo 
import time
import pyquickbench
import json
import numpy as np
import scipy
import itertools

np.set_printoptions(
    precision = 3,
    edgeitems = 10,
    # linewidth = 150,
    linewidth = 300,
    floatmode = "fixed",
)


def proj_to_zero(array, eps=1e-14):
    for idx in itertools.product(*[range(i)  for i in array.shape]):
        if abs(array[idx]) < eps:
            array[idx] = 0.

def main():
    
    doit()
    return 

    all_tests = [
        # '3q',
        # '3q3q',
        # '3q3qD',
        # '2q2q',
        # '4q4q',
        # '4q4qD',
        # '4q4qD3k',
        # '1q2q',
        # '5q5q',
        # '6q6q',
        # '2C3C',
        # '2D3D',   
        # '2C3C5k',
        # '2D3D5k',
        # '2D1',
        # '4C5k',
        # '4D3k',
        # '4C',
        # '4D',
        '3C',
        '3D',
        # '3D1',
        # '3C2k',
        # '3D2k',
        # '3Dp',
        # '3C4k',
        # '3D4k',
        # '3C5k',
        # '3D5k',
        # '3C101k',
        # '3D101k',
        # 'test_3D5k',
        # '3C7k2',
        # '3D7k2',
        # '6C',
        # '6D',
        # '6Ck5',
        # '6Dk5',
        # '5Dq',
        # '2C3C5C',
        # '3C_3dim',
        # '2D1_3dim', 
        # '3C11k',
        # '5q',
        # '5Dq_',
    ]

    TT = pyquickbench.TimeTrain(
        include_locs = False    ,
        align_toc_names = True  ,
    )

    for test in all_tests:
        print()
        print("  OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO  ")
        print()
        print(test)
        print()


        doit(test)
        
        TT.toc(test)

    print()
    # print(TT)
  

def proj_to_zero(array, eps=1e-14):
    for idx in itertools.product(*[range(i)  for i in array.shape]):
        if abs(array[idx]) < eps:
            array[idx] = 0.


def doit():
        
    eps = 1e-14
    
    all_tests = [
        '3C',
        '3D',
        # '3C2k',
    ]
        
    NBS_list = []
        
    for test in all_tests:
    
        Workspace_folder = os.path.join(__PROJECT_ROOT__, 'tests', 'NewSym_data', test)
        params_filename = os.path.join(Workspace_folder, 'choreo_config.json')
        
        with open(params_filename) as jsonFile:
            params_dict = json.load(jsonFile)

        all_kwargs = choreo.find.ChoreoLoadFromDict(params_dict, Workspace_folder, args_list=["geodim", "nbody", "mass", "charge", "inter_pow", "inter_pm", "Sym_list"])
        
        geodim = all_kwargs["geodim"]
        nbody = all_kwargs["nbody"]
        mass = all_kwargs["mass"]
        charge = all_kwargs["charge"]
        Sym_list = all_kwargs["Sym_list"]
        
        inter_law = scipy.LowLevelCallable.from_cython(choreo.cython._NBodySyst, "gravity_pot")
        # inter_law = scipy.LowLevelCallable.from_cython(choreo.cython._NBodySyst, "elastic_pot")

        NBS_list.append(choreo.cython._NBodySyst.NBodySyst(geodim, nbody, mass, charge, Sym_list, inter_law))
        
    NBSC = NBS_list[0]
    NBSD = NBS_list[1]
        
    nint = 12

    NBSD.nint = nint
    NBSC.nint = nint



    params_D = np.random.random((NBSD.nparams))
    

    segmpos_D = NBSD.params_to_segmpos(params_D)
    all_coeffs_D = NBSD.params_to_all_coeffs_noopt(params_D)        
    all_pos_D = scipy.fft.irfft(all_coeffs_D, axis=1, norm='forward')
    
    
    params_C = NBSC.all_coeffs_to_params_noopt(all_coeffs_D)
    segmpos_C = NBSC.params_to_segmpos(params_C)
    all_coeffs_C = NBSC.params_to_all_coeffs_noopt(params_C)        
    all_pos_C = scipy.fft.irfft(all_coeffs_C, axis=1, norm='forward')
    
    
    
    # 
    # print(np.linalg.norm(segmpos_D[0,:,:] - segmpos_C[0,:NBSD.segm_store,:]))
    # print(np.linalg.norm(segmpos_D[1,:,:] - segmpos_C[2,:NBSD.segm_store,:]))
    # print(np.linalg.norm(segmpos_D[2,:,:] - segmpos_C[1,:NBSD.segm_store,:]))

    print(segmpos_D[0,:,:])
    print(segmpos_C[0,:,:])




    print(np.linalg.norm(all_pos_D - all_pos_C))
    
    


    kin_nrg_D = NBSD.params_to_kin_nrg(params_D)
    pot_nrg_D = NBSD.params_to_pot_nrg(params_D)        
    

    kin_nrg_C = NBSC.params_to_kin_nrg(params_C)
    pot_nrg_C = NBSC.params_to_pot_nrg(params_C)



    print(kin_nrg_D-kin_nrg_C)
    print(pot_nrg_D-pot_nrg_C)






    
    
    
    
    






if __name__ == "__main__":
    main()
