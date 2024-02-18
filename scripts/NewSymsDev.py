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

np.set_printoptions(
    precision = 3,
    edgeitems = 10,
    # linewidth = 150,
    linewidth = 300,
    floatmode = "fixed",
)

def main():
        
    all_tests = [
        '3q',
        '3q3q',
        '3q3qD',
        '2q2q',
        '4q4q',
        '4q4qD',
        '4q4qD3k',
        '1q2q',
        '5q5q',
        '6q6q',
        '2C3C',
        '2D3D',   
        '2C3C5k',
        '2D3D5k',
        '2D1',
        '4C5k',
        '4D3k',
        '4C',
        '4D',
        '3C',
        '3D',
        '3D1',
        '3C2k',
        '3D2k',
        '3Dp',
        '3C4k',
        '3D4k',
        '3C5k',
        '3D5k',
        '3C101k',
        '3D101k',
        'test_3D5k',
        '3C7k2',
        '3D7k2',
        '6C',
        '6D',
        '6Ck5',
        '6Dk5',
        '5Dq',
        '2C3C5C',
        '3C_3dim',
        '2D1_3dim', 
        '3C11k',
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
    print(TT)
  



def doit(config_name):
        
    eps = 1e-14

    Workspace_folder = os.path.join(__PROJECT_ROOT__, 'tests', 'NewSym_data', config_name)
    params_filename = os.path.join(Workspace_folder, 'choreo_config.json')
    
    with open(params_filename) as jsonFile:
        params_dict = json.load(jsonFile)

    all_kwargs = choreo.find.ChoreoLoadFromDict(params_dict, Workspace_folder, args_list=["geodim", "nbody", "mass", "Sym_list"])
    
    geodim = all_kwargs["geodim"]
    nbody = all_kwargs["nbody"]
    mass = all_kwargs["mass"]
    Sym_list = all_kwargs["Sym_list"]
    
    NBS = choreo.cython._NBodySyst.NBodySyst(geodim, nbody, mass, Sym_list)
    
    NBS.nint_fac = 1

    eps = 1e-12
    

    params_buf = np.random.random((NBS.nparams))
    # NBS.project_params_buf(params_buf)
    
    print(f'{NBS.nparams = }') 
    print(f'{NBS.nparams_incl_o = }') 
    print(f'{NBS.nrem = }') 
    
    # return
    # print(params_buf)
    
    print(NBS.nrem)
    for il in range(NBS.nloop):
        print(NBS.co_in(il))
    
    # return

    
    
    
    # print(all_coeffs)
    # return
    
    
    
    
    
    
    all_coeffs = NBS.params_to_all_coeffs_noopt(params_buf)  
    all_pos = scipy.fft.irfft(all_coeffs, axis=1)
    # 
    # segmpos_noopt = NBS.all_pos_to_segmpos_noopt(all_pos)
    # segmpos_cy = NBS.params_to_segmpos(params_buf)
    # 
    # print(np.linalg.norm(segmpos_noopt - segmpos_cy))
    # assert np.linalg.norm(segmpos_noopt - segmpos_cy) < eps
  

    
    all_coeffs_rt = scipy.fft.rfft(all_pos, axis=1)
    print(np.linalg.norm(all_coeffs_rt - all_coeffs))
    assert (np.linalg.norm(all_coeffs - all_coeffs_rt)) < eps    


    params_buf_rt = NBS.all_coeffs_to_params_noopt(all_coeffs_rt)
    print(np.linalg.norm(params_buf - params_buf_rt))
    assert (np.linalg.norm(params_buf - params_buf_rt)) < eps
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    nparam_nosym = geodim * NBS.nint * nbody
    nparam_tot = NBS.nparams_incl_o

    print('*****************************************')
    print('')
    # print(f'{Identity_detected = }')
    print(f'All binary transforms are identity: {NBS.All_BinSegmTransformId}')
    
    
    print()
    print(f"total binary segment interaction count: {NBS.nbin_segm_tot}")
    print(f"unique binary segment interaction count: {NBS.nbin_segm_unique}")
    print(f'{NBS.nsegm = }')


    print(f"ratio of total to unique binary interactions : {NBS.nbin_segm_tot  / NBS.nbin_segm_unique}")
    print(f'ratio of integration intervals to segments : {(nbody * NBS.nint_min) / NBS.nsegm}')
    print(f"ratio of parameters before and after constraints: {nparam_nosym / nparam_tot}")

    reduction_ratio = nparam_nosym / nparam_tot

    assert abs((nparam_nosym / nparam_tot)  - reduction_ratio) < eps
    
    if NBS.All_BinSegmTransformId:
        assert abs(NBS.nbin_segm_tot  / NBS.nbin_segm_unique  - reduction_ratio) < eps
        assert abs((nbody * NBS.nint_min) / NBS.nsegm - reduction_ratio) < eps


    return

    dirname = os.path.split(store_folder)[0]
    symname = os.path.split(dirname)[1]
    filename = os.path.join(dirname, f'{symname}_graph_segm.pdf')

    PlotTimeBodyGraph(NBS.SegmGraph, nbody, NBS.nint_min, filename)





























if __name__ == "__main__":
    main()
