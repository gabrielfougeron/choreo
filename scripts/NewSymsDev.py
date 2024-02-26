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
        # '3C',
        # '3D',
        '3D1',
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


def doit(config_name):
        
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
    
    if (inter_pow == -1.) and (inter_pm == 1) :
        inter_pot_fun = scipy.LowLevelCallable.from_cython(choreo.cython._NBodySyst, "gravity_pot")
    else:
        raise NotImplementedError
    
    NBS = choreo.cython._NBodySyst.NBodySyst(geodim, nbody, mass, charge, Sym_list, inter_pot_fun)
    
    
    
    
    NBS.nint_fac = 2
    
    params_buf = np.random.random((NBS.nparams))
    segmpos = NBS.params_to_segmpos(params_buf)
    all_pos = NBS.segmpos_to_all_pos_noopt(segmpos)
    
    params_buf_long = NBS.params_resize(params_buf, 100) 
    NBS.nint_fac = 100
    segmpos_long = NBS.params_to_segmpos(params_buf_long)
    all_pos_long = NBS.segmpos_to_all_pos_noopt(segmpos_long)
    
    choreo.NBodySyst_build.plot_given_2D(all_pos, 'all_pos.png')
    choreo.NBodySyst_build.plot_given_2D(all_pos_long, 'all_pos_long.png')    
    
    
    
    params_buf_long[:] = 0
    n=10
    params_buf_long[0:n] = np.random.random((n))
    segmpos_long = NBS.params_to_segmpos(params_buf_long)
    all_pos_long = NBS.segmpos_to_all_pos_noopt(segmpos_long)
    
    params_buf = NBS.params_resize(params_buf_long, 2) 
    NBS.nint_fac = 2    
    segmpos = NBS.params_to_segmpos(params_buf)
    all_pos = NBS.segmpos_to_all_pos_noopt(segmpos)
    
    choreo.NBodySyst_build.plot_given_2D(all_pos, 'all_pos_2.png')
    choreo.NBodySyst_build.plot_given_2D(all_pos_long, 'all_pos_long_2.png')



    return
    
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

    filename = os.path.join(Workspace_folder, config_name+'_graph_segm.pdf')
    choreo.cython._NBodySyst.PlotTimeBodyGraph(NBS.SegmGraph, nbody, NBS.nint_min, filename)





























if __name__ == "__main__":
    main()
