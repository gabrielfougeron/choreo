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
import scipy.sparse
import itertools
import traceback

import tests.test_config

np.set_printoptions(
    precision = 3,
    edgeitems = 10,
    # linewidth = 150,
    linewidth = 300,
    floatmode = "fixed",
)

def main():
        
    TT = pyquickbench.TimeTrain(
        include_locs = False    ,
        align_toc_names = True  ,
        relative_timings = True  ,
    )

    for config_name in tests.test_config.AllConfigNames_list:
    # for config_name in ['3D']:
    # for config_name in ['6q6q']:
        print()
        # print("  OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO  ")
        print()
        print(config_name)
        # print()


        doit(config_name)
        
        TT.toc(config_name)

    print()
    # print(TT)
        

def doit(config_name):
    
    NBS = tests.test_config.load_from_config_file(config_name)
    
    # if NBS.TimeRev > 0:
    #     return
    
    print(f'{NBS.n_ODEinitparams = }')
    print(f'{NBS.n_ODEinitparams_pos = }')
    print(f'{NBS.n_ODEinitparams_mom = }')
    print()
    print(f'{NBS.n_ODEperdef_eqproj = }')
    print(f'{NBS.n_ODEperdef_eqproj_pos = }')
    print(f'{NBS.n_ODEperdef_eqproj_mom = }')

    
    # for isegm in range(NBS.nsegm):
    #     
    # # 
    #     assert NBS.PerDefBeg_Isegm[NBS.PerDefBeg_Isegm[isegm]] == isegm
    #     assert NBS.PerDefEnd_Isegm[NBS.PerDefEnd_Isegm[isegm]] == isegm
    #     
    #     print(isegm)
    #     print(NBS.PerDefBeg_Isegm[isegm])
    #     print(NBS.PerDefEnd_Isegm[isegm])
    #     # print(NBS.PerDefEnd_Isegm[NBS.PerDefBeg_Isegm[isegm]])
    #     # print(NBS.PerDefBeg_Isegm[NBS.PerDefEnd_Isegm[isegm]])
    #     print()
    #     
    #     # print(NBS.PerDefBeg_SpaceRotVel[isegm,:,:].T)
    #     # print(NBS.PerDefEnd_SpaceRotVel[isegm,:,:].T)
    # 
    
    
        
        



if __name__ == "__main__":
    main()
