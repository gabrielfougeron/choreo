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


def proj_to_zero(array, eps=1e-14):
    for idx in itertools.product(*[range(i)  for i in array.shape]):
        if abs(array[idx]) < eps:
            array[idx] = 0.

def main():
        
    TT = pyquickbench.TimeTrain(
        include_locs = False    ,
        align_toc_names = True  ,
        relative_timings = True  ,
    )

    # for config_name in tests.test_config.AllConfigNames_list:
    # for config_name in ['3q3q']:
    for config_name in ['6q6q']:
        print()
        # print("  OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO  ")
        print()
        print(config_name)
        # print()


        doit(config_name)
        
        TT.toc(config_name)

    print()
    print(TT)
        

def doit(config_name):
    
    NBS = tests.test_config.load_from_config_file(config_name)
    
    assert (NBS.BinSpaceRotIsId.all() == (NBS.BinSourceSegm != NBS.BinTargetSegm).all())
    
    for ibin in range(NBS.nbin_segm_unique):
    #     
        print(ibin, NBS.BinSpaceRotIsId[ibin], NBS.BinSourceSegm[ibin], NBS.BinTargetSegm[ibin])
        # print()
    #     

        # print(f'{self.nbin_segm_unique = }')
        # print(f'{self.nsegm = }')
        # print(f'{self.BinSourceSegm = }')
        # print(f'{self.BinTargetSegm = }')
        # print(f'{self.BinSpaceRotIsId = }')
        # print()

    
    
    
        
        



if __name__ == "__main__":
    main()
