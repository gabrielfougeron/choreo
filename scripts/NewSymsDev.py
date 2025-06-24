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
    # for config_name in ['2D1_3dim']:
    # for config_name in ['6q6q']:
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
    
    print(NBS.params_basis_initpos)
    
    
    
        
        



if __name__ == "__main__":
    main()
