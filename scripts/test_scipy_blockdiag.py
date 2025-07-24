import os
import numpy as np
import threadpoolctl
import choreo 
import scipy
import json

__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))

import pyquickbench



def main():
    
    def inv(n):
        
        for p in choreo.ActionSym.InvolutivePermutations(n):
            pass
    
    def gen_inv(n):
        for p in choreo.ActionSym.GeneralizedInvolutivePermutations(n, 2):
            pass
    
    all_funs = [
        inv     ,
        gen_inv ,
    ]
    
    n_bench = 10
    all_sizes = range(n_bench)
        
    pyquickbench.run_benchmark(
        all_sizes   ,
        all_funs    ,
        show = True ,
        MonotonicAxes = ["n"],
    )
    



if __name__ == "__main__":
    with threadpoolctl.threadpool_limits(limits=1):
        main()
