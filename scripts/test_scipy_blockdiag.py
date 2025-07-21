import os
import numpy as np
import threadpoolctl
import choreo 
import scipy
import json

__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))

def main():
    
    n = 20
    
    perm = np.random.permutation(n).astype(np.intp)
    cycles = choreo.ActionSym.CycleDecomposition(perm)
    
    print(perm)
    print()
    
    for cycle in cycles:
        print(cycle)
    
    
    
    



if __name__ == "__main__":
    with threadpoolctl.threadpool_limits(limits=1):
        main()
