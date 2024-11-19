import os
import sys

import numpy as np
import math as m

import choreo
import itertools
import pyquickbench

# for n in range(10):
#     print()
#     print(f'{n = }')
#     for a, b in choreo.ActionSym.TimeShifts(n):
#         print(a,b)

n = 3

ntests  = 10000

# for p in itertools.permutations(range(n)):
    # print(type(p))
    # for i in range(n):
        # if p[p[i]] != i:
            # print("False")
            # break
    # else:
        # print(True)


# for i in range(1,n):
for i in [n]:
    
    print()
    print(i)
    m = (i * (i-1)) // 2
    params = np.random.random(m)
        
    TT = pyquickbench.TimeTrain(include_locs=False,relative_timings=True)

    TT.toc("beg")
        
    for j in range(ntests):
        A = choreo.ActionSym.SurjectiveDirectSpaceRot(params)        
    TT.toc("fast")
    for j in range(ntests):
        A = choreo.ActionSym.SurjectiveDirectSpaceRotSlow(params)
    TT.toc("slow")
    print(TT)
        

# SurjectiveDirectSpaceRot