import os
import threadpoolctl

import scipy
import json
import math
__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))


import numpy as np
import choreo
Sym = choreo.ActionSym.Random(nbody = 10, geodim = 4)
print(Sym.signature.ActionSym == Sym)


print(math.lcm(2,3))

l = [2,3]

print(math.lcm(*l))

choreo.DiscreteActionSymSignature.DirectTimeSignatures(3,2,5)

nbody = 1
geodim = 3
max_order = 10

nsig = 0


for SymSig in choreo.DiscreteActionSymSignature.DirectTimeSignatures(nbody, geodim, max_order):
    
    
    
    assert SymSig.IsWellFormed()
    
    nsig += 1
    
    # print(SymSig)
    # print("OK")
    
print(f'{nsig = }')