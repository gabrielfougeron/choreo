import os
import sys

import numpy as np
import math as m

import choreo

for n in range(10):
    print()
    print(f'{n = }')
    for a, b in choreo.ActionSym.TimeShifts(n):
        print(a,b)