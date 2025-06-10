import numpy as np
import cmath
import math

import kepler
import pyquickbench


# Solves E - e * sin(E) = M for E given M and e with 0 <= e < 1
def kepler_solve(M, e, tol=1e-12):
    
    assert e >= 0.
    assert e < 1.
    
    twopi = 2*math.pi
    
    R = math.floor(M / twopi)
    dEo = R * twopi
    
    Mo = M - dEo
    
    E = Mo
    fE = Mo + e * math.sin(E)
    r = fE - E
    
    n = 1
    
    while np.abs(r) > tol:
            
        E = fE
        fE = Mo + e * math.sin(E)
        r = fE - E
        
        n += 1
    
    return dEo + fE, r, n


e = 0.9

tol = 1e-12

N = 10000

Mo = 0.
arr_M = np.linspace(0., math.pi, num=N, endpoint=True) + Mo

arr_r = np.empty(N, dtype=np.float64)
arr_n = np.empty(N, dtype=np.intp)

arr_diff = np.empty(N, dtype=np.float64)

# for i in range(N):
#             
#     E, arr_r[i], arr_n[i] = kepler_solve(arr_M[i] , e, tol)
#     Ek = kepler.solve(arr_M[i], e)
#     
#     arr_diff[i] = E - Ek
#     
#     
#     
#     
# 
# imax = arr_n.argmax()
# print(arr_n[imax])
# print(arr_r[imax])
# print()
#     
# imax = abs(arr_r).argmax()
# print(arr_n[imax])
# print(arr_r[imax])
#     
# print()    
#     
# imax = abs(arr_diff).argmax()
# print(abs(arr_diff[imax]))
# print(arr_n[imax])
# print(arr_r[imax])
#     

TT = pyquickbench.TimeTrain()
    


for i in range(N):            
    _ = kepler_solve(arr_M[i] , e, tol)

TT.toc("iterated fun")

arr_E = kepler.solve(arr_M, e)
TT.toc("Kepler")

print(TT)

arr_r = np.abs(arr_E - e*np.sin(arr_E)-arr_M)
    
print(arr_r.max())
            