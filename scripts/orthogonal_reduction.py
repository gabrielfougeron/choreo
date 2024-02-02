import os

import numpy as np
import sys
import scipy

__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(__PROJECT_ROOT__)

import choreo 


np.set_printoptions(
    precision = 3,
    edgeitems = 10,
    linewidth = 150,
    floatmode = "fixed",
)

eps = 1e-12

dim = 2

I = np.identity(dim)

O = choreo.scipy_plus.linalg.random_orthogonal_matrix(dim)

# 
# eigvals, P = scipy.linalg.eig(a=O, b=None, left=False, right=True, overwrite_a=False, overwrite_b=False, check_finite=True, homogeneous_eigvals=False)
# 
# Q = P.real
# S = P.imag
# D = np.diag(eigvals)
# 
# 
# 
# assert np.linalg.norm( np.matmul( np.matmul(P, D ), P.conj().T ) - O ) < eps
# assert np.linalg.norm( np.matmul(Q, Q.T) + np.matmul(S, S.T) - I ) < eps
# assert np.linalg.norm( np.matmul(S, Q.T) - np.matmul(Q, S.T)     ) < eps  
# 
# print()
# print("D")
# print(eigvals)
# 
# print()
# print("Q")
# print(Q)
# 
# print()
# print("S")
# print(S)
# 
# print()
# print("QQT")
# print( np.matmul(Q, Q.T) )
# 
# print()
# print("SST")
# print( np.matmul(S, S.T) )
# 
# 
# print("===========================")

A = I - O

# Q, R, P = scipy.linalg.qr(
#     A                   ,
#     overwrite_a = False ,
#     mode = 'economic'   ,
#     pivoting = True     ,
# )
# 
# for i in range(min(R.shape)):
#     if (abs(R[i, i]) < eps):
#         img_dim = i
#         break
# else:
#     img_dim = min(R.shape)
# 
# 
# print(img_dim)
# 
# 
# print(Q.shape)
# print(R.shape)
# print(P.shape)
# 
# assert np.linalg.norm(A[:,P] - np.matmul(Q, R)) < eps 
# 
# x = np.random.random((dim))
# 
# Ax = np.dot(A, x)
# Rx = np.dot(R[:img_dim,:], x[P])
# 
# print(Ax.shape[0], Rx.shape[0])
# 
# print(np.linalg.norm(Ax))
# print(np.linalg.norm(Rx))



eigvals, P = scipy.linalg.eig(a=A, b=None, left=False, right=True, overwrite_a=False, overwrite_b=False, check_finite=True, homogeneous_eigvals=False)


print(abs(eigvals))

print(P.real)
print(P.imag)
 
D = np.diag(eigvals)


