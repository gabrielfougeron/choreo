import os

import numpy as np
import sys
import scipy

__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(__PROJECT_ROOT__)

import choreo 


def print_diag(A):
    for i in range(min(A.shape)):
        print(A[i,i])
    print()
    
np.set_printoptions(
    precision = 3,
    edgeitems = 10,
    linewidth = 150,
    floatmode = "fixed",
)

eps = 1e-12
n = 5
m = 5
rank = 2

rot_n = choreo.scipy_plus.linalg.random_orthogonal_matrix(n)
rot_m = choreo.scipy_plus.linalg.random_orthogonal_matrix(m)

nullspace_dim = m - rank
diag = np.random.rand(rank)
diagmat = np.zeros((n,m))

for i in range(rank):
    diagmat[i,i] = 2 + diag[i]

A = np.matmul(rot_n, np.matmul(diagmat, rot_m))

Q, R, P = scipy.linalg.qr(
    A.T,
    overwrite_a = False ,
    mode = 'full'       ,
    # mode = 'economic'       ,
    pivoting = True     ,
)

print_diag(R)

assert np.linalg.norm(A[P,:] - np.matmul(R.T, Q.T)) < eps 

Q, R, P = scipy.linalg.qr(
    A,
    overwrite_a = False ,
    mode = 'full'       ,
    # mode = 'economic'       ,
    pivoting = True     ,
)


print_diag(R)

assert np.linalg.norm(A[:,P] - np.matmul(Q, R)) < eps 

O, L, U = scipy.linalg.lu(
    A, 
    permute_l=False, 
    overwrite_a=False, 
    check_finite=True, 
    p_indices=True
)

# print_diag(L)
print_diag(U)

assert np.linalg.norm(A - np.matmul(L[O,:], U)) < eps 

O, L, U = scipy.linalg.lu(
    A.T, 
    permute_l=False, 
    overwrite_a=False, 
    check_finite=True, 
    p_indices=True
)

# print_diag(L)
print_diag(U)

assert np.linalg.norm(A - np.matmul(U.T, L.T[:,O])) < eps 



