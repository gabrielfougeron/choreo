import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'

import concurrent.futures
import multiprocessing
import shutil
import random
import time
import math as m
import numpy as np
import sys
import fractions
import scipy

__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(__PROJECT_ROOT__)

import choreo 

import datetime


n = 6
k = 3


mat_sqsym = np.random.random((n,n))
mat_sqsym = mat_sqsym - mat_sqsym.T
rot = scipy.linalg.expm(mat_sqsym)

# rot = np.identity(n)

perm = np.random.permutation(n)
choice = np.random.choice(n, size = k, replace=False)

print(choice)


small_mat = np.zeros((n,n))
for i in range(k):
    small_mat[perm[i],choice[i]] = 1



mat_dense = np.matmul(small_mat,rot.T)


print(mat_dense)

# exit()


# mat_dense = np.array([[ 0, 1], [0,0]])


mat_sparse_T = scipy.sparse.coo_matrix(mat_dense.T)



nullspace_sparse = choreo.null_space_sparseqr(mat_sparse_T)



nullspace_sparse_dense = nullspace_sparse.todense()

print(nullspace_sparse_dense)


nullspace_dense = scipy.linalg.null_space(mat_dense)


print(nullspace_dense)


print(np.linalg.norm(np.matmul(mat_dense, nullspace_sparse_dense)))
print(np.linalg.norm(np.matmul(mat_dense, nullspace_dense)))