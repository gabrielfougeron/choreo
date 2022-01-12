import sys,argparse
import random
import numpy as np
import math as m
import scipy.optimize
import scipy
import scipy.sparse.linalg

class ExactKrylovJacobian(scipy.optimize.nonlin.KrylovJacobian):

    def __init__(self,exactgrad, rdiff=None, method='lgmres', inner_maxiter=20,inner_M=None, outer_k=10, **kw):
        
        scipy.optimize.nonlin.KrylovJacobian.__init__(self, rdiff, method, inner_maxiter,inner_M, outer_k)
        self.exactgrad = exactgrad

    def matvec(self, v):
        return self.exactgrad(self.x0,v)

    def rmatvec(self, v):
        return self.exactgrad(self.x0,v)
 
def inv_op(op):
    
    def the_matvec(x):
        
        print('aaa',op.shape)
    
        # ~ v,_ = scipy.sparse.linalg.lgmres(op,x,atol=1e-3)
        # ~ return v
        v,_ = scipy.sparse.linalg.lgmres(op,x,atol=1e-1)
        return v
    
    
    return scipy.sparse.linalg.LinearOperator(op.shape,matvec = the_matvec)
