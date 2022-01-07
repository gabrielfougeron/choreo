import sys,argparse
import random
import numpy as np
import math as m
import scipy.optimize
import scipy


class ExactKrylovJacobian(scipy.optimize.nonlin.KrylovJacobian):


    def __init__(self,exactgrad, rdiff=None, method='lgmres', inner_maxiter=20,inner_M=None, outer_k=10, **kw):
        
        scipy.optimize.nonlin.KrylovJacobian.__init__(self, rdiff, method, inner_maxiter,inner_M, outer_k)
        self.exactgrad = exactgrad

    def matvec(self, v):
        return self.exactgrad(self.x0,v)

    def rmatvec(self, v):
        return self.exactgrad(self.x0,v)
 
