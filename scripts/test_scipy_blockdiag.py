import os
import numpy as np
import threadpoolctl
import choreo 
import scipy
import json

__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))

# TODO : Add this as pytests

def main():
    
    eps = 1e-12
    
    n = 3
    
    P = choreo.scipy_plus.random_orthogonal_matrix(n)
    
    # Mat = choreo.scipy_plus.random_orthogonal_matrix(n)
    
    angle = 2 * np.pi * 2/3
    
    Mat = P @ np.array([[np.cos(angle),np.sin(angle),0],[-np.sin(angle),np.cos(angle),0.],[0.,0.,1.]]) @ P.T
    
    print(Mat)
    
    # D, P = choreo.scipy_plus.InstabilityDecomposition(Mat, eps=1e-12)
    # 
    # print(D)
    # print(P)
    
    print()
    w, v = scipy.linalg.eig(Mat)
    print(np.linalg.norm( Mat @ v - v @ np.diag(w))) 

    wr, vr = scipy.linalg.cdf2rdf(w, v)
    print(np.linalg.norm( Mat @ vr - vr @ wr))     
    
    print(wr)
    print(vr)
    print()
    
    cs_angles, subspace_dim, vr = choreo.scipy_plus.DecomposeRotation(Mat, eps=1e-12)
    
    print(cs_angles)
    print(subspace_dim)
    print()
         
    i = 0
    for d in subspace_dim:
        
        if d == 2:
            
            angle = choreo.scipy_plus.cs_to_angle(cs_angles[i], cs_angles[i+1])
            print(angle / (2*np.pi))

        
        
        i += d
    

# Fraction(1.1).limit_denominator()




if __name__ == "__main__":
    with threadpoolctl.threadpool_limits(limits=1):
        main()
