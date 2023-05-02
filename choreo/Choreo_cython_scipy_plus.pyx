'''
Choreo_cython_scipy_plus.pyx : Defines things I designed I feel ought to be in scipy ... but faster !


'''


import os
import numpy as np
cimport numpy as np
np.import_array()

cimport cython

from libc.math cimport pow as cpow
from libc.math cimport fabs as cfabs
from libc.math cimport cos as ccos
from libc.math cimport sin as csin
from libc.math cimport sqrt as csqrt
from libc.math cimport isnan as cisnan
from libc.math cimport isinf as cisinf

def ExplicitSymplecticWithTable_VX_cython(
    object fun,
    object gun,
    (double, double) t_span,
    double[::1] x0,
    double[::1] v0,
    long nint,
    double[::1] c_table,
    double[::1] d_table,
    long nsteps
):
    r"""
    
    This function computes an approximate solution to the :ref:`partitioned Hamiltonian system<ode_PHS>` using an explicit Runge-Kutta method.
    
    :param fun: function 
    :param gun: function 

    :param t_span: initial and final time for simulation
    :type t_span: (double, double)


    :return: two np.ndarray[double, ndim=1, mode="c"] for the final 


    """

    cdef long istep,id
    cdef long ndof = x0.size

    cdef double tx = t_span[0]
    cdef double tv = t_span[0]
    cdef double dt = (t_span[1] - t_span[0]) / nint

    cdef double[::1] cdt = np.empty((nsteps),dtype=np.float64)
    cdef double[::1] ddt = np.empty((nsteps),dtype=np.float64)

    cdef np.ndarray[double, ndim=1, mode="c"] x_np = np.empty((ndof),dtype=np.float64)
    cdef double[::1] x = x_np
    for idof in range(ndof):
        x[idof] = x0[idof]

    cdef np.ndarray[double, ndim=1, mode="c"] v_np = np.empty((ndof),dtype=np.float64)
    cdef double[::1] v = v_np
    for idof in range(ndof):
        v[idof] = v0[idof]

    cdef double[::1] res

    for istep in range(nsteps):
        cdt[istep] = c_table[istep]*dt
        ddt[istep] = d_table[istep]*dt

    for iint in range(nint):

        for istep in range(nsteps):

            res = fun(tv,v)  
            for idof in range(ndof):
                x[idof] += cdt[istep] * res[idof]  

            tx += cdt[istep]

            res = gun(tx,x)   
            for idof in range(ndof):
                v[idof] += ddt[istep] * res[idof]  
            tv += ddt[istep]

    return x_np, v_np

def ExplicitSymplecticWithTable_XV_cython(
    object fun,
    object gun,
    (double, double) t_span,
    double[::1] x0,
    double[::1] v0,
    long nint,
    double[::1] c_table,
    double[::1] d_table,
    long nsteps
):

    cdef long istep,idof
    cdef long ndof = x0.size

    cdef double tx = t_span[0]
    cdef double tv = t_span[0]
    cdef double dt = (t_span[1] - t_span[0]) / nint

    cdef double[::1] cdt = np.empty((nsteps),dtype=np.float64)
    cdef double[::1] ddt = np.empty((nsteps),dtype=np.float64)

    cdef np.ndarray[double, ndim=1, mode="c"] x_np = np.empty((ndof),dtype=np.float64)
    cdef double[::1] x = x_np
    for idof in range(ndof):
        x[idof] = x0[idof]

    cdef np.ndarray[double, ndim=1, mode="c"] v_np = np.empty((ndof),dtype=np.float64)
    cdef double[::1] v = v_np
    for idof in range(ndof):
        v[idof] = v0[idof]

    cdef double[::1] res

    for istep in range(nsteps):
        cdt[istep] = c_table[istep]*dt
        ddt[istep] = d_table[istep]*dt

    for iint in range(nint):

        for istep in range(nsteps):

            res = gun(tx,x)   
            for idof in range(ndof):
                v[idof] += cdt[istep] * res[idof]  
            tv += cdt[istep]

            res = fun(tv,v)  
            for idof in range(ndof):
                x[idof] += ddt[istep] * res[idof]  

            tx += ddt[istep]

    return x_np, v_np

def SymplecticStormerVerlet_XV_cython(
    object fun,
    object gun,
    (double, double) t_span,
    double[::1] x0,
    double[::1] v0,
    long nint,
):

    cdef long idof
    cdef long ndof = x0.size

    cdef double t = t_span[0]
    cdef double dt = (t_span[1] - t_span[0]) / nint
    cdef double dt_half = dt*0.5

    cdef np.ndarray[double, ndim=1, mode="c"] x_np = np.empty((ndof),dtype=np.float64)
    cdef double[::1] x = x_np
    for idof in range(ndof):
        x[idof] = x0[idof]

    cdef np.ndarray[double, ndim=1, mode="c"] v_np = np.empty((ndof),dtype=np.float64)
    cdef double[::1] v = v_np
    for idof in range(ndof):
        v[idof] = v0[idof]

    cdef double[::1] res

    res = gun(t,x)   
    for idof in range(ndof):
        v[idof] += dt_half * res[idof]  

    for iint in range(nint-1):

        res = fun(t,v)  
        for idof in range(ndof):
            x[idof] += dt* res[idof]  

        t += dt

        res = gun(t,x)   
        for idof in range(ndof):
            v[idof] += dt * res[idof]  

    res = fun(t,v)  
    for idof in range(ndof):
        x[idof] += dt * res[idof]  

    t += dt

    res = gun(t,x)   
    for idof in range(ndof):
        v[idof] += dt_half * res[idof]  

    return x_np, v_np

def SymplecticStormerVerlet_VX_cython(
    object fun,
    object gun,
    (double, double) t_span,
    double[::1] x0,
    double[::1] v0,
    long nint,
):

    cdef long idof
    cdef long ndof = x0.size

    cdef double t = t_span[0]
    cdef double dt = (t_span[1] - t_span[0]) / nint
    cdef double dt_half = dt*0.5

    cdef np.ndarray[double, ndim=1, mode="c"] x_np = np.empty((ndof),dtype=np.float64)
    cdef double[::1] x = x_np
    for idof in range(ndof):
        x[idof] = x0[idof]

    cdef np.ndarray[double, ndim=1, mode="c"] v_np = np.empty((ndof),dtype=np.float64)
    cdef double[::1] v = v_np
    for idof in range(ndof):
        v[idof] = v0[idof]

    cdef double[::1] res

    res = fun(t,v)  
    for idof in range(ndof):
        x[idof] += dt_half * res[idof]  

    t += dt_half
    
    for iint in range(nint-1):

        res = gun(t,x)   
        for idof in range(ndof):
            v[idof] += dt * res[idof]  

        res = fun(t,v)  
        for idof in range(ndof):
            x[idof] += dt* res[idof]  

        t += dt

    res = gun(t,x)   
    for idof in range(ndof):
        v[idof] += dt * res[idof]  

    res = fun(t,v)  
    for idof in range(ndof):
        x[idof] += dt_half* res[idof]  

    t += dt_half

    return x_np, v_np





def ImplicitSymplecticWithTable_VX_cython(
    object fun,
    object gun,
    (double, double) t_span,
    double[::1] x0,
    double[::1] v0,
    long nint,
    double[::1] a_table,
    double[::1] b_table,
    double[::1] c_table,
    long nsteps
):
    r"""
    
    This function computes an approximate solution to the :ref:`partitioned Hamiltonian system<ode_PHS>` using an implicit Runge-Kutta method.
    THe implicit Runge-Kutta equations are solved with Gauss-Seidel iterations.
    
    :param fun: function 
    :param gun: function 

    :param t_span: initial and final time for simulation
    :type t_span: (double, double)


    :return: two np.ndarray[double, ndim=1, mode="c"] for the final 


    """

    cdef double tx = t_span[0]
    cdef double tv = t_span[0]
    cdef double dt = (t_span[1] - t_span[0]) / nint

    cdef double[::1] cdt = np.empty((nsteps),dtype=np.float64)
    cdef double[::1] ddt = np.empty((nsteps),dtype=np.float64)

    cdef np.ndarray[double, ndim=1, mode="c"] x_np = x0
    cdef np.ndarray[double, ndim=1, mode="c"] v_np = v0

    cdef double[::1] x = x_np
    cdef double[::1] v = v_np

    cdef long ndof = x0.size
    cdef double[::1] res

    # cdef np.ndarray[double, ndim=1, mode="c"] x = x0.copy()
    # cdef np.ndarray[double, ndim=1, mode="c"] v = v0.copy()


# 
#     cdef long istep,id
#     for istep in range(nsteps):
#         cdt[istep] = c_table[istep]*dt
#         ddt[istep] = d_table[istep]*dt
# 
#     for iint in range(nint):
# 
#         for istep in range(nsteps):
# 
#             res = fun(tv,v)  
#             for idof in range(ndof):
#                 x[idof] += cdt[istep] * res[idof]  
# 
#             tx += cdt[istep]
# 
#             res = gun(tx,x)   
#             for idof in range(ndof):
#                 v[idof] += ddt[istep] * res[idof]  
#             tv += ddt[istep]

    return x_np, v_np































def Hessenberg_skew_Hamiltonian(
    np.ndarray[double, ndim=2, mode="c"] A,
    np.ndarray[double, ndim=2, mode="c"] B,
    np.ndarray[double, ndim=2, mode="c"] C,
    long n,
):
    r'''
    Following [1], we compute a real skew-Hamiltonian Schur decomposition of the input.

    The fist step is to compute 
    symplectic reduction of a skew-Hamiltonian matrix in Hessenberg form.

    WARNING: Inputs are overwritten

                                         [ A   B   ]
    Input: a skew-Hamiltonian matrix W = [ C   A^T ], where B = -B^T, C = -C^T, all matrices of size n x n
    
    
                                                 [  U1  U2 ]                  [ N1  N2   ]
    Output : An orthogonal symplectic matrix U = [ -U2  U1 ] and a matrix N = [ 0   N1^T ], where N2 = N2^T
    such that U^T W U = N
    
    [1] Structure-Preserving Schur Methods for Computing Square Roots of Real Skew-Hamiltonian Matrices
    Zhongyun Liu, Yulin Zhang, Carla Ferreira, and Rui Ralha

    [2] A Symplectic Method for Approximating All the Eigenvalues of a Hamiltonian Matrix
    C. van Loan

    '''


    cdef long i,j,k

    cdef double sqrt_wtw, wtw, two_ovr_wtw
    cdef double wtMw, Mp
    cdef double Mwi
    cdef double cgiv,sgiv

    cdef double[::1] Hd = np.empty((n))
    cdef double[::1] Mw = np.empty((n))


    
    for k in range(n-1):


        if (k < n-1):
        # Algo   H

            # Compute H
            wtw = C[k,k]*C[k,k]
            for i in range(k+1,n):
                wtw += C[i,k]*C[i,k]
            
            sqrt_wtw = csqrt(wtw)
            two_ovr_wtw = 2/wtw

            if C[k,k] > 0:
                Hd[k] = C[i,i] + sqrt_wtw
            else:
                Hd[k] = C[i,i] - sqrt_wtw 



            # Compute H A H and update A
            # Compute A H
            # i = k
            Mwi = A[k,k] * Hd[k]
            for j in range(k+1,n):
                Mwi += A[k,j] * C[j,k]

            Mw[k] = Mwi
            wtMw = Mwi * Hd[k] 

            # i > k
            for i in range(k+1,n):
                Mwi = A[i,k] * Hd[k]
                
                for j in range(k+1,n):
                    Mwi += A[i,j] * C[j,k]

                Mw[i] = Mwi
                wtMw += Mwi * C[i,k]

            wtMw *= two_ovr_wtw

            # Update A <- H A H
            # i = k
            A[k,k] += two_ovr_wtw * (wtMw * Hd[k] - 2 * Mw[k]) * Hd[k]

            for j in range(k+1,n):
                
                Mp = two_ovr_wtw * ( wtMw * Hd[k] * C[j,k]  - Hd[k] * Mw[j] - C[j,k] * Mw[k] )

                A[k,j] += Mp    
                A[j,k] += Mp

            # i > k
            for i in range(k+1,n):

                for j in range(i,n):

                    Mp = two_ovr_wtw * ( wtMw * C[i,k] * C[j,k]  - C[i,k] * Mw[j] - C[j,k] * Mw[i] )

                    A[k,j] += Mp
                    A[j,k] += Mp




            # Compute H B H and update B
            # Compute B H
            # i = k
            Mwi = B[k,k] * Hd[k]
            for j in range(k+1,n):
                Mwi += B[k,j] * C[j,k]

            Mw[k] = Mwi
            wtMw = Mwi * Hd[k] 

            # i > k
            for i in range(k+1,n):
                Mwi = B[i,k] * Hd[k]
                
                for j in range(k+1,n):
                    Mwi += B[i,j] * C[j,k]

                Mw[i] = Mwi
                wtMw += Mwi * C[i,k]

            wtMw *= two_ovr_wtw

            # Update B <- H B H
            # i = k
            B[k,k] += two_ovr_wtw * (wtMw * Hd[k] - 2 * Mw[k]) * Hd[k]

            for j in range(k+1,n):
                
                Mp = two_ovr_wtw * ( wtMw * Hd[k] * C[j,k]  - Hd[k] * Mw[j] - C[j,k] * Mw[k] )

                B[k,j] += Mp
                B[j,k] += Mp

            # i > k
            for i in range(k+1,n):

                for j in range(i,n):

                    Mp = two_ovr_wtw * ( wtMw * C[i,k] * C[j,k]  - C[i,k] * Mw[j] - C[j,k] * Mw[i] )

                    B[k,j] += Mp
                    B[j,k] += Mp

        # Algo J
        # ~ sqrt_wtw = csqrt( A[k+1,k] * A[k+1,k] + 


        # if k < n-1 :
            # Algo H



