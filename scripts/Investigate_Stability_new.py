import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'

import concurrent.futures
import multiprocessing
import json
import shutil
import random
import time
import math as m
import numpy as np
import scipy
import scipy.linalg
import sys
import fractions
import scipy.integrate
import scipy.special
import matplotlib.pyplot as plt

__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(__PROJECT_ROOT__)

import choreo 

import datetime

One_sec = 1e9

twopi = 2*np.pi

def main():

    # input_folder = os.path.join(__PROJECT_ROOT__,'Sniff_all_sym/01 - Classic gallery')
    # input_folder = os.path.join(__PROJECT_ROOT__,'Sniff_all_sym/3/')
    input_folder = os.path.join(__PROJECT_ROOT__,'choreo_GUI/choreo-gallery/01 - Classic gallery')
    # input_folder = os.path.join(__PROJECT_ROOT__,'Keep/tests')
    
#     ''' Include all files in tree '''
#     input_names_list = []
#     for root, dirnames, filenames in os.walk(input_folder):
# 
#         for filename in filenames:
#             file_path = os.path.join(root, filename)
#             file_root, file_ext = os.path.splitext(os.path.basename(file_path))
# 
#             if (file_ext == '.json' ):
# 
#                 file_path = os.path.join(root, file_root)
#                 the_name = file_path[len(input_folder):]
#                 input_names_list.append(the_name)
# 
# # # 
#     ''' Include all files in folder '''
#     input_names_list = []
#     for file_path in os.listdir(input_folder):
#         file_path = os.path.join(input_folder, file_path)
#         file_root, file_ext = os.path.splitext(os.path.basename(file_path))
#         
#         if (file_ext == '.json' ):
#             # 
#             # if int(file_root) > 8:
#             #     input_names_list.append(file_root)
# 
#             input_names_list.append(file_root)

    input_names_list = ['01 - Figure eight']
    # input_names_list = ['14 - Small mass gap']
    # input_names_list = ['04 - 5 pointed star']


    store_folder = os.path.join(__PROJECT_ROOT__,'Reconverged_sols')
    # store_folder = input_folder

    # Exec_Mul_Proc = True
    Exec_Mul_Proc = False

    if Exec_Mul_Proc:

        # n = 1
        # n = 4
        # n = multiprocessing.cpu_count()
        n = multiprocessing.cpu_count()//2
        
        print(f"Executing with {n} workers")
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=n) as executor:
            
            res = []
            
            for the_name in input_names_list:

                all_kwargs = choreo.Pick_Named_Args_From_Dict(ExecName,dict(globals(),**locals()))
                res.append(executor.submit(ExecName,**all_kwargs))
                time.sleep(0.01)

    else:
            
        for the_name in input_names_list:

            all_kwargs = choreo.Pick_Named_Args_From_Dict(ExecName,dict(globals(),**locals()))
            ExecName(the_name, input_folder, store_folder)


def ExecName(the_name, input_folder, store_folder):

    print('--------------------------------------------')
    print('')
    print(the_name)
    print('')
    print('--------------------------------------------')
    print('')

    geodim = 2

    file_basename = the_name
    
    Info_filename = os.path.join(input_folder,the_name + '.json')

    with open(Info_filename,'r') as jsonFile:
        Info_dict = json.load(jsonFile)


    input_filename = os.path.join(input_folder,the_name + '.npy')

    bare_name = the_name.split('/')[-1]

    all_pos = np.load(input_filename)
    nint = Info_dict["n_int"]
    ncoeff_init = nint // 2 + 1

    c_coeffs = choreo.default_rfft(all_pos,axis=2,norm="forward")
    all_coeffs = np.zeros((Info_dict["nloop"],geodim,ncoeff_init,2),dtype=np.float64)
    all_coeffs[:,:,:,0] = c_coeffs.real
    all_coeffs[:,:,:,1] = c_coeffs.imag


    # theta = 2*np.pi * 0.
    # SpaceRevscal = 1.
    # SpaceRot = np.array( [[SpaceRevscal*np.cos(theta) , SpaceRevscal*np.sin(theta)] , [-np.sin(theta),np.cos(theta)]])
    # TimeRev = 1.
    # TimeShiftNum = 0
    # TimeShiftDen = 1


    theta = 2*np.pi * 0/2
    SpaceRevscal = 1.
    SpaceRot = np.array( [[SpaceRevscal*np.cos(theta) , SpaceRevscal*np.sin(theta)] , [-np.sin(theta),np.cos(theta)]])
    TimeRev = 1.
    TimeShiftNum = 0
    TimeShiftDen = 2



    # all_coeffs_init = choreo.Transform_Coeffs(SpaceRot, TimeRev, TimeShiftNum, TimeShiftDen, all_coeffs)
    # Transform_Sym = choreo.ChoreoSym(SpaceRot=SpaceRot, TimeRev=TimeRev, TimeShift = fractions.Fraction(numerator=TimeShiftNum,denominator=TimeShiftDen))

    all_coeffs_init = np.copy(all_coeffs)
    Transform_Sym = None


    nbody = Info_dict['nbody']
    mass = np.array(Info_dict['mass']).astype(np.float64)
    Sym_list = choreo.Make_SymList_From_InfoDict(Info_dict,Transform_Sym)


    MomConsImposed = True
    # MomConsImposed = False
# 
#     rot_angle = 0
#     s = -1
# 
#     Sym_list.append(choreo.ChoreoSym(
#         LoopTarget=0,
#         LoopSource=0,
#         SpaceRot = np.array([[s*np.cos(rot_angle),-s*np.sin(rot_angle)],[np.sin(rot_angle),np.cos(rot_angle)]],dtype=np.float64),
#         TimeRev=-1,
#         TimeShift=fractions.Fraction(numerator=0,denominator=1)
#     ))

    n_reconverge_it_max = 0
    n_grad_change = 1.

    ActionSyst = choreo.setup_changevar(2,nbody,nint,mass,n_reconverge_it_max,Sym_list=Sym_list,MomCons=MomConsImposed,n_grad_change=n_grad_change,CrashOnIdentity=False)

    x = ActionSyst.Package_all_coeffs(all_coeffs_init)

    ActionSyst.SavePosFFT(x)

    Action,Gradaction = ActionSyst.Compute_action(x)
    Newt_err = ActionSyst.Compute_Newton_err(x)

    Newt_err_norm = np.linalg.norm(Newt_err)/(ActionSyst.nint*ActionSyst.nbody)

    print(f'Saved Newton Error : {Info_dict["Newton_Error"]}')
    print(f'Init Newton Error : {Newt_err_norm}')

    n_eig = 10

    # which_eigs = 'LM' # Largest (in magnitude) eigenvalues.
    which_eigs = 'SM' # Smallest (in magnitude) eigenvalues.
    # which_eigs = 'LA' # Largest (algebraic) eigenvalues.
    # which_eigs = 'SA' # Smallest (algebraic) eigenvalues.
    # which_eigs = 'BE' # Half (k/2) from each end of the spectrum.

    HessMat = ActionSyst.Compute_action_hess_LinOpt(x)
    # w ,v = scipy.sparse.linalg.eigsh(HessMat,k=n_eig,which=which_eigs)
    # print(w)


    Seed_with_RK_solver = True
    # Seed_with_RK_solver = False



    ncoeff = ActionSyst.ncoeff
    nint = ActionSyst.nint
    
    all_coeffs = ActionSyst.RemoveSym(x)
    c_coeffs = all_coeffs.view(dtype=np.complex128)[...,0]


    if Seed_with_RK_solver:

        LagrangeMulInit = np.zeros((2,nbody,geodim,2,nbody,geodim),dtype=np.float64)

        t_span = (0.,1.)

        # nint_ODE_mul = 128
        # nint_ODE_mul = 48
        # nint_ODE_mul = 24
        # nint_ODE_mul = 2
        nint_ODE_mul = 1
        nint_ODE = nint_ODE_mul*nint





# 
        # SymplecticMethod = 'SymplecticEuler'
        # SymplecticMethod = 'SymplecticStormerVerlet'
        # SymplecticMethod = 'SymplecticRuth3'
        SymplecticMethod = 'SymplecticRuth4Rat'
        SymplecticIntegrator = choreo.GetSymplecticIntegrator(SymplecticMethod)

        fun,gun,x0,v0 = ActionSyst.GetTangentSystemDef(x,nint_ODE,method=SymplecticMethod)




#         SymplecticMethod = 'SymplecticGauss1'
#         # SymplecticMethod = 'SymplecticGauss2'
#         # SymplecticMethod = 'SymplecticGauss3'
#         # SymplecticMethod = 'SymplecticGauss5'
#         # SymplecticMethod = 'SymplecticGauss10'
#         # SymplecticMethod = 'SymplecticGauss10'
#         # SymplecticMethod = 'SymplecticGauss15'
# 
#         descr = SymplecticMethod.removeprefix("SymplecticGauss")
#         n = int(descr)
#         Butcher_a_np, Butcher_b_np, Butcher_c_np, Butcher_beta_np, _ = choreo.ComputeGaussButcherTables_np(n)
# 
#         SymplecticIntegrator = choreo.GetSymplecticIntegrator(SymplecticMethod)
# 
#         fun,gun,x0,v0 = ActionSyst.GetTangentSystemDef_new(x,Butcher_c_np,nint)





        all_x, all_v = SymplecticIntegrator(fun,gun,t_span,x0,v0,nint,nint_ODE_mul)

        del fun,gun

        xf = all_x[-1,:].copy()
        vf = all_v[-1,:].copy()

        ndof = nbody*geodim

        MonodromyMat = np.ascontiguousarray(np.concatenate((xf,vf),axis=0).reshape(2*ndof,2*ndof))


        print(MonodromyMat)


        # MonodromyMat = np.dot(MonodromyMat,MonodromyMat)

        MonodromyMatLog = scipy.linalg.logm(MonodromyMat)


        w = np.zeros((2*ndof,2*ndof),dtype=np.float64)
        w[0:ndof,ndof:2*ndof] = np.identity(ndof)
        w[ndof:2*ndof,0:ndof] = -np.identity(ndof)

        MonodromyMatLog = (MonodromyMatLog+ w @ MonodromyMatLog.T @ w ) / 2

        MonodromyMatLogsq = np.dot(MonodromyMatLog,MonodromyMatLog)
        


        print('Symplecticity')
        print(np.linalg.norm(w - np.dot(MonodromyMat.transpose(),np.dot(w,MonodromyMat))))
        print(np.linalg.norm(np.dot(MonodromyMatLog.transpose(),w) + np.dot(w,MonodromyMatLog)))




        eigvals,eigvects = scipy.linalg.eig(a=MonodromyMat)
        # eigvals,eigvects = scipy.linalg.eig(a=MonodromyMatLog)
        # eigvals,eigvects = scipy.linalg.eig(a=MonodromyMatLogsq)

        # print(eigvals)
        # print(eigvals.real)
        print(abs(eigvals))
        # print(eigvects)


        # exit()

        all_pos_d_init = np.zeros((nbody,geodim,2,nbody,geodim,nint),dtype=np.float64)

        for iint in range(nint):

            xv = np.ascontiguousarray(np.concatenate((all_x[iint,:],all_v[iint,:]),axis=0).reshape(2*ndof,2*ndof))

            PeriodicPart = np.dot(xv,scipy.linalg.expm(-(iint / nint)*MonodromyMatLog))

            all_pos_d_init[:,:,:,:,:,iint] = (PeriodicPart.reshape(2,nbody*geodim,2,nbody*geodim)[0,:,:,:]).reshape((nbody,geodim,2,nbody,geodim))


        # print(MonodromyMatLog)
        # print(MonodromyMat)

        # Evaluates the relative accuracy of the Monodromy matrix integration process
        # zo should be an eigenvector of the Monodromy matrix, with eigenvalue 1
        yo = ActionSyst.Compute_init_pos_vel(x).reshape(-1)
        zo = ActionSyst.Compute_Auto_ODE_RHS(yo)

        # '''SVD'''
        # U,Instability_magnitude,Instability_directions = scipy.linalg.svd(MonodromyMat, full_matrices=True, compute_uv=True, overwrite_a=False, check_finite=True, lapack_driver='gesdd')
# 
        '''Eigendecomposition'''
        Instability_magnitude,Instability_directions = choreo.InstabilityDecomposition(MonodromyMat)

        # print(Instability_magnitude)

        print(zo)

        print(f'Relative error on flow eigenstate: {np.linalg.norm(MonodromyMat.dot(zo)-zo)/np.linalg.norm(zo):e}')
        # print(the_name+f' {Instability_magnitude[:]}')
        # print(the_name+f' {np.flip(1/Instability_magnitude[:])}')
        # print("Relative error on loxodromy ",np.linalg.norm(Instability_magnitude - np.flip(1/Instability_magnitude))/np.linalg.norm(Instability_magnitude))

        all_coeffs_dc_init = choreo.default_rfft(all_pos_d_init,norm="forward")
        all_coeffs_d_init = np.zeros((nbody,geodim,2,nbody,geodim,ncoeff,2),np.float64)
        all_coeffs_d_init[:,:,:,:,:,:,0] = all_coeffs_dc_init[:,:,:,:,:,:ncoeff].real
        all_coeffs_d_init[:,:,:,:,:,:,1] = all_coeffs_dc_init[:,:,:,:,:,:ncoeff].imag


        x0 = np.ascontiguousarray(np.concatenate((all_coeffs_d_init.reshape(-1),LagrangeMulInit.reshape(-1))))

    else:

        all_coeffs_d_init = np.zeros((nbody,geodim,2,nbody,geodim,ncoeff,2),dtype=np.float64)
        LagrangeMulInit = np.zeros((2,nbody,geodim,2,nbody,geodim),dtype=np.float64)
        MonodromyMatLog = np.zeros((2,nbody,geodim,2,nbody,geodim),dtype=np.float64)

        for ib in range(nbody):
            for idim in range(geodim):
                all_coeffs_d_init[ib,idim,0,ib,idim,0,0] = 1
                all_coeffs_d_init[ib,idim,1,ib,idim,1,1] = -1./(2*twopi)


        x0 = np.ascontiguousarray(np.concatenate((all_coeffs_d_init.reshape(-1), LagrangeMulInit.reshape(-1))))

    exit()
    MonodromyMatLog = np.ascontiguousarray(MonodromyMatLog.reshape(2,nbody,geodim,2,nbody,geodim))

    krylov_method = 'lgmres'
    # krylov_method = 'gmres'
    # krylov_method = 'bicgstab' 
    # krylov_method = 'cgs'
    # krylov_method = 'minres'
    # krylov_method = 'tfqmr'


    # line_search = 'armijo'
    line_search = 'wolfe'

    gradtol = 1e-13

    disp_scipy_opt = True
    # disp_scipy_opt = False

    maxiter = 100


            
    F = lambda x : choreo.TangentLagrangeResidual(
        x,
        nbody,
        ncoeff,
        nint,
        ActionSyst.mass,
        all_coeffs,
        all_pos,
        MonodromyMatLog,
    )

    res_1D = F(x0)
    # print(np.linalg.norm(res_1D))

    
    ibeg = 0
    iend = (nbody*geodim*2*nbody*geodim*ncoeff*2)
    res_1D_all_coeffs = res_1D[ibeg:iend].reshape((nbody,geodim,2,nbody,geodim,ncoeff,2))

    ibeg = iend
    iend = iend + (2*nbody*geodim*2*nbody*geodim)
    res_LagrangeMul = res_1D[ibeg:iend].reshape((2,nbody,geodim,2,nbody,geodim))


    # print(res_1D_all_coeffs)

    # print(np.linalg.norm(res_1D_all_coeffs))
    # print(np.linalg.norm(res_LagrangeMul))


    res_1D_all_coeffs_c = res_1D_all_coeffs.view(dtype=np.complex128)[...,0]
    res_1D_all_pos = choreo.default_irfft(res_1D_all_coeffs_c,n=nint,norm="forward")

#     plt.figure()
# 
#     for ib in range(nbody):
#         for idim in range(geodim):
#             for jvx in range(2):
#                 for jb in range(nbody): 
#                     for jdim in range(geodim):
# 
#                         # plt.plot(all_pos_d_init[ib,idim,jvx,jb,jdim,:])
#                         plt.plot(res_1D_all_pos[ib,idim,jvx,jb,jdim,:])
# 
# 
#     plt.savefig("out.png")


    jac_options = {'method':krylov_method}
    jacobian = scipy.optimize.nonlin.KrylovJacobian(**jac_options)

    opt_result , info = scipy.optimize.nonlin.nonlin_solve(F=F,x0=x0,jacobian=jacobian,verbose=disp_scipy_opt,maxiter=maxiter,f_tol=gradtol,line_search=line_search,raise_exception=False,full_output=True)

    print(opt_result)

if __name__ == "__main__":
    main()    
