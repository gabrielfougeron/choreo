import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import concurrent.futures
import multiprocessing
import shutil
import random
import time
import math as m
import numpy as np
import scipy.linalg
import sys
import fractions
import scipy.integrate
import scipy.special

__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(__PROJECT_ROOT__)

import choreo 

import datetime

One_sec = 1e9

def main():


    input_folder = os.path.join(__PROJECT_ROOT__,'Sniff_all_sym/5/')
    # input_folder = os.path.join(__PROJECT_ROOT__,'Sniff_all_sym/copy/')
    # input_folder = os.path.join(__PROJECT_ROOT__,'Sniff_all_sym/Gallery_videos/C-4')
# 
#     ''' Include all files in tree '''
#     input_names_list = []
#     for root, dirnames, filenames in os.walk(input_folder):
# 
#         for filename in filenames:
#             file_path = os.path.join(root, filename)
#             file_root, file_ext = os.path.splitext(os.path.basename(file_path))
# 
#             if (file_ext == '.txt' ):
# 
#                 file_path = os.path.join(root, file_root)
#                 the_name = file_path[len(input_folder):]
#                 input_names_list.append(the_name)
# 
# # 
#     ''' Include all files in folder '''
#     input_names_list = []
#     for file_path in os.listdir(input_folder):
#         file_path = os.path.join(input_folder, file_path)
#         file_root, file_ext = os.path.splitext(os.path.basename(file_path))
#         
#         if (file_ext == '.txt' ):
#             # 
#             # if int(file_root) > 8:
#             #     input_names_list.append(file_root)
# 
#             input_names_list.append(file_root)

    input_names_list = ['00002']

    store_folder = os.path.join(__PROJECT_ROOT__,'Sniff_all_sym/mod')
    # store_folder = input_folder

    # Save_All_Coeffs = True
    Save_All_Coeffs = False

    # Save_All_Coeffs_No_Sym = True
    Save_All_Coeffs_No_Sym = False

    # Save_Newton_Error = True
    Save_Newton_Error = False

    Save_img = True
    # Save_img = False
# 
    # Save_thumb = True
    Save_thumb = False

    # img_size = (12,12) # Image size in inches
    img_size = (8,8) # Image size in inches
    thumb_size = (2,2) # Image size in inches
    
    color = "body"
    # color = "loop"
    # color = "velocity"
    # color = "all"

    # Save_anim = True
    Save_anim = False

    # Save_ODE_anim = True
    Save_ODE_anim = False

    # ODE_method = 'RK23'
    # ODE_method = 'RK45'
    ODE_method = 'DOP853'
    # ODE_method = 'Radau'
    # ODE_method = 'BDF'

    atol_ode = 1e-10
    rtol_ode = 1e-12

    vid_size = (8,8) # Image size in inches
    vid_size_perturb = (4,4) # Image size in inches
    # vid_size_perturb = (8,8) # Image size in inches
    # nint_plot_anim = 2*2*2*3*3
    nint_plot_anim = 2*2*2*3*3*5
    dnint = 30

    nint_plot_img = nint_plot_anim * dnint

    min_n_steps_ode = 1*nint_plot_anim

    try:
        the_lcm
    except NameError:
        period_div = 1.
    else:
        period_div = the_lcm
# 
    # nperiod_anim = 1.
    nperiod_anim = 3.
    # nperiod_anim = 1./period_div

    Plot_trace_anim = True
    # Plot_trace_anim = False

    GradActionThresh = 1e-8
# 
    InvestigateStability = True
    # InvestigateStability = False

    # Save_Perturbed = True
    Save_Perturbed = False

    dy_perturb_mul = 1e-8

    # Relative_Perturb = False
    Relative_Perturb = True

    # InvestigateIntegration = True
    InvestigateIntegration = False

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
            ExecName(**all_kwargs)



def ExecName(
    the_name,
    input_folder,
    GradActionThresh,
    store_folder,
    Save_img,
    nint_plot_img,
    img_size,
    thumb_size,
    color,
    Save_thumb,
    Save_anim,
    nint_plot_anim,
    nperiod_anim,
    Plot_trace_anim,
    vid_size,
    vid_size_perturb,
    dnint,
    Save_Newton_Error,
    Save_All_Coeffs,
    Save_All_Coeffs_No_Sym,
    Save_ODE_anim,
    ODE_method,
    min_n_steps_ode,
    atol_ode,
    rtol_ode,
    InvestigateStability,
    InvestigateIntegration,
    Save_Perturbed,
    dy_perturb_mul,
    Relative_Perturb,
    ):

    print('')
    print(the_name)

    input_filename = os.path.join(input_folder,the_name)
    input_filename = input_filename + '.npy'

    bare_name = the_name.split('/')[-1]

    all_coeffs = np.load(input_filename)

    theta = 2*np.pi *0.
    SpaceRevscal = 1.
    SpaceRot = np.array( [[SpaceRevscal*np.cos(theta) , SpaceRevscal*np.sin(theta)] , [-np.sin(theta),np.cos(theta)]])
    TimeRev = 1.
    TimeShiftNum = 0
    TimeShiftDen = 2

    all_coeffs = choreo.Transform_Coeffs(SpaceRot, TimeRev, TimeShiftNum, TimeShiftDen, all_coeffs)

    ncoeff_init = all_coeffs.shape[2]

    the_i = -1
    the_i_max = 0

    Gradaction_OK = False

    while (not(Gradaction_OK) and (the_i < the_i_max)):

        the_i += 1
        '''
        # p = 1
        p_list = range(the_i_max)
        # p_list = [3]
        p = p_list[the_i%len(p_list)]

        nc = 3

        mm = 1
        # mm_list = [1]
        # mm = mm_list[the_i%len(mm_list)]

        nbpl=[nc]

        SymType = {
            'name'  : 'D',
            'n'     : nc,
            'm'     : mm,
            'l'     : 0,
            'k'     : 1,
            'p'     : p,
            'q'     : nc,
        }
        Sym_list = choreo.Make2DChoreoSym(SymType,range(nc))
        nbody = nc

        '''

        SymName = None
        nbpl=[5]
        Sym_list,nbody = choreo.Make2DChoreoSymManyLoops(nbpl=nbpl,SymName=SymName)


        mass = np.ones((nbody),dtype=np.float64)

        # MomConsImposed = True
        MomConsImposed = False

        n_reconverge_it_max = 0
        n_grad_change = 1.

        callfun = choreo.setup_changevar(nbody,ncoeff_init,mass,n_reconverge_it_max,Sym_list=Sym_list,MomCons=MomConsImposed,n_grad_change=n_grad_change,CrashOnIdentity=False)

        x = choreo.Package_all_coeffs(all_coeffs,callfun)

        Action,Gradaction = choreo.Compute_action(x,callfun)

        Gradaction_OK = (np.linalg.norm(Gradaction) < GradActionThresh)

    if not(Gradaction_OK):
        raise(ValueError('Correct Symmetries not found'))

    filename_output = os.path.join(store_folder,bare_name)

    print('Saving solution as '+filename_output+'.*')

    choreo.Write_Descriptor(x,callfun,filename_output+'.txt')
    
    if Save_img :
        choreo.plot_all_2D(x,nint_plot_img,callfun,filename_output+'.png',fig_size=img_size,color=color)
    
    if Save_thumb :
        choreo.plot_all_2D(x,nint_plot_img,callfun,filename_output+'_thumb.png',fig_size=thumb_size,color=color)
        
    if Save_anim :
        choreo.plot_all_2D_anim(x,nint_plot_anim,callfun,filename_output+'.mp4',nperiod_anim,Plot_trace=Plot_trace_anim,fig_size=vid_size,dnint=dnint,color_list=color_list,color=color)

    if Save_Newton_Error :
        choreo.plot_Newton_Error(x,callfun,filename_output+'_newton.png')
    
    if Save_All_Coeffs:

        np.save(filename_output+'.npy',all_coeffs)

    if Save_All_Coeffs_No_Sym:
        
        all_coeffs_nosym = choreo.RemoveSym(x,callfun)

        np.save(filename_output+'_nosym.npy',all_coeffs_nosym)

    if Save_ODE_anim:
        
        yo = choreo.Compute_init_pos_vel(x,callfun).reshape(-1)

        t_eval = np.array([i/nint_plot_img for i in range(round(nperiod_anim*nint_plot_img))])

        fun = lambda t,y: choreo.Compute_ODE_RHS(t,y,callfun)

        ode_res = scipy.integrate.solve_ivp(fun=fun, t_span=(0.,nperiod_anim), y0=yo, method=ODE_method, t_eval=t_eval, dense_output=False, events=None, vectorized=False,max_step=1./min_n_steps_ode,atol=atol_ode,rtol=rtol_ode)

        all_pos_vel = ode_res['y'].reshape(2,nbody,choreo.ndim,round(nint_plot_img*nperiod_anim))
        all_pos_ode = all_pos_vel[0,:,:,:]
        
        choreo.plot_all_2D_anim(x,nint_plot_anim,callfun,filename_output+'_ode.mp4',nperiod_anim,Plot_trace=Plot_trace_anim,fig_size=vid_size,dnint=dnint,all_pos_trace=all_pos_ode,all_pos_points=all_pos_ode,color_list=color_list,color=color)


    if Save_Perturbed:

        # print('')
        # print('Symplectic integration of tangent system')
        # print('')

        # SymplecticMethod = 'SymplecticEuler'
        # SymplecticMethod = 'SymplecticStormerVerlet'
        SymplecticMethod = 'SymplecticRuth3'
        SymplecticIntegrator = choreo.GetSymplecticIntegrator(SymplecticMethod)

        nint_mul = 128

        nint = callfun[0]['nint_list'][callfun[0]["current_cvg_lvl"]]*nint_mul

        fun,gun,x0,v0 = choreo.GetTangentSystemDef(x,callfun,nint,method=SymplecticMethod)

        ndof = nbody*choreo.ndim

        t_span = (0.,1.)

        t_beg= time.perf_counter_ns()
        xf,vf = SymplecticIntegrator(fun,gun,t_span,x0,v0,nint)
        t_end = time.perf_counter_ns()
        MonodromyMat = np.ascontiguousarray(np.concatenate((xf,vf),axis=0).reshape(2*ndof,2*ndof))

        del fun,gun

        # Evaluates the relative accuracy of the Monodromy matrix integration process
        # zo should be an eigenvector of the Monodromy matrix, with eigenvalue 1
        yo = choreo.Compute_init_pos_vel(x,callfun).reshape(-1)
        zo = choreo.Compute_Auto_ODE_RHS(yo,callfun)

        # '''SVD'''
        # U,Instability_magnitude,Instability_directions = scipy.linalg.svd(MonodromyMat, full_matrices=True, compute_uv=True, overwrite_a=False, check_finite=True, lapack_driver='gesdd')
# 
        '''Eigendecomposition'''
        Instability_magnitude,Instability_directions = choreo.InstabilityDecomposition(MonodromyMat)

        print(the_name+f' Relative error on flow eigenstate: {np.linalg.norm(MonodromyMat.dot(zo)-zo)/np.linalg.norm(zo):e} time : {(t_end-t_beg)/One_sec:f}')
        # print(the_name+f' {Instability_magnitude[:]}')
        # print(the_name+f' {np.flip(1/Instability_magnitude[:])}')
        print("Relative error on loxodromy ",np.linalg.norm(Instability_magnitude - np.flip(1/Instability_magnitude))/np.linalg.norm(Instability_magnitude))


        # n_period_unstable = -np.log(np.finfo(float).eps)/np.log(Instability_magnitude[0])
        # print(the_name+f' Number of periods before expected unstable behavior {n_period_unstable}')

        list_vid = []

        # xlim = choreo.Compute_xlim(x,callfun,extend=0.2)
        xlim = choreo.Compute_xlim(x,callfun,extend=0.03)

        n_unperturbed = 1
        n_perturbed = 11
        n_random = 0

        nvid = n_unperturbed + n_perturbed + n_random

        nx = 4
        ny = 3

        # nx,ny = choreo.factor_squarest(nvid)

        assert nx*ny == nvid

        if Relative_Perturb:
            dy_perturb = dy_perturb_mul / Instability_magnitude[0]
        else:
            dy_perturb = dy_perturb_mul

        for i in range(n_unperturbed):

            yo = choreo.Compute_init_pos_vel(x,callfun).reshape(-1)
            t_eval = np.array([i/nint_plot_img for i in range(round(nperiod_anim*nint_plot_img))])
            fun = lambda t,y: choreo.Compute_ODE_RHS(t,y,callfun)
            ode_res = scipy.integrate.solve_ivp(fun=fun, t_span=(0.,nperiod_anim), y0=yo, method=ODE_method, t_eval=t_eval, dense_output=False, events=None, vectorized=False,max_step=1./min_n_steps_ode,atol=atol_ode,rtol=rtol_ode)
            all_pos_vel = ode_res['y'].reshape(2,nbody,choreo.ndim,round(nint_plot_img*nperiod_anim))
            all_pos_ode = all_pos_vel[0,:,:,:]

            vid_filename = filename_output+'_unperturbed_'+str(i).zfill(3)+'.mp4'
            list_vid.append(vid_filename)

            choreo.plot_all_2D_anim(x,nint_plot_anim,callfun,vid_filename,nperiod_anim,Plot_trace=Plot_trace_anim,fig_size=vid_size_perturb,dnint=dnint,all_pos_trace=all_pos_ode,all_pos_points=all_pos_ode,xlim=xlim,extend=0.,color_list=color_list,color=color)

        for irank in range(n_perturbed):

            yo = choreo.Compute_init_pos_vel(x,callfun).reshape(-1) + dy_perturb * Instability_directions[irank,:]
            t_eval = np.array([i/nint_plot_img for i in range(round(nperiod_anim*nint_plot_img))])
            fun = lambda t,y: choreo.Compute_ODE_RHS(t,y,callfun)
            ode_res = scipy.integrate.solve_ivp(fun=fun, t_span=(0.,nperiod_anim), y0=yo, method=ODE_method, t_eval=t_eval, dense_output=False, events=None, vectorized=False,max_step=1./min_n_steps_ode,atol=atol_ode,rtol=rtol_ode)
            all_pos_vel = ode_res['y'].reshape(2,nbody,choreo.ndim,round(nint_plot_img*nperiod_anim))
            all_pos_ode = all_pos_vel[0,:,:,:]

            vid_filename = filename_output+'_perturbed_'+str(irank).zfill(3)+'.mp4'
            list_vid.append(vid_filename)
            
            choreo.plot_all_2D_anim(x,nint_plot_anim,callfun,vid_filename,nperiod_anim,Plot_trace=Plot_trace_anim,fig_size=vid_size_perturb,dnint=dnint,all_pos_trace=all_pos_ode,all_pos_points=all_pos_ode,xlim=xlim,extend=0.,color_list=color_list,color=color)

        for irank in range(n_random):

            yo = choreo.Compute_init_pos_vel(x,callfun).reshape(-1)
            dy = np.random.rand(*yo.shape)
            dy = dy / np.linalg.norm(dy)
            yo += dy_perturb*dy

            t_eval = np.array([i/nint_plot_img for i in range(round(nperiod_anim*nint_plot_img))])
            fun = lambda t,y: choreo.Compute_ODE_RHS(t,y,callfun)
            ode_res = scipy.integrate.solve_ivp(fun=fun, t_span=(0.,nperiod_anim), y0=yo, method=ODE_method, t_eval=t_eval, dense_output=False, events=None, vectorized=False,max_step=1./min_n_steps_ode,atol=atol_ode,rtol=rtol_ode)
            all_pos_vel = ode_res['y'].reshape(2,nbody,choreo.ndim,round(nint_plot_img*nperiod_anim))
            all_pos_ode = all_pos_vel[0,:,:,:]

            vid_filename = filename_output+'_random_'+str(irank).zfill(3)+'.mp4'
            list_vid.append(vid_filename)
            
            choreo.plot_all_2D_anim(x,nint_plot_anim,callfun,vid_filename,nperiod_anim,Plot_trace=Plot_trace_anim,fig_size=vid_size_perturb,dnint=dnint,all_pos_trace=all_pos_ode,all_pos_points=all_pos_ode,xlim=xlim,extend=0.,color_list=color_list,color=color)


        nxy = [nx,ny]
        ordering = 'RowMajor'
        # ordering = 'ColMajor'
        choreo.VideoGrid(list_vid,filename_output+'_perturbed.mp4',nxy=nxy,ordering=ordering)

        for the_file in list_vid:
            if os.path.isfile(the_file):
                os.remove(the_file)



    if InvestigateStability:

        print('')
        print('Symplectic integration of tangent system')
        print('')
        

        # SymplecticMethod = 'SymplecticEuler'
        # SymplecticMethod = 'SymplecticStormerVerlet'
        # SymplecticMethod = 'SymplecticRuth3'

        # the_integrators = {SymplecticMethod:choreo.GetSymplecticIntegrator(SymplecticMethod)}

        the_integrators = {
            # 'SymplecticEuler_XV'            : choreo.SymplecticEuler_XV,
            # 'SymplecticEuler_VX'            : choreo.SymplecticEuler_VX,
            # 'SymplecticStormerVerlet_XV'    : choreo.SymplecticStormerVerlet_XV_cython,
            # 'SymplecticStormerVerlet_VX'    : choreo.SymplecticStormerVerlet_VX_cython,
            'SymplecticRuth3_XV'            : choreo.SymplecticRuth3_XV,
            'SymplecticRuth3_VX'            : choreo.SymplecticRuth3_VX,
            }


        # the_integrators = choreo.all_unique_SymplecticIntegrators

        for SymplecticMethod,SymplecticIntegrator in the_integrators.items() :

            print('')
            print('SymplecticMethod : ',SymplecticMethod)
            print('')

            refinement_lvl = [64,128,256,512,1024]
            # refinement_lvl = [1,2,4,8,16,32,64,128]
            # refinement_lvl = [1,10,100]

            for imul in range(len(refinement_lvl)):

                nint_mul = refinement_lvl[imul]

                nint = callfun[0]['nint_list'][callfun[0]["current_cvg_lvl"]]*nint_mul

                fun,gun,x0,v0 = choreo.GetTangentSystemDef(x,callfun,nint,method=SymplecticMethod)

                ndof = nbody*choreo.ndim

                t_span = (0.,1.)

                t_beg= time.perf_counter_ns()
                xf,vf = SymplecticIntegrator(fun,gun,t_span,x0,v0,nint)
                t_end = time.perf_counter_ns()
                MonodromyMat = np.ascontiguousarray(np.concatenate((xf,vf),axis=0).reshape(2*ndof,2*ndof))

                # Evaluates the relative accuracy of the Monodromy matrix integration process
                # zo should be an eigenvector of the Monodromy matrix, with eigenvalue 1

                yo = choreo.Compute_init_pos_vel(x,callfun).reshape(-1)
                zo = choreo.Compute_Auto_ODE_RHS(yo,callfun)
                error_rel = np.linalg.norm(MonodromyMat.dot(zo)-zo)/np.linalg.norm(zo)

                
                Instability_magnitude,Instability_directions = choreo.InstabilityDecomposition(MonodromyMat)
                error_rel = np.linalg.norm(Instability_magnitude - np.flip(1/Instability_magnitude))/np.linalg.norm(Instability_magnitude)





                # print(f'error : {error_rel:e} time : {(t_end-t_beg)/One_sec:f}')

                if (imul > 0):
                    error_mul = error_rel/error_rel_prev
                    est_order = -m.log(error_mul)/m.log(refinement_lvl[imul]/refinement_lvl[imul-1])

                    # print(f'error : {error_rel:e}     error mul : {error_rel/error_rel_prev:e}     estimated order : {est_order:e}     time : {(t_end-t_beg)/One_sec:f}')
                    print(f'error : {error_rel:e}     estimated order : {est_order:.2f}     time : {(t_end-t_beg)/One_sec:f}')

                error_rel_prev = error_rel




        Instability_magnitude,Instability_directions = choreo.InstabilityDecomposition(MonodromyMat)
        print("Estimated min error ",np.finfo(float).eps * Instability_magnitude[0])



    if InvestigateIntegration:

        print('')
        print('Symplectic integration of direct system')
        print('')


        # SymplecticMethod = 'SymplecticEuler'
        # SymplecticMethod = 'SymplecticStormerVerlet'
        # SymplecticMethod = 'SymplecticRuth3'

        # the_integrators = {SymplecticMethod:choreo.GetSymplecticIntegrator(SymplecticMethod)}

        the_integrators = choreo.all_unique_SymplecticIntegrators

        for SymplecticMethod,SymplecticIntegrator in the_integrators.items() :

            print('')
            print('SymplecticMethod : ',SymplecticMethod)
            print('')

            # refinement_lvl = [1,2,4,8,16,32,64,128,256,512]
            refinement_lvl = [128,256,512]
            # refinement_lvl = [16,32]
            # refinement_lvl = [1,10,100]

            for imul in range(len(refinement_lvl)):

                nint_mul = refinement_lvl[imul]

                nint = callfun[0]['nint_list'][callfun[0]["current_cvg_lvl"]]*nint_mul

                ndim_ode = nbody*choreo.ndim

                all_pos_vel = choreo.ComputeAllPosVel(x,callfun)

                y0 = np.ascontiguousarray(all_pos_vel[:,:,:,0].reshape(-1))


                # t_span = (0.,0.5)
                # yf_exact =np.copy(all_pos_vel[:,:,:,callfun[0]['nint_list'][callfun[0]["current_cvg_lvl"]]//2].reshape(-1))
    

                t_span = (0.,1.)
                yf_exact = np.copy(y0)



                x0 = y0[0       :  ndim_ode]
                v0 = y0[ndim_ode:2*ndim_ode]

                xf_exact = yf_exact[0       :  ndim_ode]
                vf_exact = yf_exact[ndim_ode:2*ndim_ode]

                fun,gun = choreo.GetSymplecticODEDef(callfun)

                t_beg= time.perf_counter_ns()
                xf,vf = SymplecticIntegrator(fun,gun,t_span,x0,v0,nint)
                t_end = time.perf_counter_ns()

                # error_rel = np.linalg.norm(xf_exact-xf)+np.linalg.norm(vf_exact-vf)
                error_rel = np.linalg.norm(xf_exact-xf)/np.linalg.norm(xf_exact)+np.linalg.norm(vf_exact-vf)/np.linalg.norm(vf_exact)

                # print(f'error : {error_rel:e} time : {(t_end-t_beg)/One_sec:f}')

                if (imul > 0):
                    error_mul = error_rel/error_rel_prev
                    est_order = -m.log(error_mul)/m.log(refinement_lvl[imul]/refinement_lvl[imul-1])

                    # print(f'error : {error_rel:e}     error mul : {error_rel/error_rel_prev:e}     estimated order : {est_order:e}     time : {(t_end-t_beg)/One_sec:f}')
                    print(f'error : {error_rel:e}     estimated order : {est_order:.2f}     time : {(t_end-t_beg)/One_sec:f}')

                error_rel_prev = error_rel







if __name__ == "__main__":
    main()    
