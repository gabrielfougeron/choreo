import os
import itertools
import copy
import h5py
import time

import numpy as np
import math as m
import scipy.optimize as opt
import scipy.linalg as la
import scipy.sparse as sp
import sparseqr


from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import cnames
from matplotlib import animation

from Choreo_cython_funs import *


ndim = 2
n = -0.5  #coeff of x^2 in the potential power law
# ~ n = -1.  #coeff of x^2 in the potential power law

twopi = 2* np.pi
fourpi = 4 * np.pi
fourpisq = twopi*twopi

nnm1 = n*(n-1)
        
def proj_zero_mean(nloop,nbody,ncoeff,mass,all_coeffs):
    
    mass_cen = np.zeros((ndim),dtype=np.float64)
    mass_tot = 0.
    
    for il in range(nloop):
        
        mass_l =  mass[il]*nbody[il]
        mass_tot +=  mass_l
        mass_cen += mass_l * all_coeffs[il,:,0,0]

    mass_cen /= mass_tot
    
    # ~ print('Mass cent : ',mass_cen)
  
    for il in range(nloop):

        all_coeffs[il,:,0,0] -=  mass_cen

            
def setup_changevar(nloop,nbody,ncoeff,mass=None,MomCons=False,n_grad_change=1.,Sym_list=[]):
    
    eps_zero = 1e-14
    
    # il,idim,k,ift => ift + 2*(k + ncoeff*(idim + ndim*il))
    n_idx = nloop*ndim*ncoeff*2

    cstr_data = []
    cstr_row = []
    cstr_col = []

    icstr=0
    for il in range(nloop):
        for idim in range(ndim):
            
            i = 1 + 2*(0 + ncoeff*(idim + ndim*il))

            cstr_row.append(i)
            cstr_col.append(icstr)
            cstr_data.append(1.)            
            
            icstr +=1 
            
    if (MomCons):
        for k in range(ncoeff):
            NonZeroSym = False
            for il in range(nloop):
                NonZeroSym = NonZeroSym or ((k % nbody[il]) == 0)
                
            if (NonZeroSym):

                for idim in range(ndim):
                    for ift in range(2):
                        for il in range(nloop):
                            if ((k % nbody[il]) == 0):
                                
                                i = ift + 2*(k + ncoeff*(idim + ndim*il))

                                cstr_row.append(i)
                                cstr_col.append(icstr)
                                cstr_data.append(mass[il]*nbody[il])
                                
                        icstr+=1
    
    cs = np.zeros((2),dtype=np.float64)
    
    for Sym in Sym_list :
    
        for iln in range(len(Sym['LoopSelect'])):
            
            il = Sym['LoopSelect'][iln]
            ilp = Sym['LoopPerm'][iln]
            
            if (il == ilp):
                
                for k in range(ncoeff):
                    
                    if (Sym['TimeRev']):
                        cs[0] = np.cos(   twopi * k*Sym['TimeShift'])
                        cs[1] = np.sin(   twopi * k*Sym['TimeShift'])
                        
                        for idim in range(ndim):
                                
                            for jdim in range(ndim):
                                    
                                i =  0 + 2*(k + ncoeff*(jdim + ndim*il))

                                val = Sym['SpaceRot'][idim,jdim]*cs[0]
                                
                                if (idim == jdim):
                                    val -=1.
                                
                                if (abs(val) > eps_zero):
                                
                                    cstr_row.append(i)
                                    cstr_col.append(icstr)
                                    cstr_data.append(val)
                                    
                                i =  1 + 2*(k + ncoeff*(jdim + ndim*il))

                                val = Sym['SpaceRot'][idim,jdim]*cs[1]
                                
                                if (abs(val) > eps_zero):
                                
                                    cstr_row.append(i)
                                    cstr_col.append(icstr)
                                    cstr_data.append(val)
                                    
                            icstr+=1
                                
                            for jdim in range(ndim):
                                    
                                i =  1 + 2*(k + ncoeff*(jdim + ndim*il))

                                val = - Sym['SpaceRot'][idim,jdim]*cs[0]
                                
                                if (idim == jdim):
                                    val -=1.

                                if (abs(val) > eps_zero):
                                
                                    cstr_row.append(i)
                                    cstr_col.append(icstr)
                                    cstr_data.append(val)
                                    
                                i =  0 + 2*(k + ncoeff*(jdim + ndim*il))

                                val = Sym['SpaceRot'][idim,jdim]*cs[1]
                                
                                if (abs(val) > eps_zero):
                                
                                    cstr_row.append(i)
                                    cstr_col.append(icstr)
                                    cstr_data.append(val)
    
                            icstr+=1
                                                        
                    else:
                        cs[0] = np.cos(  - twopi * k*Sym['TimeShift'])
                        cs[1] = np.sin(  - twopi * k*Sym['TimeShift'])                        
                            
                        for idim in range(ndim):
                                
                            for jdim in range(ndim):
                                    
                                i =  0 + 2*(k + ncoeff*(jdim + ndim*il))

                                val = Sym['SpaceRot'][idim,jdim]*cs[0]
                                
                                if (idim == jdim):
                                    val -=1.

                                if (abs(val) > eps_zero):
                                
                                    cstr_row.append(i)
                                    cstr_col.append(icstr)
                                    cstr_data.append(val)
                                    
                                i =  1 + 2*(k + ncoeff*(jdim + ndim*il))

                                val = - Sym['SpaceRot'][idim,jdim]*cs[1]
                                
                                if (abs(val) > eps_zero):
                                
                                    cstr_row.append(i)
                                    cstr_col.append(icstr)
                                    cstr_data.append(val)
                                    
                            icstr+=1
                                
                            for jdim in range(ndim):
                                    
                                i =  1 + 2*(k + ncoeff*(jdim + ndim*il))

                                val = Sym['SpaceRot'][idim,jdim]*cs[0]
                                
                                if (idim == jdim):
                                    val -=1.

                                if (abs(val) > eps_zero):
                                
                                    cstr_row.append(i)
                                    cstr_col.append(icstr)
                                    cstr_data.append(val)
                                    
                                i =  0 + 2*(k + ncoeff*(jdim + ndim*il))

                                val = Sym['SpaceRot'][idim,jdim]*cs[1]
                                
                                if (abs(val) > eps_zero):
                                
                                    cstr_row.append(i)
                                    cstr_col.append(icstr)
                                    cstr_data.append(val)
    
                            icstr+=1
                            
            else:
        
                for k in range(ncoeff):
                    
                    if (Sym['TimeRev']):
                        cs[0] = np.cos(   twopi * k*Sym['TimeShift'])
                        cs[1] = np.sin(   twopi * k*Sym['TimeShift'])
                        
                        for idim in range(ndim):
                                
                            for jdim in range(ndim):
                                    
                                i =  0 + 2*(k + ncoeff*(jdim + ndim*ilp))

                                val = Sym['SpaceRot'][idim,jdim]*cs[0]
        
                                if (abs(val) > eps_zero):
                                
                                    cstr_row.append(i)
                                    cstr_col.append(icstr)
                                    cstr_data.append(val)
                                    
                                i =  1 + 2*(k + ncoeff*(jdim + ndim*ilp))

                                val = Sym['SpaceRot'][idim,jdim]*cs[1]

                                if (abs(val) > eps_zero):
                                
                                    cstr_row.append(i)
                                    cstr_col.append(icstr)
                                    cstr_data.append(val)

                            i =  0 + 2*(k + ncoeff*(idim + ndim*il))

                            val = -1.

                            cstr_row.append(i)
                            cstr_col.append(icstr)
                            cstr_data.append(val)

                            icstr+=1
                                
                            for jdim in range(ndim):
                                    
                                i =  1 + 2*(k + ncoeff*(jdim + ndim*ilp))

                                val = - Sym['SpaceRot'][idim,jdim]*cs[0]

                                if (abs(val) > eps_zero):
                                
                                    cstr_row.append(i)
                                    cstr_col.append(icstr)
                                    cstr_data.append(val)
                                    
                                i =  0 + 2*(k + ncoeff*(jdim + ndim*ilp))

                                val = Sym['SpaceRot'][idim,jdim]*cs[1]

                                if (abs(val) > eps_zero):
                                
                                    cstr_row.append(i)
                                    cstr_col.append(icstr)
                                    cstr_data.append(val)

                            i =  1 + 2*(k + ncoeff*(idim + ndim*il))

                            val = -1.
                        
                            cstr_row.append(i)
                            cstr_col.append(icstr)
                            cstr_data.append(val)
    
                            icstr+=1
                                                        
                    else:
                        cs[0] = np.cos(  - twopi * k*Sym['TimeShift'])
                        cs[1] = np.sin(  - twopi * k*Sym['TimeShift'])                        
                            
                        for idim in range(ndim):
                                
                            for jdim in range(ndim):
                                    
                                i =  0 + 2*(k + ncoeff*(jdim + ndim*ilp))

                                val = Sym['SpaceRot'][idim,jdim]*cs[0]
        
                                if (abs(val) > eps_zero):
                                
                                    cstr_row.append(i)
                                    cstr_col.append(icstr)
                                    cstr_data.append(val)
                                    
                                i =  1 + 2*(k + ncoeff*(jdim + ndim*ilp))

                                val = - Sym['SpaceRot'][idim,jdim]*cs[1]
                                
                                if (abs(val) > eps_zero):
                                
                                    cstr_row.append(i)
                                    cstr_col.append(icstr)
                                    cstr_data.append(val)
                                  
                            i =  0 + 2*(k + ncoeff*(idim + ndim*il))

                            val = -1.
                        
                            cstr_row.append(i)
                            cstr_col.append(icstr)
                            cstr_data.append(val)
                                
                            icstr+=1
                                
                            for jdim in range(ndim):
                                    
                                i =  1 + 2*(k + ncoeff*(jdim + ndim*ilp))

                                val = Sym['SpaceRot'][idim,jdim]*cs[0]

                                if (abs(val) > eps_zero):
                                
                                    cstr_row.append(i)
                                    cstr_col.append(icstr)
                                    cstr_data.append(val)
                                    
                                i =  0 + 2*(k + ncoeff*(jdim + ndim*ilp))

                                val = Sym['SpaceRot'][idim,jdim]*cs[1]
                                
                                if (abs(val) > eps_zero):
                                
                                    cstr_row.append(i)
                                    cstr_col.append(icstr)
                                    cstr_data.append(val)
    
                            i =  1 + 2*(k + ncoeff*(idim + ndim*il))

                            val = -1.
                                                        
                            cstr_row.append(i)
                            cstr_col.append(icstr)
                            cstr_data.append(val)
    
                            icstr+=1
              
    ncstr = icstr
    cstr_data = np.array(cstr_data,dtype=np.float64)
    cstr_row  = np.array(cstr_row,dtype=int)
    cstr_col  = np.array(cstr_col,dtype=int)
    
    cstrmat_sp =  sp.coo_matrix((cstr_data,(cstr_row,cstr_col)),shape=(n_idx,ncstr), dtype=np.float64)
    
    deb = time.time()
    param_to_coeff = null_space_sparseqr(cstrmat_sp)
    coeff_to_param = param_to_coeff.transpose(copy=True)
    
    for idx in range(param_to_coeff.nnz):
    
        res,ift = divmod(param_to_coeff.row[idx],2     )
        res,k   = divmod(res ,ncoeff)
        il ,idim= divmod(res ,ndim  )
    
        if (k >=2):
            kfac = pow(k,-n_grad_change)
        else:
            kfac = 1.
        
        param_to_coeff.data[idx] *= kfac
    
    for idx in range(coeff_to_param.nnz):
    
        res,ift = divmod(coeff_to_param.col[idx],2     )
        res,k   = divmod(res ,ncoeff)
        il ,idim= divmod(res ,ndim  )
    
        if (k >=2):
            kfac = pow(k,n_grad_change)
        else:
            kfac = 1.
        
        coeff_to_param.data[idx] *= kfac
    
    # ~ coeff_to_param = sp.identity(n_idx,dtype=np.float64,format="coo")
    # ~ param_to_coeff = sp.identity(n_idx,dtype=np.float64,format="coo")
    
    
    
    return coeff_to_param , param_to_coeff    


def plot_all_2D_anim(nloop,nbody,nint,nperiod,all_coeffs,filename,Plot_trace=True):
    
    all_pos = []
    all_shifts = []

    xmin = all_coeffs[0,0,0,0]
    xmax = xmin
    ymin = all_coeffs[0,1,0,0]
    ymax = ymin
    
    # Prepares data
    for il in range(nloop):
        
        c_coeffs = all_coeffs[il,:,:,:].view(dtype=np.complex128)[...,0]
        
        pos = np.zeros((ndim,nint+1),dtype=np.float64)
        
        pos[:,0:nint] = np.fft.irfft(c_coeffs,n=nint,axis=1)*nint
        
        pos[:,nint] = pos[:,0]
        
        all_pos.append(pos)
        
        div = nint // nbody[il]
        
        shift = np.zeros((nbody[il]),int)
        for i in range(nbody[il]):
            shift[i] = (-i*div)% nint
            
        all_shifts.append(shift)
        
        xmin = min(xmin,pos[0,...].min())
        xmax = max(xmax,pos[0,...].max())
        ymin = min(ymin,pos[1,...].min())
        ymax = max(ymax,pos[1,...].max())
    
    r = 0.03
    
    xinf = xmin - r*(xmax-xmin)
    xsup = xmax + r*(xmax-xmin)
    
    yinf = ymin - r*(ymax-ymin)
    ysup = ymax + r*(ymax-ymin)
    
    
        
    # Plot-related
    fig = plt.figure()
    ax = plt.gca()
    lines = sum([ax.plot([], [],'-')  for il in range(nloop)], [])
    points = sum([ax.plot([], [],'o')for il in range(nloop)], [])
    
    # ~ print(xinf,xsup)
    # ~ print(yinf,ysup)
    
    ax.axis('off')
    ax.set_xlim([xinf, xsup])
    ax.set_ylim([yinf, ysup ])
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    
    def init():
        
        if (Plot_trace):
            for il in range(nloop):
                lines[il].set_data(all_pos[il][0,:], all_pos[il][1,:])
        
        return lines + points

    def update(i):
        
        for il in range(nloop):
            
            for ib in range(nbody[il]):
                all_shifts[il][ib] = (all_shifts[il][ib]+1) % nint
            
            points[il].set_data(all_pos[il][0,all_shifts[il]], all_pos[il][1,all_shifts[il]])

        return lines + points

    anim = animation.FuncAnimation(fig, update, frames=nperiod*nint,init_func=init, blit=True)
                        
    # Save as mp4. This requires mplayer or ffmpeg to be installed
    # ~ anim.save(filename, fps=30, extra_args=['-vcodec', 'libx264'])
    anim.save(filename, fps=30)
    
    plt.close()
    
def plot_all_2D(nloop,nbody,nint,all_coeffs,filename):
    
    all_shifts = []

    c_coeffs = all_coeffs.view(dtype=np.complex128)[...,0]
    
    all_pos = np.zeros((nloop,ndim,nint+1),dtype=np.float64)
    all_pos[:,:,0:nint] = np.fft.irfft(c_coeffs,n=nint,axis=2)*nint
    all_pos[:,:,nint] = all_pos[:,:,0]

    # Prepares data
    for il in range(nloop):
        
        div = nint // nbody[il]
        
        shift = np.zeros((nbody[il]),int)
        for i in range(nbody[il]):
            shift[i] = (-i*div)% nint
            
        all_shifts.append(shift)
        
    xmin = all_pos[:,0,:].min()
    xmax = all_pos[:,0,:].max()
    ymin = all_pos[:,1,:].min()
    ymax = all_pos[:,1,:].max()
    
    r = 0.03
    
    xinf = xmin - r*(xmax-xmin)
    xsup = xmax + r*(xmax-xmin)
    
    yinf = ymin - r*(ymax-ymin)
    ysup = ymax + r*(ymax-ymin)

        
    # Plot-related
    fig = plt.figure()
    ax = plt.gca()
    lines = sum([ax.plot([], [],'-')  for il in range(nloop)], [])
    points = sum([ax.plot([], [],'o')for il in range(nloop)], [])
    
    # ~ print(xinf,xsup)
    # ~ print(yinf,ysup)
    
    ax.axis('off')
    ax.set_xlim([xinf, xsup])
    ax.set_ylim([yinf, ysup ])
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    
    for il in range(nloop):
        lines[il].set_data(all_pos[il,0,:], all_pos[il,1,:])

    plt.savefig(filename)
    
    plt.close()
          
# ~ def plot_Newt_err(all_Newt_err,filename):    
    
    # ~ nloop = len(all_Newt_err)
    # ~ nbody = np.zeros((nloop),int)
    # ~ for il in range(nloop):
        # ~ nbody[il] = all_Newt_err[il].shape[0]
    
    # ~ nint = all_Newt_err[0].shape[2]

    # ~ int_times = np.linspace(start = 0.,stop=1.,num=nint,endpoint=False)
    
    # ~ # Plot-related
    # ~ fig = plt.figure()
    # ~ ax = plt.gca()
    
    
    # ~ for il in range(nloop):
        # ~ for ib in range(nbody[il]):
            # ~ y = np.linalg.norm(all_Newt_err[il][ib,:,:],axis=0)

            # ~ ax.semilogy(int_times,y,'-')

    # ~ plt.savefig(filename)
    
    # ~ plt.close()
   
def Package_args(nloop,nbody,ncoeff,mass,nint,all_coeffs,coeff_to_param,param_to_coeff):
    
    callfun=[{
    'nloop'                 : nloop     ,
    'nbody'                 : nbody     ,
    'ncoeff'                : ncoeff    ,
    'mass'                  : mass      ,
    'nint'                  : nint      ,
    'coeff_to_param'        : coeff_to_param      ,
    'param_to_coeff'        : param_to_coeff      ,
        }]
    
    y = all_coeffs.reshape(-1)
    x = coeff_to_param.dot(y)
    
    return x,callfun
    
def Unpackage_all_coeffs(x,callfun):
    
    args=callfun[0]
    
    y = args['param_to_coeff'] * x
    all_coeffs = y.reshape(args['nloop'],ndim,args['ncoeff'],2)
    
    return all_coeffs
    
def Compute_action_package(x,callfun):

    args=callfun[0]
    
    y = args['param_to_coeff'] * x
    all_coeffs = y.reshape(args['nloop'],ndim,args['ncoeff'],2)

    J,GradJ =  Compute_action(args['nloop'],args['nbody'],args['ncoeff'],args['mass'],args['nint'],all_coeffs)

    GJ = GradJ.reshape(-1)
    y = GJ * args['param_to_coeff']
    
    return J,y
    
def Compute_action_onlygrad_package(x,callfun):

    J,y = Compute_action_package(x,callfun)
    
    return y
    
def Compute_action_hess_mul_package(x,dx,callfun):
    
    args=callfun[0]
    
    y = args['param_to_coeff'] * x
    all_coeffs = y.reshape(args['nloop'],ndim,args['ncoeff'],2)

    dy = args['param_to_coeff'] * dx
    all_coeffs_d = dy.reshape(args['nloop'],ndim,args['ncoeff'],2)

    HessJdx =  Compute_action_hess_mul(args['nloop'],args['nbody'],args['ncoeff'],args['mass'],args['nint'],all_coeffs,all_coeffs_d)
    
    HJdx = HessJdx.reshape(-1)
    
    z = HJdx * args['param_to_coeff']
    
    return z
    
def Compute_action_hess_LinOpt_package(x,callfun):

    args=callfun[0]

    return sp.linalg.LinearOperator((args['n_idx'],args['n_idx']),
        matvec =  (lambda dx,xl=x,callfunl=callfun : Compute_action_hess_mul_package(xl,dx,callfunl)),
        rmatvec = (lambda dx,xl=x,callfunl=callfun : Compute_action_hess_mul_package(xl,dx,callfunl)))
    
def Current_disp(transform,callfun):
    
    args = callfun[0]

    return sq_dist_transform_2d_noscal(args['nloop'],args['ncoeff'],args['all_coeffs'],args['all_coeffs2'],transform)
    
def Current_disp_funonly(transform,callfun):
    
    f,g = Current_disp(transform,callfun)
    
    return f
    
def Current_disp_gradonly(transform,callfun):
    
    f,g = Current_disp(transform,callfun)
    
    return g
    
def Compute_Dist_loops_local(all_coeffs1,all_coeffs2,init_transform = np.zeros((4),dtype=np.float64)):
    
    nloop1 = all_coeffs1.shape[0]
    ndim1 = all_coeffs1.shape[1]
    ncoeff1 = all_coeffs1.shape[2]

    nloop2 = all_coeffs2.shape[0]
    ndim2 = all_coeffs2.shape[1]
    ncoeff2 = all_coeffs2.shape[2]

    if ((nloop1 != nloop2)):
        raise ValueError('Different number of loops')

    nloop = nloop1

    if ((ndim1 != ndim2) or (ndim1 != ndim)):
        raise ValueError('Number of space dimension error')

    if (ncoeff1 != ncoeff2):
        raise ValueError('Different number of coefficients')
    
    ncoeff = ncoeff1
    
    callfun = [{
    'nloop'         : nloop,
    'ncoeff'        : ncoeff,
    'all_coeffs'    : all_coeffs1,
    'all_coeffs2'   : all_coeffs2,
    }]
    
    opt_result = opt.minimize(fun=Current_disp,x0=init_transform,args=callfun,method='L-BFGS-B',jac=True,options={'disp':False,'maxiter':100,'ftol' : 1e-13,'gtol': 1e-10,})

    return opt_result['fun']

def SelectFiles_Action(store_folder,Action_val,Action_eps):
    
    Action_msg = 'Value of the Action : '
    Action_msg_len = len(Action_msg)
    
    file_path_list = []
    for file_path in os.listdir(store_folder):
        file_path = os.path.join(store_folder, file_path)
        file_root, file_ext = os.path.splitext(os.path.basename(file_path))
        
        if (file_ext == '.txt' ):
            # ~ print(file_path)
            with open(file_path,'r') as file_read:
                file_readlines = file_read.readlines()
                for iline in range(len(file_readlines)):
                    line = file_readlines[iline]
                    
                    if (line[0:Action_msg_len] == Action_msg):
                        This_Action = float(line[Action_msg_len:])
                        # ~ print(This_Action)
                        
                        if (abs(This_Action-Action_val) < Action_eps):
                            
                            # ~ print(store_folder+'/'+file_root+'.npy')
                            
                            file_path_list.append(store_folder+'/'+file_root+'.npy')
                            
    return file_path_list

def Check_Duplicates(store_folder,all_coeffs,nbody,duplicate_eps,Action_val=-1.,Action_eps=1e-5,ncoeff_cutoff=-1,theta_rot_dupl=[],dt_shift_dupl=[],TimeReversal=False,SpaceSym=False):


    if (Action_val > 0):
        
        file_path_list = SelectFiles_Action(store_folder,Action_val,Action_eps)
        
    else:
        file_path_list = []
        for file_path in os.listdir(store_folder):
            file_path = os.path.join(store_folder, file_path)
            file_root, file_ext = os.path.splitext(os.path.basename(file_path))
            
            if (file_ext == '.npy' ):
                # ~ print(file_path)
                file_path_list.append(file_path)
            


    # ~ print('In Find Duplicates :')
    # ~ print('List of files : ',file_path_list)
    
    Found_duplicate = False
    dist_sols = 1e100
    file_path = ''
        
    nloops = all_coeffs.shape[0]
    for perm in itertools.permutations(range(nloops)):
        
        # ~ print('perm : ',perm)
        
        if (ncoeff_cutoff == -1):
            
            all_coeffs_test = np.copy(all_coeffs[perm,:,:,:])
            
        else:
        
            cut = min(all_coeffs.shape[2],ncoeff_cutoff)
            all_coeffs_test = np.copy(all_coeffs[perm,:,0:cut,:])
            
        new_cutoff = all_coeffs_test.shape[2]
        
        dt_shift_loop = np.zeros((nloops),dtype=np.float64)
        
        for itshift in itertools.product(*[range(nbody[il]) for il in range(1,nloops)]): #loop 0 does not move.
            
            # ~ print('itshift',itshift)
            
            for il in range(1,nloops):
                dt_shift_loop[il] =  itshift[il-1] / nbody[il]
                    
            t_shift_loop_indep(nloops,new_cutoff,dt_shift_loop,all_coeffs_test)

            Found_duplicate,dist_sols_new,file_path_new = Check_Duplicates_nosym(file_path_list,all_coeffs_test,duplicate_eps,new_cutoff,theta_rot_dupl,dt_shift_dupl)
            
            # ~ print(dist_sols_new)

            if (dist_sols_new < dist_sols):
                dist_sols = dist_sols_new
                file_path = file_path_new
            
            if (Found_duplicate):
                break

            if (TimeReversal):
                all_coeffs_test[:,:,:,1] = - all_coeffs_test[:,:,:,1]
                Found_duplicate,dist_sols_new,file_path_new = Check_Duplicates_nosym(file_path_list,all_coeffs_test,duplicate_eps,new_cutoff,theta_rot_dupl,dt_shift_dupl)
                
                # ~ print(dist_sols_new)
                
                if (dist_sols_new < dist_sols):
                    dist_sols = dist_sols_new
                    file_path = file_path_new
                all_coeffs_test[:,:,:,1] = - all_coeffs_test[:,:,:,1]
                
            if (Found_duplicate):
                break
            
            if (SpaceSym):
                all_coeffs_test[:,0,:,:] = - all_coeffs_test[:,0,:,:]
                Found_duplicate,dist_sols_new,file_path_new = Check_Duplicates_nosym(file_path_list,all_coeffs_test,duplicate_eps,new_cutoff,theta_rot_dupl,dt_shift_dupl)
                
                # ~ print(dist_sols_new)
                
                if (dist_sols_new < dist_sols):
                    dist_sols = dist_sols_new
                    file_path = file_path_new
                all_coeffs_test[:,0,:,:] = - all_coeffs_test[:,0,:,:]
            
            if (Found_duplicate):
                break
            
            if ((SpaceSym) and (TimeReversal)):
                all_coeffs_test[:,:,:,1] = - all_coeffs_test[:,:,:,1]
                all_coeffs_test[:,0,:,:] = - all_coeffs_test[:,0,:,:]
                Found_duplicate,dist_sols_new,file_path_new = Check_Duplicates_nosym(file_path_list,all_coeffs_test,duplicate_eps,new_cutoff,theta_rot_dupl,dt_shift_dupl)
                
                # ~ print(dist_sols_new)
                
                if (dist_sols_new < dist_sols):
                    dist_sols = dist_sols_new
                    file_path = file_path_new
                all_coeffs_test[:,:,:,1] = - all_coeffs_test[:,:,:,1]
                all_coeffs_test[:,0,:,:] = - all_coeffs_test[:,0,:,:]
                
            if (Found_duplicate):
                break
                
        if (Found_duplicate):
            break

    return Found_duplicate,dist_sols,file_path

def Check_Duplicates_nosym(file_path_list,all_coeffs,duplicate_eps,ncoeff_cutoff=-1,theta_rot_dupl=[],dt_shift_dupl=[]):

    Found_duplicate = False
    dist_sols = 1e100
    file_path_min = ''

    for file_path in file_path_list:

        # ~ print('Testing found solution against '+file_path)
        
        all_coeffs_old = np.load(file_path)
        
        if (ncoeff_cutoff == -1):
            cut = min(all_coeffs.shape[2],all_coeffs_old.shape[2])
        else:
            cut = min(all_coeffs.shape[2],all_coeffs_old.shape[2],ncoeff_cutoff)
            
        all_coeffs_load = np.copy(all_coeffs_old[:,:,0:cut,:])
        
        for theta in theta_rot_dupl:
            for dt_shift in dt_shift_dupl:
                
                init_transform = np.zeros((4),dtype=np.float64)
                init_transform[2] = theta
                init_transform[3] = dt_shift
                
                # ~ print('rot trot')
                # ~ print(theta,dt_shift)
        
                dist_sols_new = Compute_Dist_loops_local(all_coeffs_load,all_coeffs,init_transform=init_transform)
                
                # ~ print(dist_sols_new)
                # ~ print('')
                
                if (dist_sols_new < dist_sols):
                    dist_sols = dist_sols_new
                    file_path_min = file_path
                
                Found_duplicate = (dist_sols < duplicate_eps)

                if Found_duplicate:
                    break
            
            if Found_duplicate:
                break
                
        if Found_duplicate:
            break
            
    return Found_duplicate,dist_sols,file_path_min

def Write_Descriptor(nloop,nbody,ncoeff,mass,nint,all_coeffs,filename):
    with open(filename,'w') as filename_write:
        
        filename_write.write('Number of loops : {:d}\n'.format(nloop))
        
        filename_write.write('Number of bodies in each loop : ')
        for il in range(nloop):
            filename_write.write(' {:d}'.format(nbody[il]))
        filename_write.write('\n')
        
        filename_write.write('Mass of those bodies : ')
        for il in range(nloop):
            filename_write.write(' {:f}'.format(mass[il]))
        filename_write.write('\n')
        
        filename_write.write('Number of Fourier coefficients in each loop : {:d}\n'.format(ncoeff))
        filename_write.write('Number of integration points for the action : {:d}\n'.format(nint))
        
        Action,Gradaction = Compute_action(nloop,nbody,ncoeff,mass,nint,all_coeffs)
        
        filename_write.write('Value of the Action : {:.10f}\n'.format(Action))
        filename_write.write('Value of the Norm of the Gradient of the Action : {:.10E}\n'.format(np.linalg.norm(Gradaction)))

        Newt_err = Compute_Newton_err(nloop,nbody,ncoeff,mass,nint,all_coeffs)
        Newt_err_norm = np.linalg.norm(Newt_err)
        filename_write.write('Sum of Newton Errors : {:.10E}\n'.format(Newt_err_norm))
        
        dxmin = Compute_mindist(nloop,nbody,ncoeff,nint,all_coeffs)
        filename_write.write('Minimum inter-body distance : {:.10E}\n'.format(dxmin))
        
def null_space_sparseqr(AT):
    # AT must be in COO format
    # The nullspace of the TRANSPOSE of AT will be returned
    
    Q, R, E, rank = sparseqr.qr( AT )
    
    nrow = AT.shape[0]
    
    if (nrow <= rank):
        # ~ raise ValueError("Kernel is empty")
        
        return sp.coo_matrix(([],([],[])),shape=(nrow,0))
    
    else:
        
        iker = 0
        while(Q.col[iker] < rank):
            iker+=1
            
        return sp.coo_matrix((Q.data[iker:],(Q.row[iker:],Q.col[iker:]-rank)),shape=(nrow,nrow-rank))

class current_best:
    
    def __init__(self,x,f):
        
        self.x = x
        self.f = f
        self.f_norm = np.linalg.norm(f)
        
    def update(self,x,f):
        
        f_norm = np.linalg.norm(f)
        
        if (f_norm < self.f_norm):
            self.x = x
            self.f = f
            self.f_norm = f_norm

    def get_best(self):
        return self.x,self.f,self.f_norm
        
def Make2DSymOneLoop(SymType,loop):
    
    # ~ SymType  => Name of sym, see https://arxiv.org/abs/1305.0470
        # ~ 'name'
        # ~ 'n'
        # ~ 'k'
        # ~ 'l'
        
        # Classification :
        # C(n,k,l) with k and l relative primes
        # D(n,k,l) with k and l relative primes
        # Cp(n,2,#) 
        # Dp(n,1,#) 
        # Dp(n,2,#) 
        
        
    SymGens = []
    
    if ((SymType['name'] == 'C') or (SymType['name'] == 'D')):
        
        rot_angle = twopi * SymType['l'] /  SymType['k']
        s = 1
        TimeRev = False
        TimeShift = - 1 / SymType['k']
        
        SymGens.append ({
            'LoopSelect' : [loop],
            'LoopPerm' : [loop],
            'SpaceRot' : np.array([[s*np.cos(rot_angle),-s*np.sin(rot_angle)],[np.sin(rot_angle),np.cos(rot_angle)]],dtype=np.float64),
            'TimeRev' : TimeRev,
            'TimeShift' : TimeShift,
            })
    
    if (SymType['name'] == 'D'):
        
        rot_angle = 0
        s = -1
        TimeRev = True
        TimeShift = 0
        
        SymGens.append ({
            'LoopSelect' : [loop],
            'LoopPerm' : [loop],
            'SpaceRot' : np.array([[s*np.cos(rot_angle),-s*np.sin(rot_angle)],[np.sin(rot_angle),np.cos(rot_angle)]],dtype=np.float64),
            'TimeRev' : TimeRev,
            'TimeShift' : TimeShift,
            })
            
    if ((SymType['name'] == 'Cp') or ((SymType['name'] == 'Dp') and (SymType['k'] == 2))):
        
        rot_angle = 0
        s = -1
        TimeRev = False
        TimeShift =  1/2
        
        SymGens.append ({
            'LoopSelect' : [loop],
            'LoopPerm' : [loop],
            'SpaceRot' : np.array([[s*np.cos(rot_angle),-s*np.sin(rot_angle)],[np.sin(rot_angle),np.cos(rot_angle)]],dtype=np.float64),
            'TimeRev' : TimeRev,
            'TimeShift' : TimeShift,
            })

    if (SymType['name'] == 'Dp'):
        
        rot_angle = np.pi
        s = 1
        TimeRev = True
        TimeShift = 0
        
        SymGens.append ({
            'LoopSelect' : [loop],
            'LoopPerm' : [loop],
            'SpaceRot' : np.array([[s*np.cos(rot_angle),-s*np.sin(rot_angle)],[np.sin(rot_angle),np.cos(rot_angle)]],dtype=np.float64),
            'TimeRev' : TimeRev,
            'TimeShift' : TimeShift,
            })
            
    
    
    return SymGens
    
    
    
