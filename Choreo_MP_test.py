import numpy as np
import math as m
import scipy.optimize as opt
import scipy.sparse as sp
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import cnames
from matplotlib import animation
import copy
import os, shutil
import time



from Choreo_funs import *


nbody = np.array([3])
# ~ nbody = np.array([5])
# ~ nbody = np.array([4])

nloop = nbody.size
mass = np.ones((nloop))


store_folder = './Sniff_all/'
store_folder = store_folder+str(nbody[0])
for i in range(len(nbody)-1):
    store_folder = store_folder+'x'+str(nbody[i+1])


Look_for_duplicates = True
# ~ Look_for_duplicates = False

Check_loop_dist = True
# ~ Check_loop_dist = False

save_init = False
# ~ save_init = True

save_approx = False
# ~ save_approx = True

# ~ Reconverge_sols = False
Reconverge_sols = True


n_reconverge_it_max = 4

# ~ theta_rot_dupl = [0.,.2,.4,.6,.8]
theta_rot_dupl = np.linspace(start=0.,stop=twopi,endpoint=False,num=nbody[0])
dt_shift_dupl = np.linspace(start=0.,stop=1.,endpoint=False,num=nbody[0])

# ~ print(1/0)

# ~ ncoeff_init = 100
# ~ ncoeff_init = 300
ncoeff_init = 600
# ~ ncoeff_init = 90

ncoeff_cutoff = ncoeff_init

change_var_coeff =1.
# ~ change_var_coeff =0.

n_opt = 0

disp_scipy_opt = False
# ~ disp_scipy_opt = True


duplicate_eps = 1e-9








    

all_coeffs_a = np.load('./Sniff_all/3bis/1.npy')
all_coeffs_b = np.load('./Sniff_all/3bis/2.npy')



nloop_a = all_coeffs_a.shape[0]
nloop_b = all_coeffs_b.shape[0]
  

if not(nloop_a == nloop_b):
    raise ValueError('Solutions don\'t have the same number of loops')
    
nloop = nloop_a

ncoeff_a = all_coeffs_a.shape[2]  
ncoeff_b = all_coeffs_a.shape[2]  


print(ncoeff_a,ncoeff_b)
# ~ ncoeff_new = ncoeff *2
ncoeff_new = 600

all_coeffs_a_new = np.zeros((nloop,ndim,ncoeff_new,2),dtype=np.float64)
all_coeffs_b_new = np.zeros((nloop,ndim,ncoeff_new,2),dtype=np.float64)
for k in range(min(ncoeff_a,ncoeff_new)):
    all_coeffs_a_new[:,:,k,:] = all_coeffs_a[:,:,k,:]
for k in range(min(ncoeff_b,ncoeff_new)):
    all_coeffs_b_new[:,:,k,:] = all_coeffs_b[:,:,k,:]
    
all_coeffs_a = all_coeffs_a_new
all_coeffs_b = all_coeffs_b_new

        
ncoeff = ncoeff_new

nint = 2*ncoeff
n_idx,all_idx =  setup_idx(nloop,nbody,ncoeff)


VelChangeVar = {
'direct' : lambda nloop,nbody,ncoeff,all_coeffs :  VelChangeDirect(nloop,nbody,ncoeff,all_coeffs,change_var_coeff),
'inverse' : lambda nloop,nbody,ncoeff,all_coeffs :  VelChangeInverse(nloop,nbody,ncoeff,all_coeffs,change_var_coeff),
'Grad' : lambda nloop,nbody,ncoeff,all_idx,Action_grad :  VelChangeGrad(nloop,nbody,ncoeff,all_idx,Action_grad,change_var_coeff),
}

xa,callfun = Package_args(nloop,nbody,ncoeff,mass,nint,all_coeffs_a,n_idx,all_idx,VelChangeVar)
xb,callfun = Package_args(nloop,nbody,ncoeff,mass,nint,all_coeffs_b,n_idx,all_idx,VelChangeVar)


npts = 50
ndigits = 3


xMP,callfun = Package_args_MP(nloop,nbody,ncoeff,mass,nint,all_coeffs_a,all_coeffs_b,n_idx,all_idx,VelChangeVar,npts)



rigid_o = 1e7

nopt = 10
for i in range(nopt):
    
    print('iopt',i)
    print('')
    
    dt0 = np.linalg.norm(xa-xMP[0:n_idx])
    callfun[0]['elastic_wei'][0] = rigid_o * dt0
    # ~ callfun[0]['elastic_wei'][0] = 0
    
    # ~ print(dt0**2)
    # ~ print(1/0)
    
    for i in range(1,npts):
        dti = np.linalg.norm(xMP[(i-1)*n_idx:i*n_idx]-xMP[(i)*n_idx:(i+1)*n_idx])
        callfun[0]['elastic_wei'][i] = rigid_o * dti
        # ~ callfun[0]['elastic_wei'][i] = 0

    dtF = np.linalg.norm(xMP[(npts-1)*n_idx:npts*n_idx]-xb)
    callfun[0]['elastic_wei'][npts] = rigid_o * dtF
    # ~ callfun[0]['elastic_wei'][npts] = 0

    gradtol = 1e-3
    maxiter = 20

    # ~ gradtol = 1e-7
    # ~ maxiter = 1000
    opt_result = opt.minimize(fun=Compute_MP_action_package,x0=xMP,args=callfun,method='trust-krylov',jac=True,hessp=Compute_MP_action_hess_mul_package,options={'disp':True,'maxiter':maxiter,'gtol' : gradtol,'inexact': True})


    xMP = opt_result['x']

xMP_opt = xMP
# ~ xMP_opt = opt_result['x']






iplot = 0

tlist = []
actionlist = []
actionpostoptimlist = []

to = np.linalg.norm(xa-xMP_opt[0:n_idx])
t = to

Action_max = 0


for i in range(1,npts+1):
    
    x = xMP_opt[(i-1)*n_idx:i*n_idx]
    
    Action,Gradaction = Compute_action_package(x,callfun)
    
    print('t : ',t)
    print('Action : ',Action)
    print('Grad Action Norm : ',np.linalg.norm(Gradaction))
    
    tlist.append(t)
    actionlist.append(Action)
    
    if (Action > Action_max):
        Action_max = Action
        imax = i
    
    
    # ~ all_coeffs = Unpackage_all_coeffs(x,callfun)
    # ~ nint_plot = 200
    # ~ filename_output = './testplots/'+(str(iplot).zfill(ndigits))
    # ~ plot_all_2D(nloop,nbody,nint_plot,all_coeffs,filename_output+'.png')
    
    # ~ iplot+=1
    
    
    
    maxiter = 5000
    gradtol = 1e-7
    krylov_method = 'lgmres'
    opt_result = opt.root(fun=Compute_action_onlygrad_package,x0=x,args=callfun,method='krylov', options={'disp':True,'maxiter':maxiter,'fatol':gradtol,'jac_options':{'method':krylov_method}})
    
    # ~ opt_result = opt.root(fun=Compute_action_onlygrad_package,x0=x,args=callfun,method='df-sane', options={'disp':disp_scipy_opt,'maxfev':maxiter,'fatol':gradtol})

    x_opt = opt_result['x']
    Action,Gradaction = Compute_action_package(x_opt,callfun)
    print('Goes to :')
    print('Action : ',Action)
    print('Grad Action Norm : ',np.linalg.norm(Gradaction))
    
    actionpostoptimlist.append(Action)

    
    
    all_coeffs = Unpackage_all_coeffs(x_opt,callfun)
    nint_plot = 200
    filename_output = './testplots/'+(str(iplot).zfill(ndigits))
    plot_all_2D(nloop,nbody,nint_plot,all_coeffs,filename_output+'.png')
    
    iplot+=1
    
    
    if not(i == npts):
        t += np.linalg.norm(xMP_opt[(i-1)*n_idx:i*n_idx]-xMP_opt[(i)*n_idx:(i+1)*n_idx])
    
    print('')



fig = plt.figure()
ax = plt.gca()
plt.plot(tlist,actionlist,'-o')
plt.plot(tlist,actionpostoptimlist)
filename = './MP_Action.png'
plt.savefig(filename)



maxiter = 5000
gradtol = 1e-10
krylov_method = 'lgmres'
opt_result = opt.root(fun=Compute_action_onlygrad_package,x0=xMP_opt[(imax-1)*n_idx:imax*n_idx],args=callfun,method='krylov', options={'disp':False,'maxiter':maxiter,'fatol':gradtol,'jac_options':{'method':krylov_method}})
x_opt = opt_result['x']
Action,Gradaction = Compute_action_package(x_opt,callfun)
print('Goes to :')
print('Action : ',Action)
print('Grad Action Norm : ',np.linalg.norm(Gradaction))


filename_output = './MP_result'
print('Saving solution as '+filename_output+'.*')

all_coeffs = Unpackage_all_coeffs(x_opt,callfun)

nint_plot = 1000
nperiod = 2
np.save(filename_output+'.npy',all_coeffs)
plot_all_2D(nloop,nbody,nint_plot,all_coeffs,filename_output+'.png')
# ~ plot_all_2D_anim(nloop,nbody,nint_plot,nperiod,all_coeffs,filename_output+'.mp4')




'''
gradtol = 1e-8
maxiter = 1000
krylov_method = 'lgmres'
# ~ krylov_method = 'gmres'
# ~ krylov_method = 'bicgstab'
# ~ krylov_method = 'cgs'
# ~ krylov_method = 'minres'
opt_result = opt.root(fun=Compute_action_onlygrad_package,x0=x0,args=callfun,method='krylov', options={'disp':disp_scipy_opt,'maxiter':maxiter,'fatol':gradtol,'jac_options':{'method':krylov_method,'inner_M':Preco}})
opt_result = opt.root(fun=Compute_action_onlygrad_package,x0=x0,args=callfun,method='krylov', options={'disp':disp_scipy_opt,'maxiter':maxiter,'fatol':gradtol,'jac_options':{'method':krylov_method}})


# ~ y = sp.linalg.lgmres(mat,Gradaction,tol=lintol,M=None)[0]

# ~ print(y)


# ~ print(1/0)

# ~ all_coeffs = Unpackage_all_coeffs(x0,callfun)
# ~ Newt_err = Compute_Newton_err(nloop,nbody,ncoeff,mass,nint,all_coeffs,n_idx,all_idx)
# ~ Newt_err_norm = 0.
# ~ for ib in range(len(Newt_err)):
    # ~ Newt_err_norm += np.linalg.norm(Newt_err[ib])

# ~ print('Newton Error : ',Newt_err_norm)


# ~ gradtol = 1e-3
# ~ maxiter = 2000

# ~ opt_result = opt.minimize(fun=Compute_action_gradnormsq_package,x0=x0,args=callfun,method='L-BFGS-B',jac=True,options={'disp':True,'maxiter':maxiter,'gtol' : gradtol})


# ~ x0 = opt_result['x']
# ~ maxiter = 10
# ~ gradtol = 0
# ~ krylov_method = 'lgmres'
# ~ krylov_method = 'gmres'
# ~ krylov_method = 'bicgstab'
# ~ krylov_method = 'cgs'
# ~ krylov_method = 'minres'
# ~ opt_result = opt.root(fun=Compute_action_gradnormsq_gradonly_package,x0=x0,args=callfun,method='krylov', options={'disp':True,'maxiter':maxiter,'fatol':gradtol,'jac_options':{'method':krylov_method}})





# ~ opt_result = opt.minimize(fun=Compute_action_package,x0=x0,args=callfun,method='trust-krylov',jac=True,hessp=Compute_action_hess_mul_package,options={'disp':False,'maxiter':maxiter,'gtol' : gradtol,'inexact': True})
# ~ x0 = opt_result['x']


maxiter = 100
gradtol = 0
gradtol = 1e-14
krylov_method = 'lgmres'
# ~ krylov_method = 'gmres'
# ~ krylov_method = 'bicgstab'
# ~ krylov_method = 'cgs'
# ~ krylov_method = 'minres'
# ~ opt_result = opt.root(fun=Compute_action_onlygrad_package,x0=x0,args=callfun,method='krylov', options={'disp':disp_scipy_opt,'maxiter':maxiter,'fatol':gradtol,'jac_options':{'method':krylov_method}})
# ~ opt_result = opt.root(fun=Compute_action_onlygrad_package,x0=x0,args=callfun,method='krylov', options={'disp':disp_scipy_opt,'maxiter':maxiter,'fatol':gradtol,'jac_options':{'method':krylov_method,'inner_M':invmat}})
# ~ opt_result = opt.root(fun=Compute_action_onlygrad_package,x0=x0,args=callfun,method='df-sane', options={'disp':disp_scipy_opt,'maxfev':maxiter,'fatol':gradtol})





# ~ opt_result = opt.root(fun=Compute_action_onlygrad_package,x0=x0,args=callfun,method='lm',jac=Compute_action_dense_hess_package, options={'disp':True,'col_deriv':True,'ftol':gradtol,'xtol':1e-13})

# ~ opt_result = opt.root(fun=Compute_action_onlygrad_package,x0=x0,args=callfun,method='krylov', options={'disp':True,'maxiter':maxiter,'fatol':gradtol,'jac_options':{'method':krylov_method}})


# ~ opt_result = opt.root(fun=Compute_action_onlygrad_package,x0=x0,args=callfun,method='df-sane', options={'disp':True,'maxfev':maxiter,'fatol':gradtol})






x_opt = opt_result['x']

Action,Gradaction = Compute_action_package(x_opt,callfun )
print('Final : ',Action,np.linalg.norm(Gradaction))


all_coeffs = Unpackage_all_coeffs(x_opt,callfun)
Newt_err = Compute_Newton_err(nloop,nbody,ncoeff,mass,nint,all_coeffs,n_idx,all_idx)
Newt_err_norm = 0.
for ib in range(len(Newt_err)):
    Newt_err_norm += np.linalg.norm(Newt_err[ib])

print('Newton Error : ',Newt_err_norm)








# ~ all_coeffs = Unpackage_all_coeffs(x_opt,callfun)

filename_output = './opt_result'
print('Saving solution as '+filename_output+'.*')

nint_plot = 1000
nperiod = 2
np.save(filename_output+'.npy',all_coeffs)
plot_all_2D(nloop,nbody,nint_plot,all_coeffs,filename_output+'.png')
# ~ plot_all_2D_anim(nloop,nbody,nint_plot,nperiod,all_coeffs,filename_output+'.mp4')



'''
