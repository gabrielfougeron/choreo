import os
import concurrent.futures
import multiprocessing

import scipy
import numpy as np
import sys

__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(__PROJECT_ROOT__)

import choreo 

store_folder = os.path.join(__PROJECT_ROOT__,'Sniff_all_sym','Simo_tests')
NT_init_filename = os.path.join(__PROJECT_ROOT__,'NumericalTank_data','Simo_init_cond.txt')
all_NT_init = np.loadtxt(NT_init_filename)

import pyquickbench

def Integrate(n_NT_init):
    
    print('')
    print('#####################################')
    print(f'    Simo {n_NT_init}')
    print('#####################################')
    print('')

    geodim = 2
    nbody = 3

    bodymass = np.ones(nbody)
    bodycharge = bodymass.copy()
    Sym_list = []


    BodyPerm = np.array([1,2,0], dtype=np.intp)
    SpaceRot = np.identity(geodim, dtype=np.float64)

    Sym_choreo = choreo.ActionSym(
        BodyPerm.copy() ,
        SpaceRot.copy() ,
        1               ,
        1               ,
        nbody           ,
    )
    
    BodyPerm = np.array([0,2,1], dtype=np.intp)
    SpaceRot = np.array([[1,0],[0,-1]], dtype=np.float64)
    
    Sym_reflexion = choreo.ActionSym(
        BodyPerm.copy() ,
        SpaceRot.copy() ,
        -1              ,
        0               ,
        1               ,
    )
    
    Sym_list = [Sym_choreo, Sym_reflexion]

    NBS = choreo.NBodySyst(geodim, nbody, bodymass, bodycharge, Sym_list)

    NBS.ForceGreaterNStore = True

    print(NBS.DescribeSystem())

    file_basename = 'Simo_'+(str(n_NT_init).zfill(5))
    save_filename = os.path.join(store_folder, file_basename)

    ndof = NBS.geodim * NBS.nsegm

    xo = np.zeros(ndof)
    vo = np.zeros(ndof)

    # Initial positions: (-2 c1, 0), (c1, c2), (c1, -c2) * 6
    # Initial velocities: (0, -2 c4), (c3, c4), (-c3, c4)
    
    phys_exp = 1/(NBS.Homo_exp-1)
      
    a = 6

    xo[0] = -2 * all_NT_init[n_NT_init,0] * a
    xo[1] = 0
    xo[2] =   all_NT_init[n_NT_init,0]    * a
    xo[3] =   all_NT_init[n_NT_init,1]    * a
    xo[4] =   all_NT_init[n_NT_init,0]    * a
    xo[5] = - all_NT_init[n_NT_init,1]    * a

    vo[0] = 0
    vo[1] = -2 * all_NT_init[n_NT_init,3]
    vo[2] =   all_NT_init[n_NT_init,2]   
    vo[3] =   all_NT_init[n_NT_init,3]   
    vo[4] = - all_NT_init[n_NT_init,2]   
    vo[5] =   all_NT_init[n_NT_init,3]   

    T_NT = all_NT_init[n_NT_init,4] * 18

    rfac = (T_NT) ** phys_exp
    
    # print(f'{phys_exp = }')
    # print(f'{rfac = }')

    xo *= rfac
    vo *= rfac * T_NT 
    
    xo_ini = xo.copy()
    vo_ini = vo.copy()
    
    ODEparams_ini = NBS.initposmom_to_ODE_params(xo, vo)

    ODEparams_ini += np.random.random(NBS.n_ODEinitparams) * 1e0
                                      
    ODE_Syst = NBS.Get_ODE_def()
    
    # NBS.nint_fac = 32
    NBS.nint_fac = 256

    nsteps = 10
    keep_freq = 1
    method = "Gauss"
    
    rk = choreo.segm.multiprec_tables.ComputeImplicitRKTable(nsteps, method=method)
    
    def Periodicity_default(ODE_params):
        
        xo, vo = NBS.ODE_params_to_initposmom(ODE_params)
        
        nint_ODE = (NBS.segm_store-1) * keep_freq
        
        xf, vf = choreo.segm.ODE.ImplicitSymplecticIVP(
            xo = xo                 ,
            vo = vo                 ,
            rk_x = rk               ,
            rk_v = rk               ,
            nint = nint_ODE         ,
            keep_init = False       ,
            **ODE_Syst              ,
        )
        
        xf = xf.reshape(-1)
        vf = vf.reshape(-1)

        return NBS.endposmom_to_perdef(xo, vo, xf, vf)
    
    
    perdef = Periodicity_default(ODEparams_ini)
    
    print(np.linalg.norm(perdef))
    
    res = scipy.optimize.root(Periodicity_default, ODEparams_ini, method="krylov", tol = 1e-12, options={"disp":True})
    
    ODEparams_opt = res['x']
    
    print(np.linalg.norm(ODEparams_opt - ODEparams_ini))
    
    
    xo, vo = NBS.ODE_params_to_initposmom(ODEparams_opt)
    
    nint_ODE = (NBS.segm_store-1) * keep_freq
    
    segmpos_ODE, segmmom_ODE = choreo.segm.ODE.ImplicitSymplecticIVP(
        xo = xo                 ,
        vo = vo                 ,
        rk_x = rk               ,
        rk_v = rk               ,
        keep_init = True        ,
        keep_freq = keep_freq   ,
        nint = nint_ODE         ,
        **ODE_Syst              ,
    )
    
    segmpos_ODE = np.ascontiguousarray(segmpos_ODE.reshape((NBS.segm_store, NBS.nsegm, NBS.geodim)).swapaxes(0, 1))
    # segmmom_ODE = np.ascontiguousarray(segmmom_ODE.reshape((NBS.segm_store, NBS.nsegm, NBS.geodim)).swapaxes(0, 1))
    
    NBS.plot_segmpos_2D(segmpos_ODE, "test.png")
    
    


# the_NT_init = range(len(all_NT_init))
# the_NT_init = range(5)

the_NT_init = [0]
# the_NT_init.extend(range(25,len(all_NT_init)))

# # 
# 
if __name__ == "__main__":

    TT = pyquickbench.TimeTrain()

    for n_NT_init in the_NT_init:
        
        Integrate(n_NT_init)
        TT.toc(n_NT_init)
        
    # print(TT)



#     # n = multiprocessing.cpu_count() // 2
#     n = 8
# 
#     print(f"Executing with {n} workers")
# 
#     with concurrent.futures.ProcessPoolExecutor(max_workers=n) as executor:
#         
#         res = []
#         for n_NT_init in range(len(all_NT_init)):
#             res.append(executor.submit(Integrate,n_NT_init))
# 

