import numba
import multiprocessing
import scipy
import numpy as np
import types
import inspect
import tempfile

max_num_threads = multiprocessing.cpu_count()

default_numba_kwargs = {
    'nopython':True     ,
    # 'cache':True        ,
    'fastmath':True     ,
    'nogil':True        ,
}

def jit_inter_law(py_inter_law, numba_kwargs=default_numba_kwargs):
    
    sig = numba.types.void(numba.types.CPointer(numba.types.float64), numba.types.float64, numba.types.CPointer(numba.types.float64))

    cfunc_fun = numba.cfunc(sig, **numba_kwargs)(py_inter_law)
    
    return scipy.LowLevelCallable(cfunc_fun.ctypes)
    
def jit_inter_law_str(py_inter_law_str):
    
    tmpdir = tempfile.gettempdir()
    code_obj = compile(py_inter_law_str, tmpdir, 'exec')
    module_obj = types.ModuleType("modname")
    exec(code_obj, module_obj.__dict__)
    inter_law_py = module_obj.inter_law
    if not inspect.isfunction(inter_law_py):
        raise ValueError('Could not compile provided string to a Python function named "inter_law"')
    
    return jit_inter_law(inter_law_py)
    
# def Compute_forces_vectorized(
#     pos                 , pot_nrg_grad          ,
#     nvec, nsegm, geodim, nbin ,
#     BinSourceSegm       , BinTargetSegm         ,
#     BinSpaceRot         , BinSpaceRotIsId       ,
#     BinProdChargeSum    ,
#     inter_law           , inter_law_param_ptr   ,
# ):
# 
#     dx = np.empty((geodim),dtype=np.float64)
#     df = np.empty((geodim),dtype=np.float64)
#     pot = np.empty((3),dtype=np.float64)
# 
#     for ivec in range(nvec):
# 
#         for ibin in range(nbin):
# 
#             isegm = BinSourceSegm[ibin]
#             isegmp = BinTargetSegm[ibin]
# 
#             if BinSpaceRotIsId[ibin]:
#                     
#                 for idim in range(geodim):
#                     dx[idim] = pos[ivec,isegm,idim] - pos[ivec,isegmp,idim]
# 
#             else:
# 
#                 for idim in range(geodim):
#                     dx[idim] = - pos[ivec,isegmp,idim]
#                     for jdim in range(geodim):
#                         dx[idim] += BinSpaceRot[ibin,idim,jdim] * pos[ivec,isegm,jdim] 
# 
#             dx2 = dx[0]*dx[0]
#             for idim in range(1,geodim):
#                 dx2 += dx[idim]*dx[idim]
# 
#             inter_law(inter_law_param_ptr, dx2, pot)
# 
#             bin_fac = (-4)*BinProdChargeSum[ibin]
# 
#             pot[1] *= bin_fac
# 
#             for idim in range(geodim):
#                 df[idim] = pot[1]*dx[idim]
# 
#             if BinSpaceRotIsId[ibin]:
# 
#                 for idim in range(geodim):
#                     pot_nrg_grad[ivec,isegm,idim] += df[idim]
#                     
#             else:
# 
#                 for jdim in range(geodim):
#                     for idim in range(geodim):
#                         pot_nrg_grad[ivec,isegm,idim] += BinSpaceRot[ibin,jdim,idim] *  df[jdim]
# 
#             for idim in range(geodim):
#                 pot_nrg_grad[ivec,isegmp,idim] -= df[idim]
# 
# 
# def Get_Compute_forces_vectorized(NBS):
#     
#     