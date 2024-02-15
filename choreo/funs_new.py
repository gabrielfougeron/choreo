'''
funs.py : Defines useful functions in the choreo project.
'''

import os
import itertools
import time
import functools
import json

import numpy as np
import math as m
import scipy

import scipy.optimize
import networkx
import math

from matplotlib import pyplot as plt
from matplotlib import colormaps

import choreo.scipy_plus
from choreo.cython._ActionSym import *
from choreo.NBodySyst_build import *
from choreo.cython._NBodySyst import *

def setup_changevar_new(geodim, nbody, nint_init, bodymass, n_reconverge_it_max=6, MomCons=False, n_grad_change=1., Sym_list=[], CrashOnIdentity=True, ForceMatrixChangevar = False, store_folder = ""):
    
    r"""
    This function constructs a ChoreoAction
    It detects loops and constraints based on symmetries.
    It defines parameters according to given constraints and diagonal change of variable.
    It computes useful objects to optimize the computation of the action :
     - Exhaustive list of unary transformation for generator to body.
     - Exhaustive list of binary transformations from generator within each loop.
    """

    
    NBS = NBodySyst(geodim, nbody, bodymass, Sym_list)

    # ###############################################################
    # A partir d'ici on commence à "utiliser" l'objet.
    
    NBS.nint_fac = 4

    
    
    # params_to_all_coeffs_noopt
    
    # Without lists now.


    params_buf = np.random.random((NBS.nparams))

    all_coeffs = NBS.params_to_all_coeffs_noopt(params_buf)        
    all_pos = scipy.fft.irfft(all_coeffs, axis=1)
    segmpos_noopt = NBS.all_pos_to_segmpos_noopt(all_pos)
    
    segmpos_cy = NBS.params_to_segmpos(params_buf)
    
    assert np.linalg.norm(segmpos_noopt - segmpos_cy) < 1e-14
  

#         
#     AssertAllSegmGenConstraintsAreRespected(gensegm_to_all, nint_min, bodysegm, loopgen, gensegm_to_body, gensegm_to_iint , BodyLoop, all_pos)
#     AssertAllBodyConstraintAreRespected(LoopGenConstraints, all_pos)
# 
# 
#     allsegmpos = Populate_allsegmpos(all_pos, GenSpaceRot, GenTimeRev, gensegm_to_body, gensegm_to_iint, BodyLoop, nint_min)
#     
#     all_body_pos = Compute_all_body_pos(all_pos, BodyGraph, loopgen, BodyLoop)
#     
#     
#     assert np.linalg.norm(allsegmpos - segmpos_cy) < eps
#     
#     for isegm in range(nsegm):
# 
#         ib = intersegm_to_body[isegm]
#         iint = intersegm_to_iint[isegm]
#         il = BodyLoop[ib]
# 
#         ibeg = iint * segm_size
#         iend = ibeg + segm_size
#         assert iend <= nint
#         assert np.linalg.norm(allsegmpos[isegm,:,:] - all_body_pos[ib,ibeg:iend,:]) < eps
#         
# 
# 
#     for il in range(nloop):    
#         
#         ib = loopgen[il]
#         nseg_in_loop = np.count_nonzero(gensegm_to_body == ib)
#         
#         nint_loop_min = nseg_in_loop * segm_size
#         
#         npr = (ncoeffs-1) //  ncoeff_min_loop[il]
#         n_inter = npr+1
#         
#         # params_basis_reorganized = np.empty((geodim, nnz_k.shape[0], last_nparam), dtype=np.complex128) 
# 
#         
#         # assert nint_loop_min <= 2*n_inter
#         
#         # 
#         # print()
#         # print(il, nint_loop_min, n_inter)
#         # print(params_basis_reorganized_list[il].shape)
#         # print(params_basis_reorganized_list[il])
#         
#         
    nparam_nosym = geodim * NBS.nint * nbody
    nparam_tot = NBS.nparams

    # All_Id, count_tot, count_unique = CountSegmentBinaryInteractions(NBS.BinarySegm, NBS.nsegm)
    
    print('*****************************************')
    print('')
    # print(f'{Identity_detected = }')
    # print(f'All binary transforms are identity: {All_Id}')
    
    # assert All_Id
    
    # ninter_tot = NBS.nint_min * nbody * (nbody-1)//2
    # ninter_unique = NBS.nsegm * (NBS.nsegm-1)//2

    # print()
    # print(f"total binary interaction count: {ninter_tot}")
    # print(f"unique binary interaction count: {ninter_unique}")
    print(f'{NBS.nsegm = }')


    # print(f"ratio of total to unique binary interactions : {ninter_tot  / ninter_unique}")
    print(f'ratio of integration intervals to segments : {(nbody * NBS.nint_min) / NBS.nsegm}')
    print(f"ratio of parameters before and after constraints: {nparam_nosym / nparam_tot}")

    reduction_ratio = nparam_nosym / nparam_tot

#     assert abs((nparam_nosym / nparam_tot)  - reduction_ratio) < eps
#     
#     if All_Id:
#         assert abs((count_tot / count_unique)  - reduction_ratio) < eps
#         assert abs(((nbody * nint_min) / nsegm) - reduction_ratio) < eps
# 



    return

    dirname = os.path.split(store_folder)[0]
    symname = os.path.split(dirname)[1]
    filename = os.path.join(dirname, f'{symname}_graph_segm.pdf')

    PlotTimeBodyGraph(NBS.SegmGraph, nbody, NBS.nint_min, filename)





















# 
# 
# def Prepare_data_for_speed_comparison(
#     geodim                  ,
#     nbody                   ,
#     mass                    ,
#     n_reconverge_it_max     ,
#     Sym_list                ,
#     nint_fac                ,
# ):
# 
#     return Pick_Named_Args_From_Dict(params_to_all_pos, dict(**locals()))
#         
# TODO: To be removed of course
# def Pick_Named_Args_From_Dict(fun, the_dict, MissingArgsAreNone=True):
#     
#     import inspect
#     list_of_args = inspect.getfullargspec(fun).args
#     
#     if MissingArgsAreNone:
#         all_kwargs = {k:the_dict.get(k) for k in list_of_args}
#         
#     else:
#         all_kwargs = {k:the_dict[k] for k in list_of_args}
#     
#     return all_kwargs
# 
# 
# def params_to_all_pos_mod(params_basis_reorganized_list, all_params_list, nnz_k_list, ncoeff_min_loop, ncoeffs, nnpr):
#     
#     nloop = len(params_basis_reorganized_list)
#     geodim = params_basis_reorganized_list[0].shape[0]
#     all_coeffs = np.zeros((nloop,ncoeffs,geodim), dtype=np.complex128)
#     
# #     for il in range(nloop):
# #         
# #         params_loop = all_params_list[il]
# #         
# #         params_basis_reorganized = params_basis_reorganized_list[il]
# #         geodim = params_basis_reorganized.shape[0]
# #         nnz_k = nnz_k_list[il]
# #         
# #         npr = (ncoeffs-1) //  ncoeff_min_loop[il]
# #         ncoeff_min_loop_nnz = len(nnz_k)
# # 
# #         coeffs_reorganized = np.einsum('ijk,ljk->lji', params_basis_reorganized, params_loop)
# #         
# #         coeffs_dense = np.zeros((npr, ncoeff_min_loop[il], geodim), dtype=np.complex128)
# #         coeffs_dense[:,nnz_k,:] = coeffs_reorganized
# #         all_coeffs[il,:(ncoeffs-1),:] = coeffs_dense.reshape(((ncoeffs-1), geodim))
#         
#     all_pos = scipy.fft.irfft(all_coeffs, axis=1)
#         
# def params_to_all_pos(params_basis_reorganized_list, all_params_list, nnz_k_list, ncoeff_min_loop, ncoeffs, nnpr):
#     
#     nloop = len(params_basis_reorganized_list)
#     geodim = params_basis_reorganized_list[0].shape[0]
#     all_coeffs = np.zeros((nloop,ncoeffs,geodim), dtype=np.complex128)
#     
#     for il in range(nloop):
#         
#         params_loop = all_params_list[il]
#         
#         params_basis_reorganized = params_basis_reorganized_list[il]
#         geodim = params_basis_reorganized.shape[0]
#         nnz_k = nnz_k_list[il]
#         
#         npr = (ncoeffs-1) //  ncoeff_min_loop[il]
#         ncoeff_min_loop_nnz = len(nnz_k)
# 
#         coeffs_reorganized = np.einsum('ijk,ljk->lji', params_basis_reorganized, params_loop)
#         
#         coeffs_dense = np.zeros((npr, ncoeff_min_loop[il], geodim), dtype=np.complex128)
#         coeffs_dense[:,nnz_k,:] = coeffs_reorganized
#         all_coeffs[il,:(ncoeffs-1),:] = coeffs_dense.reshape(((ncoeffs-1), geodim))
#         
#     all_pos = scipy.fft.irfft(all_coeffs, axis=1)
#                 

# def params_to_all_pos_slice(params_basis_reorganized_list, all_params_list, nnz_k_list, ncoeff_min_loop, ncoeffs, nnpr):
#     
#     nloop = len(params_basis_reorganized_list)
#     geodim = params_basis_reorganized_list[0].shape[0]
#     nint = 2*(ncoeffs-1)
#     
#     for il in range(nloop):
#         
#         params_loop = all_params_list[il]
#         params_basis_reorganized = params_basis_reorganized_list[il]
#         nnz_k = nnz_k_list[il]
#         ncoeff_min = ncoeff_min_loop[il]
# 
#         params_to_pos_slice_loop(params_basis_reorganized, params_loop, nnz_k, ncoeff_min, nnpr)
#         
# def params_to_all_pos_slice_nocopy(params_basis_reorganized_list, all_params_list, nnz_k_list, ncoeff_min_loop, ncoeffs, nnpr):
#     
#     nloop = len(params_basis_reorganized_list)
#     geodim = params_basis_reorganized_list[0].shape[0]
#     nint = 2*(ncoeffs-1)
#     
#     for il in range(nloop):
#         
#         params_loop = all_params_list[il]
#         params_basis_reorganized = params_basis_reorganized_list[il]
#         nnz_k = nnz_k_list[il]
#         ncoeff_min = ncoeff_min_loop[il]
# 
#         params_to_pos_slice_loop_nocopy(params_basis_reorganized, params_loop, nnz_k, ncoeff_min, nnpr)