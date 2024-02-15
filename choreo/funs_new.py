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

    
    nbodysyst = NBodySyst(geodim, nbody, bodymass, Sym_list)

    assert bodymass.shape[0] == nbody

    eps = 1e-12

    nint_min, nloop, loopnb, loopmass, BodyLoop, Targets, BodyGraph = DetectLoops(Sym_list, nbody, bodymass)

    
    SegmGraph, nint_min, nsegm, bodysegm, BodyHasContiguousGeneratingSegments, Sym_list = ExploreGlobalShifts_BuildSegmGraph(geodim, nbody, nloop, loopnb, Targets, nint_min, Sym_list)

    loopgen = ChooseLoopGen(nloop, loopnb, BodyHasContiguousGeneratingSegments, Targets)




    # Accumulate constraints on segments. 

    SegmConstraints = AccumulateSegmentConstraints(SegmGraph, nbody, geodim, nsegm, bodysegm)
    
#     print()
#     print("Segment Constraints")
#     for isegm in range(nsegm):
#         ncstr = len(SegmConstraints[isegm])
#         print(f'{isegm = } {ncstr = }')
# 
#         for icstr, Constraint in enumerate(SegmConstraints[isegm]):
#             print(f'{icstr = }')
#             print(Constraint)
#             print()
#     print()

    # ComputeParamBasis_Segm(nbody, geodim, SegmConstraints)










    # Choose interacting segments as earliest possible times.
    intersegm_to_body, intersegm_to_iint = ChooseInterSegm(nsegm, nint_min, nbody, bodysegm)

    # Choose generating segments as contiguous chunks of loop generators
    gensegm_to_body, gensegm_to_iint, ngensegm_loop = ChooseGenSegm(nsegm, nint_min, nloop, loopgen, bodysegm)
                


    # Setting up forward ODE:
    # - What are my parameters ?
    # - Integration end + Lack of periodicity
    # - Constraints on initial values => Parametrization 

    
    InstConstraintsPos = AccumulateInstConstraints(Sym_list, nbody, geodim, nint_min, VelSym=False)
    InstConstraintsVel = AccumulateInstConstraints(Sym_list, nbody, geodim, nint_min, VelSym=True )

    # One segment is enough (checked before)
    
    # print("Initial time Position constraints")
    # ncstr = len(InstConstraintsPos[0])
    # print(f'{ncstr = }')
    # for icstr, Constraint in enumerate(InstConstraintsPos[0]):
    #     print(f'{icstr = }')
    #     print(Constraint)
    #     print()    
    # print()    
    # print("Initial time Velocity constraints")
    # ncstr = len(InstConstraintsVel[0])
    # print(f'{ncstr = }')
    # for icstr, Constraint in enumerate(InstConstraintsVel[0]):
    #     print(f'{icstr = }')
    #     print(Constraint)
    #     print()
    # print()

    # MomCons_InitVal = False
    MomCons_InitVal = True

    InitValPosBasis = ComputeParamBasis_InitVal(nbody, geodim, InstConstraintsPos[0], bodymass, MomCons=MomCons_InitVal)
    InitValVelBasis = ComputeParamBasis_InitVal(nbody, geodim, InstConstraintsVel[0], bodymass, MomCons=MomCons_InitVal)
    
    # print("Initial Position parameters")
    # print("nparam = ",InitValPosBasis.shape[2])
    # print()
    # for iparam in range(InitValPosBasis.shape[2]):
    #     
    #     print(f'{iparam = }')
    #     print(InitValPosBasis[:,:,iparam])    
    #     print()
    #     
    #     
    # print("Initial Velocity parameters")
    # print("nparam = ",InitValVelBasis.shape[2])
    # print()
    # for iparam in range(InitValVelBasis.shape[2]):
    #     
    #     print(f'{iparam = }')
    #     print(InitValVelBasis[:,:,iparam])
    #     print()
    
    
    
    gensegm_to_all = AccumulateSegmGenToTargetSym(SegmGraph, nbody, geodim, nint_min, nsegm, bodysegm, gensegm_to_iint, gensegm_to_body)
    GenTimeRev, GenSpaceRot = GatherGenSym(nsegm, geodim, intersegm_to_body, intersegm_to_iint, gensegm_to_all)
    
    GenToIntSyms = Generating_to_interacting(SegmGraph, nbody, geodim, nsegm, intersegm_to_iint, intersegm_to_body, gensegm_to_iint, gensegm_to_body)

    intersegm_to_all = AccumulateSegmGenToTargetSym(SegmGraph, nbody, geodim, nint_min, nsegm, bodysegm, intersegm_to_iint, intersegm_to_body)

    BinarySegm, Identity_detected = FindAllBinarySegments(intersegm_to_all, nbody, nsegm, nint_min, bodysegm, CrashOnIdentity, bodymass)

    All_Id, count_tot, count_unique = CountSegmentBinaryInteractions(BinarySegm, nsegm)


    




    
    # print()
    # print('================================================')
    # print()

    # This could certainly be made more efficient
    BodyConstraints = AccumulateBodyConstraints(Sym_list, nbody, geodim)
    LoopGenConstraints = [BodyConstraints[ib] for ib in loopgen]
    
    
    # for isegm in range(nsegm):
    #     
    #     print()
    #     print(isegm)
    #     print(gensegm_to_iint[isegm], gensegm_to_body[isegm])
    #     print(intersegm_to_iint[isegm], intersegm_to_body[isegm])
    #     print(GenToIntSyms[isegm])
    #     print(gensegm_to_all[intersegm_to_body[isegm]][intersegm_to_iint[isegm]])
    #     
    #     
    #     assert GenToIntSyms[isegm].IsSameRot(gensegm_to_all[intersegm_to_body[isegm]][intersegm_to_iint[isegm]])
    #     assert GenToIntSyms[isegm].IsSameTimeRev(gensegm_to_all[intersegm_to_body[isegm]][intersegm_to_iint[isegm]])
    
    
    
    
    

    All_params_basis = ComputeParamBasis_Loop(nbody, nloop, loopgen, geodim, LoopGenConstraints)
    params_basis_reorganized_list, nnz_k_list = reorganize_All_params_basis(All_params_basis)
    
    params_basis_buf, params_basis_shapes, params_basis_shifts = BundleListOfArrays(params_basis_reorganized_list)
    nnz_k_buf, nnz_k_shapes, nnz_k_shifts = BundleListOfArrays(nnz_k_list)

    ncoeff_min_loop = np.array([len(All_params_basis[il]) for il in range(nloop)], dtype=np.intp)
    ncoeff_min_loop_nnz = np.array([nnz_k_list[il].shape[0] for il in range(nloop)], dtype=np.intp)
    

    nnpr = Compute_nnpr(nloop, nint_min, ncoeff_min_loop, ngensegm_loop)

    print( f'{nnpr = }')
    

    
    
    
    # ###############################################################
    # A partir d'ici on commence à "utiliser" l'objet.
    
    nbodysyst.nint = 2 * nint_min * 4

    
    
    # Without lists now.

    
        
        
    params_buf = np.random.random((nbodysyst.nparams))
    
    return
            
    segmpos_cy = params_to_segmpos(
        params_buf.copy()   , params_shapes         , params_shifts         ,
                              ifft_shapes           , ifft_shifts           ,
        params_basis_buf    , params_basis_shapes   , params_basis_shifts   ,
        nnz_k_buf           , nnz_k_shapes          , nnz_k_shifts          ,
                              pos_slice_shapes      , pos_slice_shifts      ,
        ncoeff_min_loop     , nnpr  ,
        GenSpaceRot         ,
        GenTimeRev          ,
        gensegm_to_body     ,
        gensegm_to_iint     ,
        BodyLoop            ,
        segm_size           ,
    )
            

    all_coeffs = params_to_all_coeffs(
        params_buf, params_shapes, params_shifts,
        params_basis_reorganized_list, nnz_k_list, ncoeff_min_loop, nint
    )
        
    all_pos = scipy.fft.irfft(all_coeffs, axis=1)
 
        
    AssertAllSegmGenConstraintsAreRespected(gensegm_to_all, nint_min, bodysegm, loopgen, gensegm_to_body, gensegm_to_iint , BodyLoop, all_pos)
    AssertAllBodyConstraintAreRespected(LoopGenConstraints, all_pos)


    allsegmpos = Populate_allsegmpos(all_pos, GenSpaceRot, GenTimeRev, gensegm_to_body, gensegm_to_iint, BodyLoop, nint_min)
    
    all_body_pos = Compute_all_body_pos(all_pos, BodyGraph, loopgen, BodyLoop)
    
    
    assert np.linalg.norm(allsegmpos - segmpos_cy) < eps
    
    for isegm in range(nsegm):

        ib = intersegm_to_body[isegm]
        iint = intersegm_to_iint[isegm]
        il = BodyLoop[ib]

        ibeg = iint * segm_size
        iend = ibeg + segm_size
        assert iend <= nint
        assert np.linalg.norm(allsegmpos[isegm,:,:] - all_body_pos[ib,ibeg:iend,:]) < eps
        


    for il in range(nloop):    
        
        ib = loopgen[il]
        nseg_in_loop = np.count_nonzero(gensegm_to_body == ib)
        
        nint_loop_min = nseg_in_loop * segm_size
        
        npr = (ncoeffs-1) //  ncoeff_min_loop[il]
        n_inter = npr+1
        
        # params_basis_reorganized = np.empty((geodim, nnz_k.shape[0], last_nparam), dtype=np.complex128) 

        
        # assert nint_loop_min <= 2*n_inter
        
        # 
        # print()
        # print(il, nint_loop_min, n_inter)
        # print(params_basis_reorganized_list[il].shape)
        # print(params_basis_reorganized_list[il])
        
        
    nparam_nosym = geodim * nint * nbody
    nparam_tot = params_shifts[-1]

    
    print('*****************************************')
    print('')
    print(f'{Identity_detected = }')
    print(f'All binary transforms are identity: {All_Id}')
    
    # assert All_Id

    print()
    print(f"total binary interaction count: {count_tot}")
    print(f"total expected binary interaction count: {nint_min * nbody * (nbody-1)//2}")
    print(f"unique binary interaction count: {count_unique}")
    print(f'{nsegm = }')


    print()
    print(f"ratio of total to unique binary interactions : {count_tot / count_unique}")
    print(f'ratio of integration intervals to segments : {(nbody * nint_min) / nsegm}')
    print(f"ratio of parameters before and after constraints: {nparam_nosym / nparam_tot}")

    reduction_ratio = nparam_nosym / nparam_tot

    assert abs((nparam_nosym / nparam_tot)  - reduction_ratio) < eps
    
    if All_Id:
        assert abs((count_tot / count_unique)  - reduction_ratio) < eps
        assert abs(((nbody * nint_min) / nsegm) - reduction_ratio) < eps




    return

    dirname = os.path.split(store_folder)[0]
    symname = os.path.split(dirname)[1]
    filename = os.path.join(dirname, f'{symname}_graph_segm.pdf')

    PlotTimeBodyGraph(SegmGraph, nbody, nint_min, filename)





















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