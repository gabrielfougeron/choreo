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
import sparseqr
import networkx
import random
import math

from matplotlib import pyplot as plt
from matplotlib import colormaps

import choreo.scipy_plus

from choreo.cython.funs_new import ActionSym


def EdgesAreEitherDirectXORIndirect(FullGraph):

    AlwaysContainsExactlyOne = True

    for iedge, edge in enumerate(FullGraph.edges):

        ContainsDirect = False
        ContainsIndirect = False

        for Sym in FullGraph.edges[edge]["SymList"]:

            if Sym.TimeRev == 1:
                ContainsDirect = True
            else:
                ContainsIndirect = True

        AlwaysContainsExactlyOne =  AlwaysContainsExactlyOne and (ContainsDirect ^ ContainsIndirect)  # XOR

    return AlwaysContainsExactlyOne

def ContainsDoubleEdges(FullGraph):

    for edge in FullGraph.edges:

        if (len(FullGraph.edges[edge]["SymList"]) > 1):
            return True

    return False

def ContainsSelfReferingTimeRevSegment(FullGraph, nbody, nint):

    CCs = networkx.connected_components(FullGraph)

    for CC in CCs:

        for segm, segmp in itertools.combinations(CC,2):

            if (segm[1] == segmp[1]):

                path = networkx.shortest_path(FullGraph, source = segm, target = segmp)

                TimeRev = 1
                pathlen = len(path)

                for ipath in range(1,pathlen):

                    if (path[ipath-1] > path[ipath]):
                        edge = (path[ipath], path[ipath-1])
                    else:
                        edge = (path[ipath-1], path[ipath])

                    TimeRev *=  FullGraph.edges[edge]["SymList"][0].TimeRev

                if (TimeRev == -1):
                    
                    return True
                
    return False

def Build_FullGraph(nbody, nint, Sym_list):

    FullGraph = networkx.Graph()
    for ib in range(nbody):
        for iint in range(nint):
            FullGraph.add_node((ib,iint))

    for Sym in Sym_list:

        SymInv = Sym.Inverse()

        for ib in range(nbody):

            ib_target = Sym.BodyPerm[ib]

            for iint in range(nint):
                
                if (Sym.TimeRev == 1): 
                    tnum = iint
                
                else:
                    tnum = ((iint+1)%nint)

                tnum_target, tden_target = Sym.ApplyT(tnum, nint)

                # print(tnum,nint)
                # print(tnum_target, tden_target)

                assert nint % tden_target == 0

                iint_target = (tnum_target * (nint // tden_target) + nint) % nint

                # node_source = (ib       , iint       )
                # node_target = (ib_target, iint_target)

                node_source = (ib       , iint_target)
                node_target = (ib_target, iint       )

                if node_source <= node_target :

                    edge = (node_source, node_target)
                    EdgeSym = Sym

                else:

                    edge = (node_target, node_source)
                    EdgeSym = SymInv

                if edge in FullGraph.edges:
                    
                    AlreadyIn = False
                    for OtherEdgeSym in FullGraph.edges[edge]["SymList"]:
                        AlreadyIn = AlreadyIn or EdgeSym.IsSameRotAndTimeRev(OtherEdgeSym)

                    if not(AlreadyIn) :
                        FullGraph.edges[edge]["SymList"].append(EdgeSym)

                else:
                    FullGraph.add_edge(*edge, SymList = [EdgeSym])
    
    return FullGraph

def Build_FullGraph_NoPb(
    nbody,
    nint,
    Sym_list,
    current_recursion = 1,
    max_recursion = 5,
):
    # print(f'{current_recursion=}')
    
    if (current_recursion > max_recursion):
        raise ValueError("Achieved max recursion level in Build_FullGraph")

    FullGraph = Build_FullGraph(nbody, nint, Sym_list)

    if ContainsDoubleEdges(FullGraph):

        return Build_FullGraph_NoPb(
            nbody = nbody,
            nint = 2*nint,
            Sym_list = Sym_list,
            current_recursion = current_recursion+1,
            max_recursion = max_recursion,
        )

    if ContainsSelfReferingTimeRevSegment(FullGraph, nbody, nint):

        return Build_FullGraph_NoPb(
            nbody = nbody,
            nint = 2*nint,
            Sym_list = Sym_list,
            current_recursion = current_recursion+1,
            max_recursion = max_recursion,
        ) 

    return FullGraph, nint

def Build_BodyGraph(nbody, Sym_list):

    BodyGraph = networkx.Graph()
    for ib in range(nbody):
        BodyGraph.add_node(ib)

    for Sym in Sym_list:

        SymInv = Sym.Inverse()

        for ib in range(nbody):

            ib_target = Sym.BodyPerm[ib]

            if ib > ib_target:
                edge = (ib_target, ib)
                EdgeSym = SymInv
            else:
                edge = (ib, ib_target)
                EdgeSym = Sym

            if (edge in BodyGraph.edges):

                for FoundSym in BodyGraph.edges[edge]["SymList"]:

                    AlreadyFound = EdgeSym.IsSameRotAndTime(FoundSym)
                    if AlreadyFound:
                        break

                if not(AlreadyFound):
                    BodyGraph.edges[edge]["SymList"].append(EdgeSym)

            else:

                BodyGraph.add_edge(*edge, SymList = [EdgeSym])

    return BodyGraph

def AppendIfNotSameRotAndTime(CstrList, Constraint):

    if not(Constraint.IsIdentityRotAndTime()):
        
        AlreadyFound = False
            
        for FoundCstr in CstrList:

            AlreadyFound = Constraint.IsSameRotAndTime(FoundCstr)
            if AlreadyFound:
                break

            AlreadyFound = Constraint.IsSameRotAndTime(FoundCstr.Inverse())
            if AlreadyFound:
                break

        if not(AlreadyFound):
            
            CstrList.append(Constraint)

def AccumulateBodyConstraints(Sym_list, nbody, geodim):

    BodyConstraints = [[] for _ in range(nbody)]

    SimpleBodyGraph = networkx.Graph()
    for ib in range(nbody):
        SimpleBodyGraph.add_node(ib)

    for Sym in Sym_list:

        SymInv = Sym.Inverse()

        for ib in range(nbody):

            ib_target = Sym.BodyPerm[ib]

            if ib == ib_target:

                AppendIfNotSameRotAndTime(BodyConstraints[ib], Sym)                
            
            else:
                    
                if ib > ib_target:
                    edge = (ib_target, ib)
                    EdgeSym = SymInv
                else:
                    edge = (ib, ib_target)
                    EdgeSym = Sym

                if (edge in SimpleBodyGraph.edges):

                    Constraint = EdgeSym.Inverse().Compose(SimpleBodyGraph.edges[edge]["Sym"])
                    assert Constraint.BodyPerm[edge[0]] == edge[0]
                    AppendIfNotSameRotAndTime(BodyConstraints[edge[0]], Constraint)

                    Constraint = EdgeSym.Compose(SimpleBodyGraph.edges[edge]["Sym"].Inverse())
                    assert Constraint.BodyPerm[edge[1]] == edge[1]
                    AppendIfNotSameRotAndTime(BodyConstraints[edge[1]], Constraint)

                else:

                    SimpleBodyGraph.add_edge(*edge, Sym = EdgeSym)

    Cycles = networkx.cycle_basis(SimpleBodyGraph)

    for Cycle in itertools.chain(SimpleBodyGraph.edges, Cycles):

        Cycle_len = len(Cycle)

        FirstBodyConstraint = ActionSym.Identity(nbody, geodim)
        for iedge in range(Cycle_len):
            
            ibeg = Cycle[iedge]
            iend = Cycle[(iedge+1)%Cycle_len]

            if (ibeg <= iend):
                Sym = SimpleBodyGraph.edges[(ibeg,iend)]["Sym"]

            else:
                Sym = SimpleBodyGraph.edges[(ibeg,iend)]["Sym"].Inverse()
                
            FirstBodyConstraint = Sym.Compose(FirstBodyConstraint)

        FirstBody = Cycle[0]
        assert FirstBodyConstraint.BodyPerm[FirstBody] == FirstBody

        path_to_FirstBody = networkx.shortest_path(SimpleBodyGraph, target = FirstBody)

        # Now add the Cycle constraints to every body in the cycle
        for ib in Cycle:

            ibToFirstBodySym =  ActionSym.Identity(nbody, geodim)

            path = path_to_FirstBody[ib]            
            pathlen = len(path)

            for ipath in range(1,pathlen):

                if (path[ipath-1] > path[ipath]):

                    edge = (path[ipath], path[ipath-1])
                    Sym = SimpleBodyGraph.edges[edge]["Sym"].Inverse()

                else:

                    edge = (path[ipath-1], path[ipath])
                    Sym = SimpleBodyGraph.edges[edge]["Sym"]

                ibToFirstBodySym = Sym.Compose(ibToFirstBodySym)

            Constraint = ibToFirstBodySym.Inverse().Compose(FirstBodyConstraint.Compose(ibToFirstBodySym))

            assert Constraint.BodyPerm[ib] == ib

            AppendIfNotSameRotAndTime(BodyConstraints[ib], Constraint)

    return BodyConstraints

def AccumulateSegmentConstraints(FullGraph, nbody, geodim, nsegm, bodysegm):
    # Accumulate constraints on segments. 
    # TODO : prove that it is actually useless ?

    SegmConstraints = [ list() for isegm in range(nsegm)]

    Cycles = networkx.cycle_basis(FullGraph)

    for Cycle in Cycles:

        isegm = bodysegm[*Cycle[0]]
        Cycle_len = len(Cycle)
        
        Constraint = ActionSym.Identity(nbody, geodim)

        for iedge in range(Cycle_len):
            
            ibeg = Cycle[iedge]
            iend = Cycle[(iedge+1)%Cycle_len]

            if (ibeg <= iend):
                Constraint = FullGraph.edges[(ibeg,iend)]["SymList"][0].Compose(Constraint)
                
            else:
                Constraint = FullGraph.edges[(iend,ibeg)]["SymList"][0].Inverse().Compose(Constraint)

        if not(Constraint.IsIdentityRotAndTimeRev()):

            AlreadyFound = False
            for FoundCstr in SegmConstraints[isegm]:
                
                AlreadyFound = Constraint.IsSameRotAndTimeRev(FoundCstr)
                if AlreadyFound:
                    break

                ConstraintInv = Constraint.Inverse()
                AlreadyFound = ConstraintInv.IsSameRotAndTimeRev(FoundCstr)
                if AlreadyFound:
                    break

            if not(AlreadyFound):
                SegmConstraints[isegm].append(Constraint)

    return SegmConstraints

def AccumulateSegmGenToTargetSym(FullGraph, nbody, geodim, nloop, nint_min, nsegm, bodysegm, segmbody, loopnb, Targets, segm_to_iint, segm_to_body):

    segm_gen_to_target = [ [ None for iint in range(nint_min)] for ib in range(nbody) ]

    for isegm, ib_iint_list  in enumerate(segmbody):

        segmgen = (segm_to_body[isegm], segm_to_iint[isegm])
        isegmgen = bodysegm[*segmgen] 

        path_from_segmgen = networkx.shortest_path(FullGraph, source = segmgen)

        for ib, iint in ib_iint_list:

            segm = (ib, iint)

            assert isegm == isegmgen

            GenToTargetSym = ActionSym.Identity(nbody, geodim)

            path = path_from_segmgen[segm]
            pathlen = len(path)

            for ipath in range(1,pathlen):

                if (path[ipath-1] > path[ipath]):

                    edge = (path[ipath], path[ipath-1])
                    Sym = FullGraph.edges[edge]["SymList"][0].Inverse()

                else:

                    edge = (path[ipath-1], path[ipath])
                    Sym = FullGraph.edges[edge]["SymList"][0]

                GenToTargetSym = Sym.Compose(GenToTargetSym)

            segm_gen_to_target[ib][iint] = GenToTargetSym

    return segm_gen_to_target

def FindAllBinarySegments(segm_gen_to_target, nbody, nsegm, nint_min, bodysegm, CrashOnIdentity, mass, BodyLoop):

    Identity_detected = False

    BinarySegm = {}

    for isegm in range(nsegm):
        for isegmp in range(isegm,nsegm):
            BinarySegm[(isegm, isegmp)] = {
                "SymList" : [],
                "SymCount" : [],
                "ProdMassSum" : [],
            }

    for iint in range(nint_min):

        for ib in range(nbody-1):
            
            segm = (ib, iint)
            isegm = bodysegm[*segm]

            for ibp in range(ib+1,nbody):
                
                segmp = (ibp, iint)
                isegmp = bodysegm[*segmp] 

                if (isegm <= isegmp):

                    bisegm = (isegm, isegmp)
                    Sym = (segm_gen_to_target[ibp][iint]).Compose(segm_gen_to_target[ib][iint].Inverse())

                else:

                    bisegm = (isegmp, isegm)
                    Sym = (segm_gen_to_target[ib][iint]).Compose(segm_gen_to_target[ibp][iint].Inverse())

                if ((isegm == isegmp) and Sym.IsIdentityRotAndTimeRev()):

                    if CrashOnIdentity:
                        raise ValueError("Two bodies have identical trajectories")
                    else:
                        if not(Identity_detected):
                            print("Two bodies have identical trajectories")
                        
                    Identity_detected = True

                AlreadyFound = False
                for isym, FoundSym in enumerate(BinarySegm[bisegm]["SymList"]):
                    
                    AlreadyFound = Sym.IsSameRotAndTimeRev(FoundSym)
                    if AlreadyFound:
                        break

                    if (isegm == isegmp):
                        
                        SymInv = Sym.Inverse()
                        AlreadyFound = SymInv.IsSameRotAndTimeRev(FoundSym)
                        if AlreadyFound:
                            break

                if AlreadyFound:
                    BinarySegm[bisegm]["SymCount"][isym] += 1
                    BinarySegm[bisegm]["ProdMassSum"][isym] += mass[BodyLoop[ib]]*mass[BodyLoop[ibp]]

                else:
                    BinarySegm[bisegm]["SymList"].append(Sym)
                    BinarySegm[bisegm]["SymCount"].append(1)
                    BinarySegm[bisegm]["ProdMassSum"].append(mass[BodyLoop[ib]]*mass[BodyLoop[ibp]])

    return BinarySegm, Identity_detected

def ComputeParamBasis_Body(MomCons, nbody, geodim, BodyConstraints):

    eps = 1e-12

    if (MomCons):
        raise NotImplementedError("Momentum conservation as a constraint is not available at the moment")
    else:

        All_params_basis = []
        
        all_ncoeffs_min = []

        for ib in range(nbody):

            all_time_dens = []

            for icstr, Sym in enumerate(BodyConstraints[ib]):
                assert Sym.BodyPerm[ib] == ib
                all_time_dens.append(Sym.TimeShiftDen)

            ncoeffs_min =  math.lcm(*all_time_dens)

            all_ncoeffs_min.append(ncoeffs_min)

            ncstr = len(BodyConstraints[ib])

            NullSpace_all = []

            for k in range(ncoeffs_min):
            
                cstr_mat = np.zeros((ncstr, geodim, 2, geodim, 2), dtype = np.float64)

                for icstr, Sym in enumerate(BodyConstraints[ib]):

                    alpha = - (2 * math.pi * k * Sym.TimeShiftNum) / Sym.TimeShiftDen
                    c = math.cos(alpha)
                    s = math.sin(alpha)
                    
                    for idim in range(geodim):
                        for jdim in range(geodim):

                            cstr_mat[icstr, idim, 0, jdim, 0] =   Sym.SpaceRot[idim, jdim] * c
                            cstr_mat[icstr, idim, 0, jdim, 1] = - Sym.SpaceRot[idim, jdim] * s * Sym.TimeRev
                            cstr_mat[icstr, idim, 1, jdim, 0] =   Sym.SpaceRot[idim, jdim] * s 
                            cstr_mat[icstr, idim, 1, jdim, 1] =   Sym.SpaceRot[idim, jdim] * c * Sym.TimeRev
                            
                        cstr_mat[icstr, idim, 0, idim, 0] -= 1
                        cstr_mat[icstr, idim, 1, idim, 1] -= 1

                # Projection towards 0 will increase sparsity of NullSpace
                for icstr in range(ncstr):
                    for idim in range(geodim):
                        for ift in range(2):
                            for jdim in range(geodim):
                                for jft in range(2):

                                    if abs(cstr_mat[icstr, idim, ift, jdim, jft]) < eps :
                                        cstr_mat[icstr, idim, ift, jdim, jft] = 0


                cstr_mat_reshape = cstr_mat.reshape((ncstr*geodim*2, geodim*2))

                NullSpace = choreo.scipy_plus.linalg.null_space(cstr_mat_reshape)
                nparam = NullSpace.shape[1]
                NullSpace = NullSpace.reshape(geodim,2,nparam)

                for idim in range(geodim):
                    for ift in range(2):
                        for iparam in range(nparam):

                            if abs(NullSpace[idim, ift, iparam]) < eps:
                                NullSpace[idim, ift, iparam] = 0

                NullSpace_all.append(NullSpace)

            All_params_basis.append(NullSpace_all)

    return All_params_basis

def reorganize_All_params_basis(All_params_basis):
    
    nbody = len(All_params_basis)
    geodim = All_params_basis[0][0].shape[0]
    
    all_nnz_k = []
    all_params_basis_reoganized = []
    
    for ib in range(nbody):
        
        params_basis = All_params_basis[ib]
        
        ncoeffs_min = len(params_basis)
        
        last_nparam = None
        nnz_k = []
        for k in range(ncoeffs_min):
            
            nparam_now = params_basis[k].shape[2]
            
            if last_nparam is None and nparam_now != 0:
                last_nparam = nparam_now
            elif nparam_now != 0:
                assert nparam_now == last_nparam
            
            if nparam_now != 0:
                nnz_k.append(k)
                
        nnz_k = np.array(nnz_k)
        all_nnz_k.append(nnz_k)
            
        params_basis_reoganized = np.empty((geodim, nnz_k.shape[0], last_nparam), dtype=np.complex128)    
        
        for ik, k in enumerate(nnz_k):
            
            params_basis_reoganized[:,ik,:] = params_basis[k][:,0,:] + 1j*params_basis[k][:,1,:]
        
        all_params_basis_reoganized.append(params_basis_reoganized)

    return all_params_basis_reoganized, all_nnz_k

def setup_changevar_new(geodim,nbody,nint_init,mass,n_reconverge_it_max=6,MomCons=False,n_grad_change=1.,Sym_list=[],CrashOnIdentity=True,ForceMatrixChangevar = False):
    
    r"""
    This function constructs a ChoreoAction
    It detects loops and constraints based on symmetries.
    It defines parameters according to given constraints and diagonal change of variable.
    It computes useful objects to optimize the computation of the action :
     - Exhaustive list of unary transformation for generator to body.
     - Exhaustive list of binary transformations from generator within each loop.
    """

    GlobalTimeShift = ActionSym(
        BodyPerm  = np.array(range(nbody), dtype = np.int_),
        SpaceRot  = np.identity(geodim, dtype = np.float64),
        TimeRev   = 1,
        TimeShiftNum = 1    ,
        TimeShiftDen = 12,
    )
    
    Shifted_sym_list = []

    for Sym in Sym_list:
        Shifted_sym_list.append( Sym.Conjugate(GlobalTimeShift))
        
    Sym_list = Shifted_sym_list
        
    
    

    All_den_list_on_entry = []
    for Sym in Sym_list:
        All_den_list_on_entry.append(Sym.TimeShiftDen)

    nint_min = math.lcm(*All_den_list_on_entry) # ensures that all integer divisions will have zero remainder
    
    
    
    
    

    FullGraph, nint_min = Build_FullGraph_NoPb(nbody, nint_min, Sym_list)

    BodyGraph =  Build_BodyGraph(nbody,Sym_list)

    nloop = sum(1 for _ in networkx.connected_components(BodyGraph))
    
    loopnb = np.zeros((nloop), dtype = int)

    for il, CC in enumerate(networkx.connected_components(BodyGraph)):
        loopnb[il] = len(CC)

    maxlooplen = loopnb.max()

    bodysegm = np.zeros((nbody, nint_min), dtype = int)
    for isegm, CC in enumerate(networkx.connected_components(FullGraph)):
        for ib, iint in CC:
            bodysegm[ib, iint] = isegm

    nsegm = isegm + 1

    BodyLoop = np.zeros((nbody), dtype = int)
    Targets = np.zeros((nloop,maxlooplen), dtype = int)
    for il, CC in enumerate(networkx.connected_components(BodyGraph)):
        for ilb, ib in enumerate(CC):
            Targets[il,ilb] = ib
            BodyLoop[ib] = il

    print(f"{nsegm = }")

    bodynsegm = np.zeros((nbody), dtype = int)
    BodyHasContiguousGeneratingSegments = np.zeros((nbody), dtype = bool)

    for ib in range(nbody):

        unique, unique_indices, unique_inverse, unique_counts = np.unique(bodysegm[ib, :], return_index = True, return_inverse = True, return_counts = True)

        assert (unique == bodysegm[ib, unique_indices]).all()
        assert (unique[unique_inverse] == bodysegm[ib, :]).all()

        print()
        print(ib)
        print(bodysegm[ib, :])
        # print(unique)
        # print(unique_indices)
        # print(unique_inverse)
        # print(unique_counts)

        bodynsegm[ib] = unique.size

        BodyHasContiguousGeneratingSegments[ib] = ((unique_indices.max()+1) == bodynsegm[ib])
        
        print(BodyHasContiguousGeneratingSegments[ib])

    AllLoopsHaveContiguousGeneratingSegments = True
    for il in range(nloop):
        LoopHasContiguousGeneratingSegments = False
        for ilb in range(loopnb[il]):
            LoopHasContiguousGeneratingSegments = LoopHasContiguousGeneratingSegments or BodyHasContiguousGeneratingSegments[Targets[il,ilb]]

        AllLoopsHaveContiguousGeneratingSegments = AllLoopsHaveContiguousGeneratingSegments and LoopHasContiguousGeneratingSegments

    print(f'{AllLoopsHaveContiguousGeneratingSegments = }')

    # Accumulate constraints on segments. 
    # So far I've found zero constraints on segments. Is this because I only test on well-formed symmetries ?
    # TODO : prove that it is actually useless ?

    SegmConstraints = AccumulateSegmentConstraints(FullGraph, nbody, geodim, nsegm, bodysegm)
    
    for isegm in range(nsegm):
        assert len(SegmConstraints[isegm]) == 0

    # return



    # print()
    # print("**************************************************")
    # for isegm in range(nsegm):
    #     print()
    #     print("Segment number:", isegm)
    #     for icstr, Cstr in enumerate(SegmConstraints[isegm]):
    #         print()
    #         print("Constraint number:", icstr)
    #         print(Cstr)


    segm_to_body = np.zeros((nsegm), dtype = int)
    segm_to_iint = np.zeros((nsegm), dtype = int)

    # Choose generating segments as first contiguous intervals of generating loops
#     isegm = 0
#     for il in range(nloop):
# 
#         assert BodyHasContiguousGeneratingSegments[Targets[il,:loopnb[il]]].any()
# 
#         for ilb in range(loopnb[il]):
# 
#             assert bodynsegm[Targets[il,ilb]] == bodynsegm[Targets[il,0]]
# 
#         for ils in range(bodynsegm[Targets[il,0]]):
# 
#             segm_to_body[isegm] = Targets[il,0]
#             segm_to_iint[isegm] = ils
# 
#             isegm += 1
# 
#     assert isegm == nsegm


    # Choose generating segments as earliest possible times
    assigned_segms = set()

    for iint in range(nint_min):
        for ib in range(nbody):

            isegm = bodysegm[ib,iint]

            if not(isegm in assigned_segms):
                segm_to_body[isegm] = ib
                segm_to_iint[isegm] = iint
                assigned_segms.add(isegm)


    segmbody = [[] for isegm in range(nsegm)]

    for iint in range(nint_min):
        for ib in range(nbody):

            isegm = bodysegm[ib,iint]

            segmbody[isegm].append((ib,iint))



    BodyConstraints = AccumulateBodyConstraints(Sym_list, nbody, geodim)

    for ib in range(nbody):

        print()
        print("*************************************************")
        print()
        # print(f'{ib = }')
        print(f'loop {BodyLoop[ib]} body {ib}')
        for icstr, Sym in enumerate(BodyConstraints[ib]):
            assert Sym.BodyPerm[ib] == ib
            
            print()
            print(f'{icstr = }')
            print('Sym.SpaceRot')
            print(np.asarray(Sym.SpaceRot))
            print(f'{Sym.TimeRev}')
            print(f'Sym.TimeShift : {Sym.TimeShiftNum} / {Sym.TimeShiftDen}')

    print()
    # exit()



    segm_gen_to_target =  AccumulateSegmGenToTargetSym(FullGraph, nbody, geodim, nloop, nint_min, nsegm, bodysegm, segmbody, loopnb, Targets, segm_to_iint, segm_to_body)

    BinarySegm, Identity_detected = FindAllBinarySegments(segm_gen_to_target, nbody, nsegm, nint_min, bodysegm, CrashOnIdentity, mass, BodyLoop)

    All_Id = True

    count_tot = 0
    count_unique = 0
    for isegm in range(nsegm):
        for isegmp in range(isegm,nsegm):
            count_tot += sum(BinarySegm[(isegm, isegmp)]["SymCount"])
            count_unique += len(BinarySegm[(isegm, isegmp)]["SymCount"])

#             print()
#             print(isegm,isegmp)
#             print(BinarySegm[(isegm, isegmp)]["SymCount"])
# 
            for Sym in BinarySegm[(isegm, isegmp)]["SymList"]:

                All_Id = All_Id and Sym.IsIdentityRotAndTimeRev()

                # print()
                # print(Sym.SpaceRot)
                # print(Sym.TimeRev)

                # print(np.linalg.norm(Sym.SpaceRot - np.identity(geodim)) < 1e-11)
                # assert Sym.TimeRev == 1








    
    eps = 1e-12


    # print()
    # print('================================================')
    # print()


    All_params_basis = ComputeParamBasis_Body(MomCons, nbody, geodim, BodyConstraints)


    avg_param_per_k = np.zeros((nbody),dtype=np.float64)
    for ib in range(nbody):

        nparam_body = 0
        fill_num = 0
        k_mul = len(All_params_basis[ib])
        for k, NullSpace  in enumerate(All_params_basis[ib]):
            # print()
            # print(BodyLoop[ib],ib,k)
            # print(NullSpace)

            nparam_body += NullSpace.shape[2]
            fill_num += np.count_nonzero(NullSpace) 

        avg_param_per_k[ib] = nparam_body / k_mul
# 
        # print()
        # print(f'loop {BodyLoop[ib]} body {ib}')
        # print(nparam_body / k_mul)

    for il in range(nloop):

        for ilb in range(loopnb[il]):

            assert abs(avg_param_per_k[Targets[il,ilb]]  - avg_param_per_k[Targets[il,0]]) < eps


    nint = math.lcm(2, nint_min) 
    ncoeffs = nint//2 + 1

    all_coeffs = np.zeros((nbody,ncoeffs,geodim,2), dtype = np.float64)
    
    for ib in range(nbody):
        ncoeffs_min = len(All_params_basis[ib])
        for k in range(ncoeffs):

            kmin = (k % ncoeffs_min)

            NullSpace = All_params_basis[ib][kmin]
            nparam = NullSpace.shape[2]

            all_coeffs[ib,k,:,:] = np.matmul(NullSpace, np.random.rand(nparam))

    all_coeffs_c = all_coeffs.view(dtype=np.complex128)[...,0]
    all_pos = scipy.fft.irfft(all_coeffs_c, axis=1)

    AllConstraintAreRespected = True

    for ib in range(nbody):

        for icstr, Sym in enumerate(BodyConstraints[ib]):

            ConstraintIsRespected = True

            for iint in range(nint):

                assert (nint % Sym.TimeShiftDen) == 0

                tnum, tden = Sym.ApplyT(iint, nint)
                jint = tnum * nint // tden

                err = np.linalg.norm(all_pos[ib,iint,:] - np.matmul(Sym.SpaceRot, all_pos[ib,jint,:]))

                ConstraintIsRespected = ConstraintIsRespected and (err < eps)

            AllConstraintAreRespected = AllConstraintAreRespected and ConstraintIsRespected
            if not(ConstraintIsRespected):

                print(f'Constraint {icstr} is not respected')
            
    assert AllConstraintAreRespected

    nparam_body_sum = np.zeros((nbody),dtype=int)
    nparam_body_max = np.zeros((nbody),dtype=int)
    ncoeff_min_body = np.zeros((nbody),dtype=int)
    nnz_body = np.zeros((nbody),dtype=int)
    for ib in range(nbody):

        il = BodyLoop[ib]

        ncoeff_min_body[ib] = len(All_params_basis[ib])

        for k in range(ncoeff_min_body[ib]):

            nparam_body_sum[ib] += All_params_basis[ib][k].shape[2]
            nparam_body_max[ib] = max(nparam_body_max[ib], All_params_basis[ib][k].shape[2])
            nnz_body[ib] += np.count_nonzero(All_params_basis[ib][k])


    nparam_per_k_tot = 0
    nparam_per_k_before_tot = 0

    for il in range(nloop):

        for ilb in range(loopnb[il]):
# 
#             print()
#             print(il,ilb)
#             print(ncoeff_min_body[Targets[il,ilb]])
#             print(nparam_body_sum[Targets[il,ilb]])
            # print(nnz_body[Targets[il,ilb]])
# 
#             print(nparam_body[Targets[il,ilb]] / ncoeff_min_body[Targets[il,ilb]])

            assert nparam_body_sum[Targets[il,0]] / ncoeff_min_body[Targets[il,0]] == nparam_body_sum[Targets[il,ilb]] / ncoeff_min_body[Targets[il,ilb]]

        nparam_per_k_before_tot += geodim * 2 * loopnb[il]
        nparam_per_k_tot += nparam_body_sum[Targets[il,0]] /  len(All_params_basis[Targets[il,0]])



# 
#     # Choose loop generators with maximal exploitable FFT symmetry
#     loopgen = - np.ones((nloop), dtype = int)
#     loopnsegm = np.zeros((nloop), dtype = int)
#     for il in range(nloop):
#         for ilb in range(loopnb[il]):
# 
#             if BodyHasContiguousGeneratingSegments[Targets[il,ilb]]:
#                 loopgen[il] = Targets[il,ilb]
#                 break
# 
#         assert loopgen[il] >= 0


# 
#     # Choose loop generators with smallest 
#     loopgen = - np.ones((nloop), dtype = int)
#     loopnsegm = np.zeros((nloop), dtype = int)
#     for il in range(nloop):
#         loopgen[il] = Targets[il,0]












    
    all_params_basis_reoganized, all_nnz_k = reorganize_All_params_basis(All_params_basis)

    ncoeff_min_body_nnz = [all_nnz_k[ib].shape[0] for ib in range(nbody)]







    nint = 2 * nint_min
        
    # nint = 2 * math.lcm(*ncoeff_min_body)
    ncoeffs = nint //2 + 1

    # Create parameters
    all_params = []
    for ib in range(nbody):

        # print()
        # print(f'{ib = }')

        nparams_body = 0
        
        ncoeffs_min = len(All_params_basis[ib])
     
        for k in range(ncoeffs-1):
            l = (k % ncoeffs_min)

            NullSpace = All_params_basis[ib][l]
            nparams_body += NullSpace.shape[2]

        params_body = np.random.rand(nparams_body)
        all_params.append(params_body)
        
    # Parameters to all_pos through coeffs
    all_coeffs = np.zeros((nbody,ncoeffs,geodim,2), dtype = np.float64)
    
    for ib in range(nbody):

        nparams_body = 0
        
        ncoeffs_min = len(All_params_basis[ib])
        
        iparam = 0

        for k in range(ncoeffs-1):
            l = (k % ncoeffs_min)
            NullSpace = All_params_basis[ib][l]
            
            jparam = iparam + NullSpace.shape[2]
            
            all_coeffs[ib, k, : ,:] = np.dot(NullSpace, all_params[ib][iparam:jparam])
            
            iparam = jparam

        
    all_coeffs_c = all_coeffs.view(dtype=np.complex128)[...,0]
    all_pos = scipy.fft.irfft(all_coeffs_c, axis=1)

    # Make sure body constraints are respected

    AllConstraintAreRespected = True

    for ib in range(nbody):

        for icstr, Sym in enumerate(BodyConstraints[ib]):

            ConstraintIsRespected = True

            for iint in range(nint):

                assert (nint % Sym.TimeShiftDen) == 0

                tnum, tden = Sym.ApplyT(iint, nint)
                jint = tnum * nint // tden

                err = np.linalg.norm(all_pos[ib,iint,:] - np.matmul(Sym.SpaceRot, all_pos[ib,jint,:]))

                ConstraintIsRespected = ConstraintIsRespected and (err < eps)

            AllConstraintAreRespected = AllConstraintAreRespected and ConstraintIsRespected
            
            if not(ConstraintIsRespected):
                print(f'Body {ib} constraint {icstr} is not respected')
            
    assert AllConstraintAreRespected
    
    

    # return

    # nint = math.lcm(2, nint_min) 
    # nint = 2 * math.lcm(*ncoeff_min_body)
    ncoeffs = nint // 2 + 1
    
    all_coeffs_simple_c = np.zeros((nbody, ncoeffs, geodim), dtype=np.complex128)
    all_coeffs_a_c = np.zeros((nbody, ncoeffs, geodim), dtype=np.complex128)
    all_coeffs_b_c = np.zeros((nbody, ncoeffs, geodim), dtype=np.complex128)
    all_pos_slice_b = []
    all_pos_slice_a = []

    # Parameters to all_pos through coeffs
    all_coeffs = np.zeros((nbody,ncoeffs,geodim,2), dtype = np.float64)

    # Create parameters
    for ib in range(nbody):

        # print()
        # print(f'{ib = }')

        params_basis_reoganized = all_params_basis_reoganized[ib]
        nparam_per_period_body = params_basis_reoganized.shape[2]
        nnz_k = all_nnz_k[ib]
        
        ncoeffs_min = len(All_params_basis[ib])
        npr = (ncoeffs-1) //  ncoeff_min_body[ib]
        nperiods_body = npr * ncoeff_min_body_nnz[ib]


        params_body = np.random.random((nparam_per_period_body, npr, ncoeff_min_body_nnz[ib]))

        ip = 0
        for k in range(ncoeffs-1):
            l = (k % ncoeffs_min)
            NullSpace = All_params_basis[ib][l]
            if  (NullSpace.shape[2] != 0):

                p, q = divmod(ip,  ncoeff_min_body_nnz[ib])

                assert l == nnz_k[q]
                
                all_coeffs[ib, k, : ,:] = np.dot(NullSpace, params_body[:, p, q])
                ip+=1
                
        assert ip == nperiods_body
                
        
        assert npr * ncoeff_min_body[ib] == (ncoeffs-1)
        
        for idim in range(geodim):
            for i in range(ncoeff_min_body_nnz[ib]):
                for ipr in range(npr):
                
                    k = nnz_k[i] + ncoeff_min_body[ib] * ipr
    
                    all_coeffs_simple_c[ib, k, idim] = np.matmul(params_basis_reoganized[idim,i,:], params_body[:, ipr, i])
        
        coeffs_reorganized = np.einsum('ijk,klj->lji', params_basis_reoganized, params_body)
        coeffs_dense = np.zeros((npr, ncoeff_min_body[ib], geodim), dtype=np.complex128)
        coeffs_dense[:,nnz_k,:] = coeffs_reorganized
        
        all_coeffs_a_c[ib,:(ncoeffs-1),:] = coeffs_dense.reshape(((ncoeffs-1), geodim))
        
        
        params_basis_dense = np.zeros((geodim, ncoeff_min_body[ib], nparam_per_period_body), dtype=np.complex128)
        params_body_dense =  np.zeros((nparam_per_period_body, npr, ncoeff_min_body[ib]), dtype=np.float64)
        
        for ik, k in enumerate(nnz_k):
            
            params_basis_dense[:, k, :] = params_basis_reoganized[:, ik, :]
            params_body_dense[:, :, k] = params_body[:, :, ik]
        
        all_coeffs_b_c[ib,:(ncoeffs-1),:] = np.einsum('ijk,klj->lji', params_basis_dense, params_body_dense).reshape(((ncoeffs-1), geodim))
        
        # Dense version with lots of zeros
        
        ifft_b =  scipy.fft.rfft(params_body_dense, axis=1, n=2*npr)
        n_inter = npr+1

        fac = 1./(npr * ncoeff_min_body[ib])
        wo = np.exp(-2j*np.pi/nint)
        wref = 1.
        for m in range(n_inter):
            w = fac
            for j in range(ncoeff_min_body[ib]):
                ifft_b[:,m,j] *= w
                w *= wref
            wref *= wo
        
        meanval = np.matmul(params_basis_dense[:,0,:].real, params_body_dense[:,0,0]) / nint
        
        pos_slice = np.einsum('ijk,klj->li', params_basis_dense.real, ifft_b.real) + np.einsum('ijk,klj->li', params_basis_dense.imag, ifft_b.imag)
        
        for idim in range(geodim):
            pos_slice[:,idim] -= meanval[idim]
        
        all_pos_slice_b.append(pos_slice)
        
        
        # Sparse version with fewer zeros
        
        
        ifft_b =  scipy.fft.rfft(params_body, axis=1, n=2*npr)
        n_inter = npr+1

        fac = 1./(npr * ncoeff_min_body[ib])
        for m in range(n_inter):
            for j in range(ncoeff_min_body_nnz[ib]):
                w = fac * np.exp(-2j*np.pi*nnz_k[j] * m/nint)
                ifft_b[:,m,j] *= w


        
        pos_slice = np.einsum('ijk,klj->li', params_basis_reoganized.real, ifft_b.real) + np.einsum('ijk,klj->li', params_basis_reoganized.imag, ifft_b.imag)
        
        
        if nnz_k[0] == 0:
            
            meanval = np.matmul(params_basis_reoganized[:,0,:].real, params_body[:,0,0]) / nint            
            for idim in range(geodim):
                pos_slice[:,idim] -= meanval[idim]

        
        all_pos_slice_a.append(pos_slice)
        
        
        
        
        
        
        
        
        
        
        
                
    all_coeffs_c = all_coeffs.view(dtype=np.complex128)[...,0]
    assert np.linalg.norm(all_coeffs_c - all_coeffs_simple_c) < 1e-14
    assert np.linalg.norm(all_coeffs_c - all_coeffs_a_c) < 1e-14
    assert np.linalg.norm(all_coeffs_c - all_coeffs_b_c) < 1e-14

    # return
    

    all_pos = scipy.fft.irfft(all_coeffs_c, axis=1)
    
    for ib in range(nbody):
        
        pos_slice = all_pos_slice_a[ib]
        n_inter = pos_slice.shape[0]

        assert np.linalg.norm(all_pos[ib,:n_inter,:] - pos_slice) < 1e-14
        assert np.linalg.norm(all_pos_slice_a[ib] - pos_slice) < 1e-14
        assert np.linalg.norm(all_pos_slice_b[ib] - pos_slice) < 1e-14
        
        
        
        
        

    # Make sure body constraints are respected

    AllConstraintAreRespected = True

    for ib in range(nbody):
        print()
        print(f'Body {ib}')

        for icstr, Sym in enumerate(BodyConstraints[ib]):
            print(f'{icstr = }')
            
            assert (nint % Sym.TimeShiftDen) == 0

            ConstraintIsRespected = True

            for iint in range(nint):

                tnum, tden = Sym.ApplyT(iint, nint)
                jint = tnum * nint // tden
                
                err = np.linalg.norm(all_pos[ib,iint,:] - np.matmul(Sym.SpaceRot, all_pos[ib,jint,:]))
                
                # print(iint, jint, err)
                print(iint, jint)
                

                ConstraintIsRespected = ConstraintIsRespected and (err < eps)

            AllConstraintAreRespected = AllConstraintAreRespected and ConstraintIsRespected
            
            if not(ConstraintIsRespected):
                print(f'Body {ib} constraint {icstr} is not respected')
            
    assert AllConstraintAreRespected
    

    print(nint, nint_min)
    # print((nbody * nint_min) // nsegm)
    # print(nint // nint_min)
    # print(nint_min, nsegm)
    
    
        
    assert nint % nint_min == 0
    nint_fac = (nint // nint_min) 

    for il in range(nloop):
        

        print()
        print(il)
        
        expected_nparam = geodim * nint_fac * loopnb[il]
        
        for ilb in range(loopnb[il]):
            
            ib = Targets[il,ilb]
            
            params_basis_reoganized = all_params_basis_reoganized[ib]
            nparam_per_period_body = params_basis_reoganized.shape[2]
            nnz_k = all_nnz_k[ib]
            
            npr = (ncoeffs-1) //  ncoeff_min_body[ib]
            nperiods_body = npr * ncoeff_min_body_nnz[ib]
        
            nparams = nparam_per_period_body * npr * ncoeff_min_body_nnz[ib]
            
            # assert (nparams == expected_nparam)
            # assert (nparams == expected_nparam) or not(All_Id)
            
            if ilb == 0:
                nparam_per_period_body_ref = nparam_per_period_body
            else:
                assert nparam_per_period_body_ref == nparam_per_period_body
            
            
            pos_slice = all_pos_slice_b[ib]
            n_inter = pos_slice.shape[0]
            
            assert n_inter == (npr+1)

            # print(ib, nparams, nparam_per_period_body, npr, ncoeff_min_body_nnz[ib])
            # print(ib, n_inter-1)
            print(ib, nparams, nparam_per_period_body, npr, ncoeff_min_body_nnz[ib], ncoeff_min_body[ib])
            # print(ib, nparams,)
            
            # Size of biggest possible contiguous segment
            # assert npr % nint_fac == 0
            # print(ib, nparams, npr, nint_fac )

        

    
    
    
    








    print('*****************************************')
    print('')
    print(f'{AllConstraintAreRespected = }')     
    print(f'{Identity_detected = }')
    print(f'All binary transforms are identity: {All_Id}')

    print()
    print(f"total binary interaction count: {count_tot}")
    print(f"total expected binary interaction count: {nint_min * nbody * (nbody-1)//2}")
    print(f"unique binary interaction count: {count_unique}")



    print()
    print(f"ratio of total to unique binary interactions : {count_tot / count_unique}")
    print(f'ratio of integration intervals to segments : {(nbody * nint_min) / nsegm}')
    print(f"ratio of parameters before and after constraints: {nparam_per_k_before_tot / nparam_per_k_tot}")


    reduction_ratio = count_tot / count_unique
    assert abs((count_tot / count_unique)  - reduction_ratio) < eps
    assert abs(((nbody * nint_min) / nsegm) - reduction_ratio) < eps
    assert abs((nparam_per_k_before_tot / nparam_per_k_tot)  - reduction_ratio) < eps



    # return
# 

    # MakePlots = False
    MakePlots = True

    if MakePlots:


        nnodes = nbody*nint_min
        node_color = np.zeros(nnodes)

        for icolor, CC in enumerate(networkx.connected_components(FullGraph)):
            for node in CC:
                inode = node[1] + nint_min * node[0]
                node_color[inode] = icolor

        nedges = len(FullGraph.edges)
        edge_color = np.zeros(nedges)

        for iedge, edge in enumerate(FullGraph.edges):

            ContainsDirect = False
            ContainsIndirect = False

            for Sym in FullGraph.edges[edge]["SymList"]:

                if Sym.TimeRev == 1:
                    ContainsDirect = True
                else:
                    ContainsIndirect = True

            if ContainsDirect:
                if ContainsIndirect:
                    color = 2
                else:
                    color = 1
            else:
                color = 0

            edge_color[iedge] = color

        pos = {i:(i[1],i[0]) for i in FullGraph.nodes }


        # edgelist = []
    #     for iedge, edge in enumerate(FullGraph.edges):
    # 
    #         Sym = FullGraph.edges[edge]["SymList"][0]
    # 
    #         issqrtid = (np.linalg.norm(np.matmul(Sym.SpaceRot,Sym.SpaceRot) - np.identity(geodim)) < 1e-12)
    #         isid = (np.linalg.norm(Sym.SpaceRot - np.identity(geodim)) < 1e-12)
    # 
    #         if issqrtid and not(isid):
    # 
    #             edgelist.append(edge)

        fig, ax = plt.subplots()

        # networkx.draw(
        #     FullGraph,
        #     pos = pos,
        #     labels = {i:i for i in FullGraph.nodes},
        #     node_color = node_color,
        #     cmap = 'jet',
        #     arrows = False,
        #     # connectionstyle = "arc3,rad=0.1",
        # )

        networkx.draw_networkx_nodes(
            FullGraph,
            pos = pos,
            ax = ax,
            node_color = node_color,
            # cmap = 'tab20',
            cmap = 'turbo',
        )

        # networkx.draw_networkx_labels(
        #     FullGraph,
        #     pos = pos,
        #     ax = ax,
            # labels = {i:i for i in FullGraph.nodes},
        # )
# 
        networkx.draw_networkx_edges(
            FullGraph,
            pos = pos,
            ax = ax,
            arrows = True,
            connectionstyle = "arc3,rad=0.1",
            edge_color = edge_color,
            edge_vmin = 0,
            edge_vmax = 1,
            edge_cmap = colormaps['Set1'],
        )


        plt.axis('off')
        fig.tight_layout()

        plt.savefig('./NewSym_data/graph.pdf')
        plt.close()



