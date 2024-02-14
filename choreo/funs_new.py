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

from choreo.cython.funs_new import *


def ContainsDoubleEdges(SegmGraph):

    for edge in SegmGraph.edges:

        if (len(SegmGraph.edges[edge]["SymList"]) > 1):
            return True
    
    return False

def ContainsSelfReferingTimeRevSegment(SegmGraph):

    CCs = networkx.connected_components(SegmGraph)

    for CC in CCs:

        for segm, segmp in itertools.combinations(CC,2):

            if (segm[1] == segmp[1]):

                path = networkx.shortest_path(SegmGraph, source = segm, target = segmp)

                TimeRev = 1
                pathlen = len(path)

                for ipath in range(1,pathlen):

                    if (path[ipath-1] > path[ipath]):
                        edge = (path[ipath], path[ipath-1])
                    else:
                        edge = (path[ipath-1], path[ipath])

                    TimeRev *=  SegmGraph.edges[edge]["SymList"][0].TimeRev

                if (TimeRev == -1):
                    
                    return True
                
    return False

def Build_BodyTimeGraph(nbody, nint, Sym_list, Tfun = None, VelSym = False):
    
    if Tfun is None:
        return ValueError('Invalid Tfun')

    Graph = networkx.Graph()
    for ib in range(nbody):
        for iint in range(nint):
            Graph.add_node((ib,iint))

    for Sym in Sym_list:

        SymInv = Sym.Inverse()

        for ib in range(nbody):

            ib_target = Sym.BodyPerm[ib]

            for iint in range(nint):

                tnum_target, tden_target = Tfun(Sym, iint, nint)

                assert nint % tden_target == 0

                iint_target = (tnum_target * (nint // tden_target) + nint) % nint

                node_source = (ib       , iint       )
                node_target = (ib_target, iint_target)
                
                if node_source <= node_target :

                    edge = (node_source, node_target)
                    EdgeSym = Sym

                else:

                    edge = (node_target, node_source)
                    EdgeSym = SymInv

                if VelSym:
                    EdgeSym = EdgeSym.TimeDerivative()
                
                edge_dict = Graph.edges.get(edge)
                
                if edge_dict is None:
                    Graph.add_edge(*edge, SymList = [EdgeSym])
                
                else:
                    
                    for OtherEdgeSym in edge_dict["SymList"]:
                        if EdgeSym.IsSameRotAndTimeRev(OtherEdgeSym):
                            break
                    else:
                        edge_dict["SymList"].append(EdgeSym)
    
    return Graph

Build_SegmGraph    = functools.partial(Build_BodyTimeGraph, Tfun = ActionSym.ApplyTSegm, VelSym = False)
# Build_InstGraphPos = functools.partial(Build_BodyTimeGraph, Tfun = ActionSym.ApplyT    , VelSym = False)
# Build_InstGraphVel = functools.partial(Build_BodyTimeGraph, Tfun = ActionSym.ApplyT    , VelSym = True)

def Build_SegmGraph_NoPb(
    nbody,
    nint,
    Sym_list,
    current_recursion = 1,
    max_recursion = 5,
):

    if (current_recursion > max_recursion):
        raise ValueError("Achieved max recursion level in Build_SegmGraph")

    SegmGraph = Build_SegmGraph(nbody, nint, Sym_list)

    if ContainsDoubleEdges(SegmGraph):

        return Build_SegmGraph_NoPb(
            nbody = nbody,
            nint = 2*nint,
            Sym_list = Sym_list,
            current_recursion = current_recursion+1,
            max_recursion = max_recursion,
        )

    if ContainsSelfReferingTimeRevSegment(SegmGraph):

        return Build_SegmGraph_NoPb(
            nbody = nbody,
            nint = 2*nint,
            Sym_list = Sym_list,
            current_recursion = current_recursion+1,
            max_recursion = max_recursion,
        ) 

    return SegmGraph, nint

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

            edge_dict = BodyGraph.edges.get(edge)
            
            if edge_dict is None:
                BodyGraph.add_edge(*edge, SymList = [EdgeSym])
                
            else:

                for FoundSym in edge_dict["SymList"]:
                    if EdgeSym.IsSameRotAndTime(FoundSym):
                        break

                else:
                    edge_dict["SymList"].append(EdgeSym)

    return BodyGraph

def AppendIfNot(CstrList, Constraint, test_callback):

    if not(test_callback(Constraint)):

        for FoundCstr in CstrList:

            if test_callback(Constraint.Compose(FoundCstr)):
                break

            if test_callback(Constraint.Compose(FoundCstr.Inverse())):
                break

        else:

            CstrList.append(Constraint)

AppendIfNotSameRotAndTime = functools.partial(AppendIfNot, test_callback = ActionSym.IsIdentityRotAndTime)
AppendIfNotSame = functools.partial(AppendIfNot, test_callback = ActionSym.IsIdentity)
AppendIfNotSamePermAndRot = functools.partial(AppendIfNot, test_callback = ActionSym.IsIdentityPermAndRot)

def AccumulateBodyConstraints(Sym_list, nbody, geodim):

    BodyConstraints = [list() for _ in range(nbody)]

    SimpleBodyGraph = networkx.Graph()
    for ib in range(nbody):
        SimpleBodyGraph.add_node(ib)

    for Sym in Sym_list:

        for ib in range(nbody):

            ib_target = Sym.BodyPerm[ib]

            if ib != ib_target:

                if ib > ib_target:
                    edge = (ib_target, ib)
                    EdgeSym = Sym.Inverse()
                else:
                    edge = (ib, ib_target)
                    EdgeSym = Sym

                if not(edge in SimpleBodyGraph.edges):

                    SimpleBodyGraph.add_edge(*edge, Sym = EdgeSym)
                    
    for Sym in Sym_list:

        for ib in range(nbody):

            ib_target = Sym.BodyPerm[ib]

            if ib == ib_target:

                AppendIfNotSameRotAndTime(BodyConstraints[ib], Sym)                
            else:
                    
                if ib > ib_target:
                    edge = (ib_target, ib)
                    EdgeSym = Sym.Inverse()
                else:
                    edge = (ib, ib_target)
                    EdgeSym = Sym
                    

                edge_dict = SimpleBodyGraph.edges.get(edge)
                
                if not(edge_dict is None):
                
                    ParallelEdgeSym = edge_dict["Sym"]

                    Constraint = EdgeSym.Inverse().Compose(ParallelEdgeSym)
                    assert Constraint.BodyPerm[edge[0]] == edge[0]
                    AppendIfNotSameRotAndTime(BodyConstraints[edge[0]], Constraint)

                    Constraint = EdgeSym.Compose(ParallelEdgeSym.Inverse())
                    assert Constraint.BodyPerm[edge[1]] == edge[1]
                    AppendIfNotSameRotAndTime(BodyConstraints[edge[1]], Constraint)




    Cycles = networkx.cycle_basis(SimpleBodyGraph)

    for Cycle in itertools.chain(SimpleBodyGraph.edges, Cycles):

        Cycle_len = len(Cycle)
        FirstBody = Cycle[0]

        FirstBodyConstraint = ActionSym.Identity(nbody, geodim)
        for iedge in range(Cycle_len):
            
            ibeg = Cycle[iedge]
            iend = Cycle[(iedge+1)%Cycle_len]
            
            if (ibeg > iend):
                Sym = SimpleBodyGraph.edges[(iend,ibeg)]["Sym"].Inverse()

            else:
                Sym = SimpleBodyGraph.edges[(ibeg,iend)]["Sym"]
                
            assert Sym.BodyPerm[ibeg] == iend
                
            FirstBodyConstraint = Sym.Compose(FirstBodyConstraint)
        
        assert FirstBodyConstraint.BodyPerm[FirstBody] == FirstBody

        if not(FirstBodyConstraint.IsIdentityRotAndTime()):
            
            path_from_FirstBody = networkx.shortest_path(SimpleBodyGraph, source = FirstBody)

            # Now add the Cycle constraints to every body in the cycle
            for ib in Cycle:

                FirstBodyToibSym = ActionSym.Identity(nbody, geodim)

                path = path_from_FirstBody[ib]            
                pathlen = len(path)

                for ipath in range(1,pathlen):

                    if (path[ipath-1] > path[ipath]):

                        edge = (path[ipath], path[ipath-1])
                        Sym = SimpleBodyGraph.edges[edge]["Sym"].Inverse()

                    else:

                        edge = (path[ipath-1], path[ipath])
                        Sym = SimpleBodyGraph.edges[edge]["Sym"]

                    FirstBodyToibSym = Sym.Compose(FirstBodyToibSym)

                assert FirstBodyToibSym.BodyPerm[FirstBody] == ib
                
                Constraint = FirstBodyConstraint.Conjugate(FirstBodyToibSym)

                assert Constraint.BodyPerm[ib] == ib

                AppendIfNotSameRotAndTime(BodyConstraints[ib], Constraint)

    return BodyConstraints

def AccumulateSegmentConstraints(SegmGraph, nbody, geodim, nsegm, bodysegm):
    # Accumulate constraints on segments, assuming there is only one symmetry per edge in the graph (checked before)

    SegmConstraints = [ list() for isegm in range(nsegm)]

    Cycles = networkx.cycle_basis(SegmGraph)

    for Cycle in Cycles:

        isegm = bodysegm[*Cycle[0]]
        Cycle_len = len(Cycle)
        
        Constraint = ActionSym.Identity(nbody, geodim)

        for iedge in range(Cycle_len):
            
            ibeg = Cycle[iedge]
            iend = Cycle[(iedge+1)%Cycle_len]

            if (ibeg <= iend):
                Constraint = SegmGraph.edges[(ibeg,iend)]["SymList"][0].Compose(Constraint)
                
            else:
                Constraint = SegmGraph.edges[(iend,ibeg)]["SymList"][0].Inverse().Compose(Constraint)

        AppendIfNotSameRotAndTime(SegmConstraints[isegm], Constraint)

    return SegmConstraints

def AccumulateInstConstraints(Sym_list, nbody, geodim, nint, VelSym=False):

    InstConstraints = [list() for iint in range(nint)]

    InstGraph = networkx.Graph()
    for iint in range(nint):
        InstGraph.add_node(iint)

    for Sym in Sym_list:

        if VelSym:
            Sym = Sym.TimeDerivative()

        SymInv = Sym.Inverse()

        for iint in range(nint):

            tnum_target, tden_target = Sym.ApplyT(iint, nint)

            assert nint % tden_target == 0

            iint_target = (tnum_target * (nint // tden_target) + nint) % nint

            if iint <= iint_target :
                edge = (iint, iint_target)
                EdgeSym = Sym
            else:
                edge = (iint_target, iint)
                EdgeSym = SymInv

            if not(edge in InstGraph.edges):
                InstGraph.add_edge(*edge, Sym = EdgeSym)
                
    for Sym in Sym_list:
        
        if VelSym:
            Sym = Sym.TimeDerivative()
        
        SymInv = Sym.Inverse()

        for iint in range(nint):

            tnum_target, tden_target = Sym.ApplyT(iint, nint)

            assert nint % tden_target == 0

            iint_target = (tnum_target * (nint // tden_target) + nint) % nint

            if iint == iint_target:

                AppendIfNotSamePermAndRot(InstConstraints[iint], Sym)                
                
            else:
                    
                if iint <= iint_target :
                    edge = (iint, iint_target)
                    EdgeSym = Sym
                else:
                    edge = (iint_target, iint)
                    EdgeSym = SymInv
                    
                edge_dict = InstGraph.edges.get(edge)
                
                if not(edge_dict is None):
                
                    ParallelEdgeSym = edge_dict["Sym"]

                    Constraint = EdgeSym.Inverse().Compose(ParallelEdgeSym)
                    tnum_target, tden_target = Constraint.ApplyT(edge[0], nint)
                    assert nint % tden_target == 0
                    assert edge[0] == (tnum_target * (nint // tden_target) + nint) % nint
                    AppendIfNotSamePermAndRot(InstConstraints[edge[0]], Constraint)

                    Constraint = EdgeSym.Compose(ParallelEdgeSym.Inverse())
                    tnum_target, tden_target = Constraint.ApplyT(edge[1], nint)
                    assert nint % tden_target == 0
                    assert edge[1] == (tnum_target * (nint // tden_target) + nint) % nint
                    AppendIfNotSamePermAndRot(InstConstraints[edge[1]], Constraint)

    Cycles = networkx.cycle_basis(InstGraph)

    for Cycle in itertools.chain(InstGraph.edges, Cycles):

        Cycle_len = len(Cycle)
        FirstInst = Cycle[0]

        FirstInstConstraint = ActionSym.Identity(nbody, geodim)
        for iedge in range(Cycle_len):
                        
            ibeg = Cycle[iedge]
            iend = Cycle[(iedge+1)%Cycle_len]

            if (ibeg > iend):
                Sym = InstGraph.edges[(iend,ibeg)]["Sym"].Inverse()

            else:
                Sym = InstGraph.edges[(ibeg,iend)]["Sym"]
            
            tnum_target, tden_target = Sym.ApplyT(ibeg, nint)
            assert nint % tden_target == 0
            assert iend == (tnum_target * (nint // tden_target) + nint) % nint

            FirstInstConstraint = Sym.Compose(FirstInstConstraint)
        
        tnum_target, tden_target = FirstInstConstraint.ApplyT(FirstInst, nint)
        assert nint % tden_target == 0
        assert FirstInst == (tnum_target * (nint // tden_target) + nint) % nint

        if not(FirstInstConstraint.IsIdentityPermAndRotAndTimeRev()):
            
            path_from_FirstInst = networkx.shortest_path(InstGraph, source = FirstInst)

            # Now add the Cycle constraints to every instant in the cycle
            for iint in Cycle:

                FirstInstToiintSym = ActionSym.Identity(nbody, geodim)

                path = path_from_FirstInst[iint]            
                pathlen = len(path)

                for ipath in range(1, pathlen):

                    if (path[ipath-1] > path[ipath]):

                        edge = (path[ipath], path[ipath-1])
                        Sym = InstGraph.edges[edge]["Sym"].Inverse()

                    else:

                        edge = (path[ipath-1], path[ipath])
                        Sym = InstGraph.edges[edge]["Sym"]

                    FirstInstToiintSym = Sym.Compose(FirstInstToiintSym)

                tnum_target, tden_target = FirstInstToiintSym.ApplyT(FirstInst, nint)
                assert nint % tden_target == 0
                assert iint == (tnum_target * (nint // tden_target) + nint) % nint
                
                Constraint = FirstInstConstraint.Conjugate(FirstInstToiintSym)

                tnum_target, tden_target = Constraint.ApplyT(iint, nint)
                assert nint % tden_target == 0
                assert iint == (tnum_target * (nint // tden_target) + nint) % nint

                AppendIfNotSamePermAndRot(InstConstraints[iint], Constraint)

    return InstConstraints
                    
def AccumulateSegmGenToTargetSym(SegmGraph, nbody, geodim, nint_min, nsegm, bodysegm, segm_to_iint, segm_to_body):
    
    segmbody = [[] for isegm in range(nsegm)]
    for iint in range(nint_min):
        for ib in range(nbody):
            isegm = bodysegm[ib,iint]
            segmbody[isegm].append((ib,iint))

    segm_gen_to_target = [ [ None for iint in range(nint_min)] for ib in range(nbody) ]

    for isegm, ib_iint_list  in enumerate(segmbody):

        segmgen = (segm_to_body[isegm], segm_to_iint[isegm])
        
        isegmgen = bodysegm[*segmgen] 
        assert isegm == isegmgen

        path_from_segmgen = networkx.shortest_path(SegmGraph, source = segmgen)

        for ib, iint in ib_iint_list:

            segm = (ib, iint)

            GenToTargetSym = ActionSym.Identity(nbody, geodim)

            path = path_from_segmgen[segm]
            pathlen = len(path)

            for ipath in range(1,pathlen):

                if (path[ipath-1] > path[ipath]):

                    edge = (path[ipath], path[ipath-1])
                    Sym = SegmGraph.edges[edge]["SymList"][0].Inverse()

                else:

                    edge = (path[ipath-1], path[ipath])
                    Sym = SegmGraph.edges[edge]["SymList"][0]

                GenToTargetSym = Sym.Compose(GenToTargetSym)

            tnum_target, tden_target = GenToTargetSym.ApplyTSegm(segm_to_iint[isegm], nint_min)
            assert nint_min % tden_target == 0
            assert iint == (tnum_target * (nint_min // tden_target) + nint_min) % nint_min  

            segm_gen_to_target[ib][iint] = GenToTargetSym

    return segm_gen_to_target                        

def FindAllBinarySegments(segm_gen_to_target, nbody, nsegm, nint_min, bodysegm, CrashOnIdentity, mass, BodyLoop):

    Identity_detected = False

    BinarySegm = {}

    for isegm in range(nsegm):
        for isegmp in range(isegm,nsegm):
            BinarySegm[(isegm, isegmp)] = {
                "SymList" : []      ,
                "SymCount" : []     ,
                "ProdMassSum" : []  ,
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

                isym = 0 # In case BinarySegm[bisegm]["SymList"] is empty
                for isym, FoundSym in enumerate(BinarySegm[bisegm]["SymList"]):
                    
                    if Sym.IsSameRotAndTimeRev(FoundSym):
                        break

                    if (isegm == isegmp):
                        SymInv = Sym.Inverse()
                        if SymInv.IsSameRotAndTimeRev(FoundSym):
                            break
                        
                else:
                    BinarySegm[bisegm]["SymList"].append(Sym)
                    BinarySegm[bisegm]["SymCount"].append(0)
                    BinarySegm[bisegm]["ProdMassSum"].append(0.)

                BinarySegm[bisegm]["SymCount"][isym] += 1
                BinarySegm[bisegm]["ProdMassSum"][isym] += mass[BodyLoop[ib]]*mass[BodyLoop[ibp]]

    return BinarySegm, Identity_detected

def ComputeParamBasis_InitVal(nbody, geodim, InstConstraints, mass, BodyLoop, MomCons=True, eps=1e-12):

    ncstr = len(InstConstraints)
    
    nbuf_nomomcons = ncstr * nbody * geodim * nbody * geodim
    nbuf = nbuf_nomomcons
    if MomCons:
       nbuf +=  geodim * nbody * geodim
    
    cstr_buf = np.zeros((nbuf), dtype = np.float64)
    
    # I do not use reshape here because I want an error to throw if for some reason the data needs to be copied
    # cf https://stackoverflow.com/a/14271298
    cstr_mat = cstr_buf[:nbuf_nomomcons].view()
    cstr_mat.shape = (ncstr, nbody, geodim, nbody, geodim)

    for icstr, Sym in enumerate(InstConstraints):
        for ib in range(nbody): 
            
            jb = Sym.BodyPerm[ib]
                   
            for idim in range(geodim):
                for jdim in range(geodim):

                    cstr_mat[icstr, ib, idim, jb, jdim] = Sym.SpaceRot[idim, jdim]
                    
                cstr_mat[icstr, ib, idim, ib, idim] -= 1

    if MomCons:
        cstr_mat = cstr_buf[nbuf_nomomcons:].view()
        cstr_mat.shape = (geodim, nbody, geodim)
        
        for ib in range(nbody): 
            for idim in range(geodim):
                cstr_mat[idim, ib, idim] = mass[BodyLoop[ib]]

    cstr_mat = cstr_buf.reshape((-1, nbody*geodim))

    NullSpace = choreo.scipy_plus.linalg.null_space(cstr_mat)
    
    nparam = NullSpace.shape[1]
    NullSpace = NullSpace.reshape(nbody, geodim, nparam)

    for ib in range(nbody): 
        for idim in range(geodim):
            for iparam in range(nparam):

                if abs(NullSpace[ib, idim, iparam]) < eps:
                    NullSpace[ib, idim, iparam] = 0
       
    return NullSpace

def ComputeParamBasis_Loop(MomCons, nbody, nloop, loopgen, geodim, LoopGenConstraints, eps=1e-12):

    if (MomCons):
        raise NotImplementedError("Momentum conservation as a constraint is not available at the moment")
    else:

        All_params_basis = []

        for il in range(nloop):
            ib = loopgen[il]

            all_time_dens = []

            for Sym in LoopGenConstraints[il]:
                assert Sym.BodyPerm[ib] == ib
                all_time_dens.append(Sym.TimeShiftDen)

            ncoeffs_min =  math.lcm(*all_time_dens)

            ncstr = len(LoopGenConstraints[il])
            
            NullSpace_all = []

            for k in range(ncoeffs_min):
            
                cstr_mat = np.zeros((ncstr, geodim, 2, geodim, 2), dtype = np.float64)

                for icstr, Sym in enumerate(LoopGenConstraints[il]):

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
    
    nloop = len(All_params_basis)
    geodim = All_params_basis[0][0].shape[0]
    
    nnz_k_list = []
    params_basis_reorganized_list = []
    
    for il in range(nloop):
        
        params_basis = All_params_basis[il]
        ncoeff_min = len(params_basis)
        
        last_nparam = None
        nnz_k = []
        for k in range(ncoeff_min):
            
            nparam_now = params_basis[k].shape[2]
            
            if last_nparam is None and nparam_now != 0:
                last_nparam = nparam_now
            elif nparam_now != 0:
                assert nparam_now == last_nparam
            
            if nparam_now != 0:
                nnz_k.append(k)
        
        if last_nparam is None:
            last_nparam = 0
                
        nnz_k = np.array(nnz_k, dtype=np.intp)
        nnz_k_list.append(nnz_k)

        params_basis_reorganized = np.empty((geodim, nnz_k.shape[0], last_nparam), dtype=np.complex128)    
        
        for ik, k in enumerate(nnz_k):
            
            params_basis_reorganized[:,ik,:] = params_basis[k][:,0,:] + 1j*params_basis[k][:,1,:]
        
        params_basis_reorganized_list.append(params_basis_reorganized)

    return params_basis_reorganized_list, nnz_k_list

def ExploreGlobalShifts_BuildSegmGraph(geodim, nbody, nloop, loopnb, Targets, nint_min, Sym_list):

    # Making sure nint_min is big enough
    SegmGraph, nint_min = Build_SegmGraph_NoPb(nbody, nint_min, Sym_list)
    
    for i_shift in range(nint_min):
        
        if i_shift != 0:
            
            GlobalTimeShift = ActionSym(
                BodyPerm  = np.array(range(nbody), dtype = np.int_) ,
                SpaceRot  = np.identity(geodim, dtype = np.float64) ,
                TimeRev   = 1                                       ,
                TimeShiftNum = i_shift                              ,
                TimeShiftDen = nint_min                             ,
            )
            
            Shifted_sym_list = []
            for Sym in Sym_list:
                Shifted_sym_list.append(Sym.Conjugate(GlobalTimeShift))
            Sym_list = Shifted_sym_list
        
            SegmGraph = Build_SegmGraph(nbody, nint_min, Sym_list)

        bodysegm = np.zeros((nbody, nint_min), dtype = int)
        for isegm, CC in enumerate(networkx.connected_components(SegmGraph)):
            for ib, iint in CC:
                bodysegm[ib, iint] = isegm

        nsegm = isegm + 1
        
        bodynsegm = np.zeros((nbody), dtype = int)
        BodyHasContiguousGeneratingSegments = np.zeros((nbody), dtype = bool)

        for ib in range(nbody):

            unique, unique_indices, unique_inverse, unique_counts = np.unique(bodysegm[ib, :], return_index = True, return_inverse = True, return_counts = True)

            assert (unique == bodysegm[ib, unique_indices]).all()
            assert (unique[unique_inverse] == bodysegm[ib, :]).all()

            bodynsegm[ib] = unique.size
            BodyHasContiguousGeneratingSegments[ib] = ((unique_indices.max()+1) == bodynsegm[ib])
            
        AllLoopsHaveContiguousGeneratingSegments = True
        for il in range(nloop):
            LoopHasContiguousGeneratingSegments = False
            for ilb in range(loopnb[il]):
                LoopHasContiguousGeneratingSegments = LoopHasContiguousGeneratingSegments or BodyHasContiguousGeneratingSegments[Targets[il,ilb]]

            AllLoopsHaveContiguousGeneratingSegments = AllLoopsHaveContiguousGeneratingSegments and LoopHasContiguousGeneratingSegments
        
        if AllLoopsHaveContiguousGeneratingSegments:
            break
    
    else:
        
        raise ValueError("Could not find time shift such that all loops have contiguous generating segments")

    # print(f"Required {i_shift} shifts to find reference such that all loops have contiguous generating segments")
    
    return SegmGraph, nint_min, nsegm, bodysegm, BodyHasContiguousGeneratingSegments, Sym_list

def DetectLoops(Sym_list, nbody, nint_min_fac = 1):

    All_den_list_on_entry = []
    for Sym in Sym_list:
        All_den_list_on_entry.append(Sym.TimeShiftDen)

    nint_min = nint_min_fac * math.lcm(*All_den_list_on_entry) # ensures that all integer divisions will have zero remainder
    
    BodyGraph = Build_BodyGraph(nbody, Sym_list)

    nloop = sum(1 for _ in networkx.connected_components(BodyGraph))
    
    loopnb = np.zeros((nloop), dtype = int)

    for il, CC in enumerate(networkx.connected_components(BodyGraph)):
        loopnb[il] = len(CC)

    maxlooplen = loopnb.max()
    
    BodyLoop = np.zeros((nbody), dtype = int)
    Targets = np.zeros((nloop,maxlooplen), dtype = int)
    for il, CC in enumerate(networkx.connected_components(BodyGraph)):
        for ilb, ib in enumerate(CC):
            Targets[il,ilb] = ib
            BodyLoop[ib] = il

    return nint_min, nloop, loopnb, BodyLoop, Targets, BodyGraph
        
def PlotTimeBodyGraph(Graph, nbody, nint_min, filename):

    nnodes = nbody*nint_min
    node_color = np.zeros(nnodes)

    for icolor, CC in enumerate(networkx.connected_components(Graph)):
        for node in CC:
            inode = node[1] + nint_min * node[0]
            node_color[inode] = icolor

    nedges = len(Graph.edges)
    edge_color = np.zeros(nedges)

    for iedge, (key, edge) in enumerate(Graph.edges.items()):

        ContainsDirect = False
        ContainsIndirect = False

        for Sym in edge["SymList"]:

            if Sym.TimeRev == 1:
                ContainsDirect = True
            else:
                ContainsIndirect = True

        # if ContainsDirect:
        #     if ContainsIndirect:
        #         color = 2
        #     else:
        #         color = 1
        # else:
        #     color = 0
        
        if ContainsIndirect:
            fac = -1
        else:
            fac = 1
        
        color = len(edge["SymList"]) * fac

        edge_color[iedge] = color

    # print(edge_color)
    
    pos = {i:(i[1],i[0]) for i in Graph.nodes }

    fig, ax = plt.subplots()

    networkx.draw_networkx_nodes(
        Graph,
        pos = pos,
        ax = ax,
        node_color = node_color,
        # cmap = 'tab20',
        cmap = 'turbo',
    )
    
    color_min = edge_color.min()
    color_max = edge_color.max()
    
    edge_vmax = max(abs(color_min), abs(color_max))
    edge_vmin = - edge_vmax

    networkx.draw_networkx_edges(
        Graph,
        pos = pos,
        ax = ax,
        arrows = True,
        connectionstyle = "arc3,rad=0.1",
        edge_color = edge_color,
        edge_vmin = edge_vmin,
        edge_vmax = edge_vmax,
        edge_cmap = colormaps['Set1'],
    )

    plt.axis('off')
    fig.tight_layout()
    
    plt.savefig(filename)
    plt.close()

def CountSegmentBinaryInteractions(BinarySegm, nsegm):
    
    All_Id = True

    count_tot = 0
    count_unique = 0
    for isegm in range(nsegm):
        for isegmp in range(isegm,nsegm):
            count_tot += sum(BinarySegm[(isegm, isegmp)]["SymCount"])
            count_unique += len(BinarySegm[(isegm, isegmp)]["SymCount"])

            for Sym in BinarySegm[(isegm, isegmp)]["SymList"]:

                All_Id = All_Id and Sym.IsIdentityRotAndTimeRev()    
    
    return All_Id, count_tot, count_unique

def AssertAllBodyConstraintAreRespected(LoopGenConstraints, all_pos, eps=1e-12):
    # Make sure loop constraints are respected

    nint = all_pos.shape[1]
    
    for il, Constraints in enumerate(LoopGenConstraints):

        for icstr, Sym in enumerate(Constraints):

            assert (nint % Sym.TimeShiftDen) == 0

            ConstraintIsRespected = True

            for iint in range(nint):

                tnum, tden = Sym.ApplyT(iint, nint)
                jint = tnum * nint // tden
                
                err = np.linalg.norm(all_pos[il,jint,:] - np.matmul(Sym.SpaceRot, all_pos[il,iint,:]))

                assert (err < eps)
                
def AssertAllSegmGenConstraintsAreRespected(gensegm_to_all, nint_min, bodysegm, loopgen, gensegm_to_body, gensegm_to_iint , BodyLoop, all_pos, eps=1e-12):

    nloop = all_pos.shape[0]
    nint = all_pos.shape[1]
    geodim = all_pos.shape[2]

    for il in range(nloop):
        
        ib = loopgen[il] # because only loops have been computed in all_pos so far.
        
        for iint in range(nint_min):
            
            isegm = bodysegm[ib, iint]
            
            Sym = gensegm_to_all[ib][iint]
            
            ib_source = gensegm_to_body[isegm]
            iint_source = gensegm_to_iint[isegm]
            
            il_source = BodyLoop[ib_source]
            assert il_source == il
            
            segm_size = nint // nint_min

            ibeg_source = iint_source * segm_size          
            iend_source = ibeg_source + segm_size
            assert iend_source <= nint
            
            # One position at a time
            for iint_s in range(ibeg_source, iend_source):
                
                tnum, tden = Sym.ApplyT(iint_s, nint)
                
                assert nint % tden == 0
                iint_t = (tnum * (nint // tden) + nint) % nint

                xs = np.matmul(Sym.SpaceRot, all_pos[il, iint_s,:])
                xt = all_pos[il, iint_t,:]
                dx = xs - xt

                assert (np.linalg.norm(dx)) < eps
                
            # All positions at once
            tnum_target, tden_target = Sym.ApplyTSegm(iint_source, nint_min)
            assert nint_min % tden_target == 0
            iint_target = (tnum_target * (nint_min // tden_target) + nint_min) % nint_min   
                
            ibeg_target = iint_target * segm_size         
            iend_target = ibeg_target + segm_size
            
            # IMPORTANT !!!!
            if Sym.TimeRev == -1:
                ibeg_target += 1
                iend_target += 1
            
            pos_source_segm = all_pos[il,ibeg_source:iend_source,:]
            pos_target_segm = np.empty((segm_size,geodim),dtype=np.float64)

            Sym.TransformSegment(pos_source_segm, pos_target_segm)

            if iend_target <= nint:
               
                assert (np.linalg.norm(pos_target_segm - all_pos[il, ibeg_target:iend_target, :])) < eps
                
            else:
                
                assert iend_target == nint+1
                assert (np.linalg.norm(pos_target_segm[:segm_size-1,:] - all_pos[il, ibeg_target:iend_target-1, :])) < eps
                assert (np.linalg.norm(pos_target_segm[segm_size-1,:] - all_pos[il, 0, :])) < eps
                
                



# def ComputePeriodicitydefault(SegmGraph, bodysegm, nint_min, xo, xf):
#     
#     # xo belongs to the first segment
#     iint = 0
#     # xf
#     jint = (iint+1) % nint_min
#     
#     nbody = xo.shape[0]
#     geodim = xo.shape[1]
#     
#     for ib in range(nbody):
#     

# def PosSliceToAllPos()

# def PrepareParamBuf():



def BundleListOfShapes(ListOfShapes):
    
    n = len(ListOfShapes)
    
    ref_shp = ListOfShapes[0]
    ref_ndim = len(ref_shp)
    
    n_shapes = np.zeros((n, ref_ndim), dtype=np.intp)
    n_shifts = np.zeros((n+1), dtype=np.intp)
    
    for i, shp in enumerate(ListOfShapes):
        
        assert ref_ndim == len(shp)
        
        n_shapes[i,:] = shp
        n_shifts[i+1] = n_shifts[i] + math.prod(shp)
        
    return n_shapes, n_shifts

def BundleListOfArrays(ListOfArrays):
        
    ListOfShapes = [arr.shape for arr in ListOfArrays]
    n_shapes, n_shifts = BundleListOfShapes(ListOfShapes)

    ref_arr = ListOfArrays[0]

    buf = np.empty((n_shifts[-1]), dtype=ref_arr.dtype)
    
    for i, arr in enumerate(ListOfArrays):
        
        assert ref_arr.dtype == arr.dtype
        
        buf[n_shifts[i]:n_shifts[i+1]] = arr.reshape(-1)
        
    return buf, n_shapes, n_shifts


def Generating_to_interacting(SegmGraph, nbody, geodim, nsegm, intersegm_to_iint, intersegm_to_body, gensegm_to_iint, gensegm_to_body):
    
    AllSyms = []
    
    for isegm in range(nsegm):
    
        gensegm = (gensegm_to_body[isegm], gensegm_to_iint[isegm])
        intersegm = (intersegm_to_body[isegm], intersegm_to_iint[isegm])
        
        path = networkx.shortest_path(SegmGraph, source = gensegm, target = intersegm)
        
        SourceToTargetSym = ActionSym.Identity(nbody, geodim)
        
        for iedge in range(len(path)-1):
       
            if (path[iedge] > path[iedge+1]):

                edge = (path[iedge+1], path[iedge] )
                Sym = SegmGraph.edges[edge]["SymList"][0].Inverse()

            else:

                edge = (path[iedge] , path[iedge+1])
                Sym = SegmGraph.edges[edge]["SymList"][0]
        
            SourceToTargetSym = Sym.Compose(SourceToTargetSym)

        AllSyms.append(SourceToTargetSym)
        
    return AllSyms

def Compute_all_body_pos(all_pos, BodyGraph, loopgen, BodyLoop):
    
    nbody = BodyLoop.shape[0]
    nint = all_pos.shape[1]
    geodim = all_pos.shape[2]
    
    all_body_pos = np.zeros((nbody,nint,geodim), dtype=np.float64)

    for ib in range(nbody):
        
        il = BodyLoop[ib]
        ib_gen = loopgen[il]
        
        if (ib == ib_gen) :
            
            all_body_pos[ib,:,:] = all_pos[il,:,:]
            
        else:
        
            path = networkx.shortest_path(BodyGraph, source = ib_gen, target = ib)

            pathlen = len(path)

            TotSym = ActionSym.Identity(nbody, geodim)
            for ipath in range(1,pathlen):

                if (path[ipath-1] > path[ipath]):
                    Sym = BodyGraph.edges[(path[ipath], path[ipath-1])]["SymList"][0].Inverse()
                else:
                    Sym = BodyGraph.edges[(path[ipath-1], path[ipath])]["SymList"][0]

                TotSym = Sym.Compose(TotSym)
        
            for iint_gen in range(nint):
                
                tnum, tden = TotSym.ApplyT(iint_gen, nint)
                iint_target = tnum * nint // tden
                
                all_body_pos[ib,iint_target,:] = np.matmul(TotSym.SpaceRot, all_pos[il,iint_gen,:])

    return all_body_pos

def Populate_allsegmpos(all_pos, GenSpaceRot, GenTimeRev, gensegm_to_body, gensegm_to_iint, BodyLoop, nint_min):
    
    nsegm = gensegm_to_body.shape[0]
    nint = all_pos.shape[1]
    geodim = all_pos.shape[2]
    segm_size = nint // nint_min
    
    allsegmpos = np.empty((nsegm, segm_size, geodim), dtype=np.float64)

    for isegm in range(nsegm):

        ib = gensegm_to_body[isegm]
        iint = gensegm_to_iint[isegm]
        il = BodyLoop[ib]


        if GenTimeRev[isegm] == 1:
                
            ibeg = iint * segm_size         
            iend = ibeg + segm_size
            assert iend <= nint
            
            np.matmul(
                all_pos[il,ibeg:iend,:]     ,
                GenSpaceRot[isegm,:,:].T    ,
                out=allsegmpos[isegm,:,:]   ,
            )            

        else:

            ibeg = iint * segm_size + 1
            iend = ibeg + segm_size
            assert iend <= nint
            
            allsegmpos[isegm,:,:] = np.matmul(
                all_pos[il,ibeg:iend,:]     ,
                GenSpaceRot[isegm,:,:].T    ,
            )[::-1,:]
    

    return allsegmpos
    
def params_to_all_coeffs(
    params_buf, params_shapes, params_shifts,
    params_basis_reorganized_list, nnz_k_list, ncoeff_min_loop, nint
):
    
    assert nint % 2 == 0
    ncoeffs = nint//2 +1
    
    geodim = params_basis_reorganized_list[0].shape[0]
    nloop = ncoeff_min_loop.shape[0]
    
    all_coeffs = np.zeros((nloop, ncoeffs, geodim), dtype=np.complex128)
    
    for il in range(nloop):
        
        params_basis_reorganized = params_basis_reorganized_list[il]
        nparam_per_period_loop = params_basis_reorganized.shape[2]
        nnz_k = nnz_k_list[il]
        
        npr = (ncoeffs-1) //  ncoeff_min_loop[il]
        
        params_loop = params_buf[params_shifts[il]:params_shifts[il+1]].reshape(params_shapes[il])
        
        coeffs_reorganized = np.einsum('ijk,ljk->lji', params_basis_reorganized, params_loop)
        
        coeffs_dense = all_coeffs[il,:(ncoeffs-1),:].reshape(npr, ncoeff_min_loop[il], geodim)
        coeffs_dense[:,nnz_k,:] = coeffs_reorganized
        
    return all_coeffs
    

def setup_changevar_new(geodim, nbody, nint_init, mass, n_reconverge_it_max=6, MomCons=False, n_grad_change=1., Sym_list=[], CrashOnIdentity=True, ForceMatrixChangevar = False, store_folder = ""):
    
    r"""
    This function constructs a ChoreoAction
    It detects loops and constraints based on symmetries.
    It defines parameters according to given constraints and diagonal change of variable.
    It computes useful objects to optimize the computation of the action :
     - Exhaustive list of unary transformation for generator to body.
     - Exhaustive list of binary transformations from generator within each loop.
    """

    eps = 1e-12

    nint_min, nloop, loopnb, BodyLoop, Targets, BodyGraph = DetectLoops(Sym_list, nbody)
    
    SegmGraph, nint_min, nsegm, bodysegm, BodyHasContiguousGeneratingSegments, Sym_list = ExploreGlobalShifts_BuildSegmGraph(geodim, nbody, nloop, loopnb, Targets, nint_min, Sym_list)



    # Choose loop generators with maximal exploitable FFT symmetry
    loopgen = -np.ones((nloop), dtype = np.intp)
    for il in range(nloop):
        for ilb in range(loopnb[il]):

            if BodyHasContiguousGeneratingSegments[Targets[il,ilb]]:
                loopgen[il] = Targets[il,ilb]
                break

        assert loopgen[il] >= 0



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

    intersegm_to_body = np.zeros((nsegm), dtype = int)
    intersegm_to_iint = np.zeros((nsegm), dtype = int)

    assigned_segms = set()

    for iint in range(nint_min):
        for ib in range(nbody):

            isegm = bodysegm[ib,iint]

            if not(isegm in assigned_segms):
                
                intersegm_to_body[isegm] = ib
                intersegm_to_iint[isegm] = iint
                assigned_segms.add(isegm)

    # Checking that all segments occur in all intervals

    for iint in range(nint_min):        
        for isegm in range(nsegm):
            assert isegm in bodysegm[:,iint]



    # Choose generating segments as contiguous chunks of loop generators
    assigned_segms = set()

    gensegm_to_body = np.zeros((nsegm), dtype = int)
    gensegm_to_iint = np.zeros((nsegm), dtype = int)
    ngensegm_loop = np.zeros((nloop), dtype = int)

    for iint in range(nint_min):
        for il in range(nloop):
            ib = loopgen[il]

            isegm = bodysegm[ib,iint]

            if not(isegm in assigned_segms):
                gensegm_to_body[isegm] = ib
                gensegm_to_iint[isegm] = iint
                assigned_segms.add(isegm)
                ngensegm_loop[il] += 1
                


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

    InitValPosBasis = ComputeParamBasis_InitVal(nbody, geodim, InstConstraintsPos[0], mass, BodyLoop, MomCons=MomCons_InitVal)
    InitValVelBasis = ComputeParamBasis_InitVal(nbody, geodim, InstConstraintsVel[0], mass, BodyLoop, MomCons=MomCons_InitVal)
    
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
    GenToIntSyms = Generating_to_interacting(SegmGraph, nbody, geodim, nsegm, intersegm_to_iint, intersegm_to_body, gensegm_to_iint, gensegm_to_body)



    
    intersegm_to_all = AccumulateSegmGenToTargetSym(SegmGraph, nbody, geodim, nint_min, nsegm, bodysegm, intersegm_to_iint, intersegm_to_body)

    BinarySegm, Identity_detected = FindAllBinarySegments(intersegm_to_all, nbody, nsegm, nint_min, bodysegm, CrashOnIdentity, mass, BodyLoop)

    All_Id, count_tot, count_unique = CountSegmentBinaryInteractions(BinarySegm, nsegm)







    
    # print()
    # print('================================================')
    # print()

    # This could certainly be made more efficient
    BodyConstraints = AccumulateBodyConstraints(Sym_list, nbody, geodim)
    LoopGenConstraints = [BodyConstraints[ib]for ib in loopgen]
    
    
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
    
    
    
    
    

    All_params_basis = ComputeParamBasis_Loop(MomCons, nbody, nloop, loopgen, geodim, LoopGenConstraints)

 
    params_basis_reorganized_list, nnz_k_list = reorganize_All_params_basis(All_params_basis)

    ncoeff_min_loop = np.array([len(All_params_basis[il]) for il in range(nloop)], dtype=np.intp)
    ncoeff_min_loop_nnz = np.array([nnz_k_list[il].shape[0] for il in range(nloop)], dtype=np.intp)

    n_sub_fft = np.zeros((nloop), dtype=np.intp)
    for il in range(nloop):
        
        assert nint_min % ncoeff_min_loop[il] == 0
        assert (nint_min // ncoeff_min_loop[il]) % ngensegm_loop[il] == 0        
        assert (nint_min // (ncoeff_min_loop[il] * ngensegm_loop[il])) in [1,2]
        
        n_sub_fft[il] = (nint_min // (ncoeff_min_loop[il] * ngensegm_loop[il]))
        
    assert (n_sub_fft == n_sub_fft[0]).all()
    if n_sub_fft[0] == 1:
        nnpr = 2
    else:
        nnpr = 1

    print( f'{nnpr = }')
    
    

    nint = 2 * nint_min * 4
    ncoeffs = nint // 2 + 1
    segm_size = nint // nint_min

    
    
    
    # Without lists now.
    
    params_basis_buf, params_basis_shapes, params_basis_shifts = BundleListOfArrays(params_basis_reorganized_list)
    nnz_k_buf, nnz_k_shapes, nnz_k_shifts = BundleListOfArrays(nnz_k_list)
    
    params_shapes_list = []
    ifft_shapes_list = []
    pos_slice_shapes_list = []
    for il in range(nloop):

        nppl = params_basis_reorganized_list[il].shape[2]
        assert nint % (2*ncoeff_min_loop[il]) == 0
        npr = nint //  (2*ncoeff_min_loop[il])
        
        params_shapes_list.append((npr, ncoeff_min_loop_nnz[il], nppl))
        ifft_shapes_list.append((npr+1, ncoeff_min_loop_nnz[il], nppl))
        pos_slice_shapes_list.append((npr*nnpr, geodim))
        

    params_shapes, params_shifts = BundleListOfShapes(params_shapes_list)
    ifft_shapes, ifft_shifts = BundleListOfShapes(ifft_shapes_list)
    pos_slice_shapes, pos_slice_shifts = BundleListOfShapes(pos_slice_shapes_list)
        
        
    params_buf = np.random.random((params_shifts[-1]))
      
    pos_slice_buf = np.zeros((pos_slice_shifts[-1]),dtype=np.float64)    
            
    params_to_pos_slice(
        params_buf.copy()   , params_shapes         , params_shifts         ,
                              ifft_shapes           , ifft_shifts           ,
        params_basis_buf    , params_basis_shapes   , params_basis_shifts   ,
        nnz_k_buf           , nnz_k_shapes          , nnz_k_shifts          ,
        pos_slice_buf       , pos_slice_shapes      , pos_slice_shifts      ,
        ncoeff_min_loop     , nnpr  ,
    )
            
    all_coeffs = params_to_all_coeffs(
        params_buf, params_shapes, params_shifts,
        params_basis_reorganized_list, nnz_k_list, ncoeff_min_loop, nint
    )
        
    all_pos = scipy.fft.irfft(all_coeffs, axis=1)
 
        
    for il in range(nloop):
        
        n_inter = pos_slice_shapes[il,0]
        assert n_inter == (segm_size * ngensegm_loop[il])
        
        pos_slice = pos_slice_buf[pos_slice_shifts[il]:pos_slice_shifts[il+1]].reshape(pos_slice_shapes[il])

        
        assert np.linalg.norm(all_pos[il,:n_inter,:] - pos_slice) < 1e-14
        
        
        
        
        
        
        
        
        
        
        
        
        

        
        
    AssertAllSegmGenConstraintsAreRespected(gensegm_to_all, nint_min, bodysegm, loopgen, gensegm_to_body, gensegm_to_iint , BodyLoop, all_pos)
    AssertAllBodyConstraintAreRespected(LoopGenConstraints, all_pos)

    GenTimeRev = np.zeros((nsegm), dtype=np.intp)
    GenSpaceRot = np.zeros((nsegm, geodim, geodim), dtype=np.float64)

    for isegm in range(nsegm):

        ib = intersegm_to_body[isegm]
        iint = intersegm_to_iint[isegm]

        Sym = gensegm_to_all[ib][iint]
        
        GenTimeRev[isegm] = Sym.TimeRev
        GenSpaceRot[isegm,:,:] = Sym.SpaceRot
        
        # tnum_target, tden_target = Sym.ApplyTSegm(gensegm_to_iint[isegm], nint_min)
        # assert nint_min % tden_target == 0
        # assert intersegm_to_iint[isegm] == (tnum_target * (nint_min // tden_target) + nint_min) % nint_min  

    
    
    
    allsegmpos = Populate_allsegmpos(all_pos, GenSpaceRot, GenTimeRev, gensegm_to_body, gensegm_to_iint, BodyLoop, nint_min)
    
    all_body_pos = Compute_all_body_pos(all_pos, BodyGraph, loopgen, BodyLoop)
    
    # allsegmpos_cy = np.empty((nsegm, segm_size, geodim), dtype=np.float64)
    # Populate_allsegmpos_cy(all_pos, allsegmpos_cy, GenSpaceRot, GenTimeRev, gensegm_to_body, gensegm_to_iint, BodyLoop)
    # assert np.linalg.norm((allsegmpos_cy - allsegmpos)) < eps


    for isegm in range(nsegm):

        ib = intersegm_to_body[isegm]
        iint = intersegm_to_iint[isegm]
        il = BodyLoop[ib]
        
        # print(ib, loopgen[il])
        
#         if ib == loopgen[il]: # Since all_pos only containts LOOP positions!
# 
#             ibeg = iint * segm_size
#             iend = ibeg + segm_size
#             assert iend <= nint
#             assert np.linalg.norm(allsegmpos[isegm,:,:] - all_pos[il,ibeg:iend,:]) < eps
#             
#             Sym = gensegm_to_all[ib][iint]        
    

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























def Prepare_data_for_speed_comparison(
    geodim                  ,
    nbody                   ,
    mass                    ,
    n_reconverge_it_max     ,
    Sym_list                ,
    nint_fac                ,
):


    eps = 1e-12

    # nint_min, nloop, loopnb, BodyLoop, Targets = DetectLoops(Sym_list, nbody, nint_min_fac = 2)
    nint_min, nloop, loopnb, BodyLoop, Targets = DetectLoops(Sym_list, nbody)
    
    SegmGraph, nint_min, nsegm, bodysegm, BodyHasContiguousGeneratingSegments, Sym_list = ExploreGlobalShifts_BuildSegmGraph(geodim, nbody, nloop, loopnb, Targets, nint_min, Sym_list)



    # Choose loop generators with maximal exploitable FFT symmetry
    loopgen = -np.ones((nloop), dtype = np.intp)
    for il in range(nloop):
        for ilb in range(loopnb[il]):

            if BodyHasContiguousGeneratingSegments[Targets[il,ilb]]:
                loopgen[il] = Targets[il,ilb]
                break

        assert loopgen[il] >= 0



    # Accumulate constraints on segments. 

    SegmConstraints = AccumulateSegmentConstraints(SegmGraph, nbody, geodim, nsegm, bodysegm)






    # Choose interacting segments as earliest possible times.

    intersegm_to_body = np.zeros((nsegm), dtype = int)
    intersegm_to_iint = np.zeros((nsegm), dtype = int)

    assigned_segms = set()

    for iint in range(nint_min):
        for ib in range(nbody):

            isegm = bodysegm[ib,iint]

            if not(isegm in assigned_segms):
                
                intersegm_to_body[isegm] = ib
                intersegm_to_iint[isegm] = iint
                assigned_segms.add(isegm)

    # Checking that all segments occur in all intervals

    for iint in range(nint_min):        
        for isegm in range(nsegm):
            assert isegm in bodysegm[:,iint]



    # Choose generating segments as contiguous chunks of loop generators
    assigned_segms = set()

    gensegm_to_body = np.zeros((nsegm), dtype = int)
    gensegm_to_iint = np.zeros((nsegm), dtype = int)
    ngensegm_loop = np.zeros((nloop), dtype = int)

    for iint in range(nint_min):
        for il in range(nloop):
            ib = loopgen[il]

            isegm = bodysegm[ib,iint]

            if not(isegm in assigned_segms):
                gensegm_to_body[isegm] = ib
                gensegm_to_iint[isegm] = iint
                assigned_segms.add(isegm)
                ngensegm_loop[il] += 1






    
    gensegm_to_all = AccumulateSegmGenToTargetSym(SegmGraph, nbody, geodim, nint_min, nsegm, bodysegm, gensegm_to_iint, gensegm_to_body)
    
    GenToIntSyms = Generating_to_interacting(SegmGraph, nbody, geodim, nsegm, intersegm_to_iint, intersegm_to_body, gensegm_to_iint, gensegm_to_body)


    
    intersegm_to_all = AccumulateSegmGenToTargetSym(SegmGraph, nbody, geodim, nint_min, nsegm, bodysegm, intersegm_to_iint, intersegm_to_body)

    BinarySegm, Identity_detected = FindAllBinarySegments(intersegm_to_all, nbody, nsegm, nint_min, bodysegm, True, mass, BodyLoop)

    All_Id, count_tot, count_unique = CountSegmentBinaryInteractions(BinarySegm, nsegm)



    # This could certainly be made more efficient
    BodyConstraints = AccumulateBodyConstraints(Sym_list, nbody, geodim)
    LoopGenConstraints = [BodyConstraints[ib]for ib in loopgen]


    All_params_basis = ComputeParamBasis_Loop(False, nbody, nloop, loopgen, geodim, LoopGenConstraints)

    params_basis_reorganized_list, nnz_k_list = reorganize_All_params_basis(All_params_basis)

    ncoeff_min_loop = np.array([len(All_params_basis[il]) for il in range(nloop)], dtype=np.intp)
    ncoeff_min_loop_nnz = np.array([nnz_k_list[il].shape[0] for il in range(nloop)], dtype=np.intp)


    n_sub_fft = np.zeros((nloop), dtype=np.intp)
    for il in range(nloop):
        
        assert nint_min % ncoeff_min_loop[il] == 0
        assert (nint_min // ncoeff_min_loop[il]) % ngensegm_loop[il] == 0        
        assert (nint_min // (ncoeff_min_loop[il] * ngensegm_loop[il])) in [1,2]
        
        n_sub_fft[il] = (nint_min // (ncoeff_min_loop[il] * ngensegm_loop[il]))
        

    assert (n_sub_fft == n_sub_fft[0]).all()
    if n_sub_fft[0] == 1:
        nnpr = 2
    else:
        nnpr = 1







    nint = 2 * nint_min * nint_fac
    ncoeffs = nint // 2 + 1
    
    all_params_list = []
    # Create parameters
    for il in range(nloop):
        
        params_basis_reorganized = params_basis_reorganized_list[il]
        nparam_per_period_loop = params_basis_reorganized.shape[2]
        nnz_k = nnz_k_list[il]
        
        npr = (ncoeffs-1) //  ncoeff_min_loop[il]
        nperiods_loop = npr * ncoeff_min_loop_nnz[il]
        
        params_loop = np.random.random((npr, ncoeff_min_loop_nnz[il], nparam_per_period_loop))
        all_params_list.append(params_loop)
        
    return Pick_Named_Args_From_Dict(params_to_all_pos, dict(**locals()))
        
# TODO: To be removed of course
def Pick_Named_Args_From_Dict(fun, the_dict, MissingArgsAreNone=True):
    
    import inspect
    list_of_args = inspect.getfullargspec(fun).args
    
    if MissingArgsAreNone:
        all_kwargs = {k:the_dict.get(k) for k in list_of_args}
        
    else:
        all_kwargs = {k:the_dict[k] for k in list_of_args}
    
    return all_kwargs


def params_to_all_pos_mod(params_basis_reorganized_list, all_params_list, nnz_k_list, ncoeff_min_loop, ncoeffs, nnpr):
    
    nloop = len(params_basis_reorganized_list)
    geodim = params_basis_reorganized_list[0].shape[0]
    all_coeffs = np.zeros((nloop,ncoeffs,geodim), dtype=np.complex128)
    
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
        
    all_pos = scipy.fft.irfft(all_coeffs, axis=1)
        
def params_to_all_pos(params_basis_reorganized_list, all_params_list, nnz_k_list, ncoeff_min_loop, ncoeffs, nnpr):
    
    nloop = len(params_basis_reorganized_list)
    geodim = params_basis_reorganized_list[0].shape[0]
    all_coeffs = np.zeros((nloop,ncoeffs,geodim), dtype=np.complex128)
    
    for il in range(nloop):
        
        params_loop = all_params_list[il]
        
        params_basis_reorganized = params_basis_reorganized_list[il]
        geodim = params_basis_reorganized.shape[0]
        nnz_k = nnz_k_list[il]
        
        npr = (ncoeffs-1) //  ncoeff_min_loop[il]
        ncoeff_min_loop_nnz = len(nnz_k)

        coeffs_reorganized = np.einsum('ijk,ljk->lji', params_basis_reorganized, params_loop)
        
        coeffs_dense = np.zeros((npr, ncoeff_min_loop[il], geodim), dtype=np.complex128)
        coeffs_dense[:,nnz_k,:] = coeffs_reorganized
        all_coeffs[il,:(ncoeffs-1),:] = coeffs_dense.reshape(((ncoeffs-1), geodim))
        
    all_pos = scipy.fft.irfft(all_coeffs, axis=1)
                

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