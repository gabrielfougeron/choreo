
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
import pyquickbench

from choreo.cython._ActionSym import *
# from choreo.cython._ActionSym import AccumulateSegmGenToTargetSym

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
                    
def ComputeParamBasis_InitVal(nbody, geodim, InstConstraints, bodymass, MomCons=True, eps=1e-12):

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
                cstr_mat[idim, ib, idim] = bodymass[ib]

    cstr_mat = cstr_buf.reshape((-1, nbody*geodim))

    NullSpace = choreo.scipy_plus.linalg.null_space(cstr_mat)
    
    nparam = NullSpace.shape[1]
    NullSpace = NullSpace.reshape(nbody, geodim, nparam)

    for ib in range(nbody): 
        for idim in range(geodim):
            for iparam in range(nparam):

                if abs(NullSpace[ib, idim, iparam]) < eps:
                    NullSpace[ib, idim, iparam] = 0
       
    return np.ascontiguousarray(NullSpace)

def ComputeParamBasis_Loop(nbody, nloop, loopgen, geodim, LoopGenConstraints, eps=1e-12):

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

        ContainsIndirect = False
        for Sym in edge["SymList"]:

            if Sym.TimeRev == -1:
                ContainsIndirect = True
        
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


def AccumulateSegmGenToTargetSym(
        SegmGraph       ,
        nbody           , 
        geodim          , 
        nint_min        , 
        nsegm           , 
        bodysegm        , 
        segm_to_iint    ,  
        segm_to_body    ,
    ):

    segm_gen_to_target = [ [ None for iint in range(nint_min)] for ib in range(nbody) ]

    for isegm in range(nsegm):
        
        ib = segm_to_body[isegm]
        iint = segm_to_iint[isegm]
        
        segm_gen_to_target[ib][iint] = ActionSym.Identity(nbody, geodim)
        segm_source = (ib, iint)
        
        for edge in networkx.dfs_edges(SegmGraph, source=segm_source):
            
            Sym = segm_gen_to_target[edge[0][0]][edge[0][1]]
            
            assert Sym is not None
            
            if edge[0] > edge[1]:
                EdgeSym = SegmGraph.edges[edge]["SymList"][0].Inverse()
            else:
                EdgeSym = SegmGraph.edges[edge]["SymList"][0]


                
            segm_gen_to_target[edge[1][0]][edge[1][1]] = EdgeSym.Compose(Sym)
            
    return segm_gen_to_target                       

