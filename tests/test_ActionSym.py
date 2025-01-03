""" Tests properties of :class:`choreo.ActionSym`.

.. autosummary::
    :toctree: _generated/
    :template: tests-formatting/base.rst
    :nosignatures:

    test_Identity
    test_Random
    test_rotation_generation
    test_Cayley_graph

"""

import pytest
from .test_config import *
import numpy as np
import scipy
import networkx
import choreo

@ParametrizeDocstrings
@pytest.mark.parametrize("geodim", Physical_dims)
@pytest.mark.parametrize("nbody", Few_bodies)
def test_Identity(float64_tols, geodim, nbody):
    """ Tests properties of the identity transformation.
    """

    Id = choreo.ActionSym.Identity(nbody, geodim)

    assert Id.IsIdentity(atol = float64_tols.atol)
    assert Id.IsWellFormed(atol = float64_tols.atol)

    Id2 = Id.Compose(Id)

    assert Id2.IsIdentity(atol = float64_tols.atol)

    InvId = Id.Inverse()

    assert Id.IsSame(InvId, atol = float64_tols.atol)

@ParametrizeDocstrings
@RepeatTest()
@pytest.mark.parametrize("geodim", Physical_dims)
@pytest.mark.parametrize("nbody", Few_bodies)
def test_Random(float64_tols, geodim, nbody):
    """ Tests group properties on random transformations.
    """

    Id = choreo.ActionSym.Identity(nbody, geodim)

    A = choreo.ActionSym.Random(nbody, geodim)
    AInv = A.Inverse()

    assert A.IsWellFormed(atol = float64_tols.atol)
    assert AInv.IsWellFormed(atol = float64_tols.atol)

    assert Id.IsSame(A.Compose(AInv), atol = float64_tols.atol)
    assert Id.IsSame(AInv.Compose(A), atol = float64_tols.atol)

    B = choreo.ActionSym.Random(nbody, geodim)
    BInv = B.Inverse()

    assert not(A.IsSame(B, atol = float64_tols.atol))

    AB = A.Compose(B)
    BA = B.Compose(A)
    
    assert AB.IsWellFormed(atol = float64_tols.atol)
    assert BA.IsWellFormed(atol = float64_tols.atol)            

    n = AB.TimeShiftDen
    for i in range(n):

        tb = B.ApplyT(i, n)
        tab = A.ApplyT(*tb)

        assert tab == AB.ApplyT(i, n)
        
    ABInv = AB.Inverse()
    BAInv = BA.Inverse()
    
    assert ABInv.IsWellFormed(atol = float64_tols.atol)
    assert BAInv.IsWellFormed(atol = float64_tols.atol)            
    
    assert ABInv.IsSame(BInv.Compose(AInv), atol = float64_tols.atol)
    assert BAInv.IsSame(AInv.Compose(BInv), atol = float64_tols.atol)

    C = choreo.ActionSym.Random(nbody, geodim)

    A_BC = A.Compose(B.Compose(C))
    AB_C = A.Compose(B).Compose(C)
    
    assert A_BC.IsWellFormed(atol = float64_tols.atol)
    assert AB_C.IsWellFormed(atol = float64_tols.atol)           

    assert A_BC.IsSame(AB_C, atol = float64_tols.atol)

@ParametrizeDocstrings
@pytest.mark.parametrize("geodim", Physical_dims)
def test_rotation_generation(float64_tols, geodim):
    """ Tests parametrization of the orthogonal group.
    """

    n = (geodim * (geodim - 1)) // 2
    params = np.random.random(n)
    idmat = np.identity(geodim,dtype=np.float64)
    
    mat = choreo.ActionSym.SurjectiveDirectSpaceRot(params)
    matTmat = np.matmul(mat.T,mat  )
    matmatT = np.matmul(mat  ,mat.T)
    
    assert np.allclose(matTmat, idmat, rtol = float64_tols.rtol, atol = float64_tols.atol)  
    assert np.allclose(matmatT, idmat, rtol = float64_tols.rtol, atol = float64_tols.atol)     
    assert abs(scipy.linalg.det(mat) - 1.) < float64_tols.atol

@ParametrizeDocstrings
@pytest.mark.parametrize("Sym_list", [pytest.param(Sym_list, id=name) for name, Sym_list in SymList_dict.items()])
def test_Cayley_graph(Sym_list):
    """ Tests properties of Cayley graphs.
    
Tests:

* Strong connectedness
* Regularity 

    """

    nsym = len(Sym_list)
    
    if nsym < 1:
        return

    Sym = Sym_list[0]
    nbody = Sym.BodyPerm.shape[0]
    geodim = Sym.SpaceRot.shape[0]
    
    CayleyGraph = choreo.ActionSym.BuildCayleyGraph(nbody, geodim, GeneratorList = Sym_list)
    
    assert networkx.is_strongly_connected(CayleyGraph)

    assert  len(Sym_list) * CayleyGraph.number_of_nodes() == CayleyGraph.number_of_edges()
    
    for node in CayleyGraph:
        
        nneigh = 0
        for neigh in CayleyGraph.successors(node):
            nneigh += 1
        
        assert nneigh == nsym            
        
        nneigh = 0
        for neigh in CayleyGraph.predecessors(node):
            nneigh += 1
        
        assert nneigh == nsym
