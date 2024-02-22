import numpy as np
cimport numpy as np
np.import_array()
cimport cython

from libc.math cimport fabs as cfabs
from libc.math cimport sqrt as csqrt
from libc.complex cimport cexp

cimport scipy.linalg.cython_blas
from libc.stdlib cimport malloc, free
from libc.string cimport memset

from choreo.scipy_plus.cython.blas_consts cimport *

import choreo.scipy_plus.linalg

from choreo.NBodySyst_build import *

import scipy


@cython.final
cdef class NBodySyst():
    r"""
    This class defines a N-body system
    """
    
    cdef readonly long geodim
    cdef readonly long nbody
    cdef readonly long nint_min
    cdef readonly long nloop
    cdef readonly long nsegm
    cdef readonly long nnpr
    cdef readonly long nbin_segm_tot
    cdef readonly long nbin_segm_unique

    cdef readonly bint All_BinSegmTransformId

    cdef long[::1] _loopnb
    @property
    def loopnb(self):
        return np.asarray(self._loopnb)

    cdef long[::1] _bodyloop
    @property
    def bodyloop(self):
        return np.asarray(self._bodyloop)

    cdef double[::1] _loopmass
    @property
    def loopmass(self):
        return np.asarray(self._loopmass)

    cdef long[:,::1] _Targets
    @property
    def Targets(self):
        return np.asarray(self._Targets)

    cdef long[:,::1] _bodysegm
    @property
    def bodysegm(self):
        return np.asarray(self._bodysegm)

    cdef long[::1] _loopgen
    @property
    def loopgen(self):
        return np.asarray(self._loopgen)

    cdef long[::1] _intersegm_to_body
    @property
    def intersegm_to_body(self):
        return np.asarray(self._intersegm_to_body)

    cdef long[::1] _intersegm_to_iint
    @property
    def intersegm_to_iint(self):
        return np.asarray(self._intersegm_to_iint)

    cdef long[::1] _gensegm_to_body
    @property
    def gensegm_to_body(self):
        return np.asarray(self._gensegm_to_body)

    cdef long[::1] _gensegm_to_iint
    @property
    def gensegm_to_iint(self):
        return np.asarray(self._gensegm_to_iint)

    cdef long[::1] _ngensegm_loop
    @property
    def ngensegm_loop(self):
        return np.asarray(self._ngensegm_loop)

    cdef long[::1] _BinSourceSegm
    @property
    def BinSourceSegm(self):
        return np.asarray(self._BinSourceSegm)

    cdef long[::1] _BinTargetSegm
    @property
    def BinTargetSegm(self):
        return np.asarray(self._BinTargetSegm)
        
    cdef long[::1] _BinTimeRev
    @property
    def BinTimeRev(self):
        return np.asarray(self._BinTimeRev)
        
    cdef double[:,:,::1] _BinSpaceRot
    @property
    def BinSpaceRot(self):
        return np.asarray(self._BinSpaceRot)        

    cdef bint[::1] _BinSpaceRotIsId
    @property
    def BinSpaceRotIsId(self):
        return np.asarray(self._BinSpaceRotIsId) > 0
        
    cdef double[::1] _BinProdMassSum
    @property
    def BinProdMassSum(self):
        return np.asarray(self._BinProdMassSum)

    cdef long[::1] _InterTimeRev
    @property
    def InterTimeRev(self):
        return np.asarray(self._InterTimeRev)

    cdef bint[::1] _InterSpaceRotIsId
    cdef double[:,:,::1] _InterSpaceRot
    @property
    def InterSpaceRot(self):
        return np.asarray(self._InterSpaceRot)

    cdef double[:,:,::1] _InitValPosBasis
    @property
    def InitValPosBasis(self):
        return np.asarray(self._InitValPosBasis)

    cdef double[:,:,::1] _InitValVelBasis
    @property
    def InitValVelBasis(self):
        return np.asarray(self._InitValVelBasis)
        
    cdef double complex[::1] _params_basis_buf
    cdef long[:,::1] _params_basis_shapes
    cdef long[::1] _params_basis_shifts

    def params_basis(self, long il):
        return np.asarray(self._params_basis_buf[self._params_basis_shifts[il]:self._params_basis_shifts[il+1]]).reshape(self._params_basis_shapes[il])

    cdef long[::1] _nnz_k_buf
    cdef long[:,::1] _nnz_k_shapes
    cdef long[::1] _nnz_k_shifts

    def nnz_k(self, long il):
        return np.asarray(self._nnz_k_buf[self._nnz_k_shifts[il]:self._nnz_k_shifts[il+1]]).reshape(self._nnz_k_shapes[il])

    # Removal of imaginary part of c_o
    cdef bint[::1] _co_in_buf
    cdef long[:,::1] _co_in_shapes
    cdef long[::1] _co_in_shifts
    cdef long[::1] _ncor_loop
    cdef long[::1] _nco_in_loop
    cdef readonly long nrem

    def co_in(self, long il):
        return np.asarray(self._co_in_buf[self._co_in_shifts[il]:self._co_in_shifts[il+1]]).reshape(self._co_in_shapes[il]) > 0

    cdef long[::1] _ncoeff_min_loop
    @property
    def ncoeff_min_loop(self):
        return np.asarray(self._ncoeff_min_loop)

    cdef bint[::1] _BodyHasContiguousGeneratingSegments

    cdef readonly list Sym_list
    cdef readonly object BodyGraph
    cdef readonly object SegmGraph

    cdef readonly list gensegm_to_all
    cdef readonly list intersegm_to_all
    cdef readonly list LoopGenConstraints
    
    # Things that change with nint
    cdef long _nint
    @property
    def nint(self):
        return self._nint
    
    cdef long _nint_fac
    @property
    def nint_fac(self):
        return self._nint_fac
        
    cdef readonly long ncoeffs
    cdef readonly long segm_size    # number of interacting nodes in segment
    cdef readonly long segm_store   # number of stored values in segment, including repeated values for nnpr == 1
    cdef readonly long nparams
    cdef readonly long nparams_incl_o

    # WARNING: These are the shapes and shifts of POS params, not MOM params
    cdef long[:,::1] _params_shapes   
    cdef long[::1] _params_shifts

    cdef long[:,::1] _ifft_shapes      
    cdef long[::1] _ifft_shifts

    cdef long[:,::1] _pos_slice_shapes
    cdef long[::1] _pos_slice_shifts

    def __init__(
        self                        ,
        long geodim                 ,
        long nbody                  ,
        double[::1] bodymass        ,
        double[::1] bodycharge      ,
        list Sym_list               , 
        bint CrashOnIdentity = True ,
    ):

        cdef Py_ssize_t i, il, ibin, ib
        cdef double eps = 1e-12

        self._nint = -1 # Signals that things that scale with loop size are not set yet

        if (bodymass.shape[0] != nbody):
            raise ValueError(f'Incompatible number of bodies {nbody} vs number of masses {bodymass.shape[0]}')

        self.geodim = geodim
        self.nbody = nbody
        self.Sym_list = Sym_list
        for ib in range(nbody):
            assert bodymass[ib] != 0.

        self.DetectLoops(bodymass)

        self.ExploreGlobalShifts_BuildSegmGraph()

        self.ChooseLoopGen()

        # SegmConstraints = AccumulateSegmentConstraints(self.SegmGraph, nbody, geodim, self.nsegm, self._bodysegm)

        self.ChooseInterSegm()
        self.ChooseGenSegm()


        # Setting up forward ODE:
        # - What are my parameters ?
        # - Integration end + Lack of periodicity
        # - Constraints on initial values => Parametrization 

        
        InstConstraintsPos = AccumulateInstConstraints(self.Sym_list, nbody, geodim, self.nint_min, VelSym=False)
        InstConstraintsVel = AccumulateInstConstraints(self.Sym_list, nbody, geodim, self.nint_min, VelSym=True )

        self._InitValPosBasis = ComputeParamBasis_InitVal(nbody, geodim, InstConstraintsPos[0], bodymass, MomCons=True)
        self._InitValVelBasis = ComputeParamBasis_InitVal(nbody, geodim, InstConstraintsVel[0], bodymass, MomCons=True)


        self.gensegm_to_all = AccumulateSegmGenToTargetSym(self.SegmGraph, nbody, geodim, self.nint_min, self.nsegm, self._bodysegm, self._gensegm_to_iint, self._gensegm_to_body)

        # GenToIntSyms = Generating_to_interacting(self.SegmGraph, nbody, geodim, self.nsegm, self._intersegm_to_iint, self._intersegm_to_body, self._gensegm_to_iint, self._gensegm_to_body)

        self.intersegm_to_all = AccumulateSegmGenToTargetSym(self.SegmGraph, nbody, geodim, self.nint_min, self.nsegm, self._bodysegm, self._intersegm_to_iint, self._intersegm_to_body)

        self.GatherInterSym()

        BinarySegm, Identity_detected = FindAllBinarySegments(self.intersegm_to_all, nbody, self.nsegm, self.nint_min, self._bodysegm, CrashOnIdentity, bodymass)
        self.All_BinSegmTransformId, self.nbin_segm_tot, self.nbin_segm_unique = CountSegmentBinaryInteractions(BinarySegm, self.nsegm)

        self._BinSourceSegm, self._BinTargetSegm, self._BinTimeRev, self._BinSpaceRot, self._BinProdMassSum = ReorganizeBinarySegments(BinarySegm)
        assert self._BinSourceSegm.shape[0] == self.nbin_segm_unique
        self._BinSpaceRotIsId = np.zeros((self.nbin_segm_unique), dtype=np.intc)
        for ibin in range(self.nbin_segm_unique):
            self._BinSpaceRotIsId[ibin] = (np.linalg.norm(self._BinSpaceRot[ibin,:,:] - np.identity(self.geodim)) < eps)

        # This could certainly be made more efficient
        BodyConstraints = AccumulateBodyConstraints(self.Sym_list, nbody, geodim)
        self.LoopGenConstraints = [BodyConstraints[ib] for ib in self._loopgen]

        # Idem, but I'm too lazy to change it and it is not performance critical
        All_params_basis = ComputeParamBasis_Loop(nbody, self.nloop, self._loopgen, geodim, self.LoopGenConstraints)
        params_basis_reorganized_list, nnz_k_list, co_in_list = reorganize_All_params_basis(All_params_basis)
        
        self._params_basis_buf, self._params_basis_shapes, self._params_basis_shifts = BundleListOfArrays(params_basis_reorganized_list)

        self._nnz_k_buf, self._nnz_k_shapes, self._nnz_k_shifts = BundleListOfArrays(nnz_k_list)
        self._co_in_buf, self._co_in_shapes, self._co_in_shifts = BundleListOfArrays(co_in_list)

        self._nco_in_loop = np.zeros((self.nloop), dtype=np.intp)
        self._ncor_loop = np.zeros((self.nloop), dtype=np.intp)
        for il in range(self.nloop):
            for i in range(self._co_in_shifts[il],self._co_in_shifts[il+1]):
                if self._co_in_buf[i]:
                    self._ncor_loop[il] +=1
                else:
                    self._nco_in_loop[il] +=1

        self.nrem = np.sum(self._nco_in_loop)

        self._ncoeff_min_loop = np.array([len(All_params_basis[il]) for il in range(self.nloop)], dtype=np.intp)

        self.Compute_nnpr()

        # I'd rather do this twice than leave __init__ in a partially initialized state
        self.nint_fac = 1


    def DetectLoops(self, bodymass, nint_min_fac = 1):

        All_den_list_on_entry = []
        for Sym in self.Sym_list:
            All_den_list_on_entry.append(Sym.TimeShiftDen)

        self.nint_min = nint_min_fac * math.lcm(*All_den_list_on_entry) # ensures that all integer divisions will have zero remainder
        
        BodyGraph = Build_BodyGraph(self.nbody, self.Sym_list)

        self.nloop = sum(1 for _ in networkx.connected_components(BodyGraph))
        
        loopnb = np.zeros((self.nloop), dtype = np.intp)
        self._loopnb = loopnb

        for il, CC in enumerate(networkx.connected_components(BodyGraph)):
            loopnb[il] = len(CC)

        maxlooplen = loopnb.max()
        
        BodyLoop = np.zeros((self.nbody), dtype = np.intp)
        self._bodyloop = BodyLoop
        Targets = np.zeros((self.nloop, maxlooplen), dtype=np.intp)
        self._Targets = Targets
        for il, CC in enumerate(networkx.connected_components(BodyGraph)):
            for ilb, ib in enumerate(CC):
                Targets[il,ilb] = ib
                BodyLoop[ib] = il

        loopmass = np.zeros((self.nloop), dtype=np.float64)
        self._loopmass = loopmass
        for il in range(self.nloop):
            loopmass[il] = bodymass[Targets[il,0]]
            for ilb in range(loopnb[il]):
                ib = Targets[il,ilb]
                assert loopmass[il] == bodymass[ib]

        self.BodyGraph = BodyGraph


    def ExploreGlobalShifts_BuildSegmGraph(self):

        cdef Py_ssize_t ib

        # Making sure nint_min is big enough
        self.SegmGraph, self.nint_min = Build_SegmGraph_NoPb(self.nbody, self.nint_min, self.Sym_list)
        
        for i_shift in range(self.nint_min):
            
            if i_shift != 0:
                
                GlobalTimeShift = ActionSym(
                    BodyPerm  = np.array(range(self.nbody), dtype=np.intp)  ,
                    SpaceRot  = np.identity(self.geodim, dtype=np.float64)  ,
                    TimeRev   = 1                                           ,
                    TimeShiftNum = i_shift                                  ,
                    TimeShiftDen = self.nint_min                            ,
                )
                
                Shifted_sym_list = []
                for Sym in self.Sym_list:
                    Shifted_sym_list.append(Sym.Conjugate(GlobalTimeShift))
                self.Sym_list = Shifted_sym_list
            
                self.SegmGraph = Build_SegmGraph(self.nbody, self.nint_min, self.Sym_list)

            bodysegm = np.zeros((self.nbody, self.nint_min), dtype = np.intp)
            self._bodysegm = bodysegm
            for isegm, CC in enumerate(networkx.connected_components(self.SegmGraph)):
                for ib, iint in CC:
                    bodysegm[ib, iint] = isegm

            self.nsegm = isegm + 1
            
            bodynsegm = np.zeros((self.nbody), dtype = int)
            
            BodyHasContiguousGeneratingSegments = np.zeros((self.nbody), dtype = np.intc) 
            self._BodyHasContiguousGeneratingSegments = BodyHasContiguousGeneratingSegments

            for ib in range(self.nbody):

                unique, unique_indices, unique_inverse, unique_counts = np.unique(bodysegm[ib, :], return_index = True, return_inverse = True, return_counts = True)

                assert (unique == bodysegm[ib, unique_indices]).all()
                assert (unique[unique_inverse] == bodysegm[ib, :]).all()

                bodynsegm[ib] = unique.size
                self._BodyHasContiguousGeneratingSegments[ib] = ((unique_indices.max()+1) == bodynsegm[ib])
                
            AllLoopsHaveContiguousGeneratingSegments = True
            for il in range(self.nloop):
                LoopHasContiguousGeneratingSegments = False
                for ilb in range(self._loopnb[il]):
                    LoopHasContiguousGeneratingSegments = LoopHasContiguousGeneratingSegments or self._BodyHasContiguousGeneratingSegments[self._Targets[il,ilb]]

                AllLoopsHaveContiguousGeneratingSegments = AllLoopsHaveContiguousGeneratingSegments and LoopHasContiguousGeneratingSegments
            
            if AllLoopsHaveContiguousGeneratingSegments:
                break
        
        else:
            
            raise ValueError("Could not find time shift such that all loops have contiguous generating segments")

        # print(f"Required {i_shift} shifts to find reference such that all loops have contiguous generating segments")

    def ChooseLoopGen(self):
        
        # Choose loop generators with maximal exploitable FFT symmetry
        loopgen = -np.ones((self.nloop), dtype = np.intp)
        self._loopgen = loopgen
        for il in range(self.nloop):
            for ilb in range(self._loopnb[il]):

                if self._BodyHasContiguousGeneratingSegments[self._Targets[il,ilb]]:
                    loopgen[il] = self._Targets[il,ilb]
                    break

            assert loopgen[il] >= 0    


    def ChooseInterSegm(self):

        # Choose interacting segments as earliest possible times.

        intersegm_to_body = np.zeros((self.nsegm), dtype = np.intp)
        intersegm_to_iint = np.zeros((self.nsegm), dtype = np.intp)

        self._intersegm_to_body = intersegm_to_body
        self._intersegm_to_iint = intersegm_to_iint

        assigned_segms = set()

        for iint in range(self.nint_min):
            for ib in range(self.nbody):

                isegm = self._bodysegm[ib,iint]

                if not(isegm in assigned_segms):
                    
                    intersegm_to_body[isegm] = ib
                    intersegm_to_iint[isegm] = iint
                    assigned_segms.add(isegm)

    def ChooseGenSegm(self):
        
        assigned_segms = set()

        gensegm_to_body = np.zeros((self.nsegm), dtype = np.intp)
        gensegm_to_iint = np.zeros((self.nsegm), dtype = np.intp)
        ngensegm_loop = np.zeros((self.nloop), dtype = np.intp)

        self._gensegm_to_body = gensegm_to_body
        self._gensegm_to_iint = gensegm_to_iint
        self._ngensegm_loop = ngensegm_loop

        for iint in range(self.nint_min):
            for il in range(self.nloop):
                ib = self._loopgen[il]

                isegm = self._bodysegm[ib,iint]

                if not(isegm in assigned_segms):
                    gensegm_to_body[isegm] = ib
                    gensegm_to_iint[isegm] = iint
                    assigned_segms.add(isegm)
                    ngensegm_loop[il] += 1


    def GatherInterSym(self):
        
        InterTimeRev = np.zeros((self.nsegm), dtype=np.intp)
        InterSpaceRot = np.zeros((self.nsegm, self.geodim, self.geodim), dtype=np.float64)
        InterSpaceRotIsId = np.zeros((self.nsegm), dtype=np.intc)

        self._InterTimeRev = InterTimeRev
        self._InterSpaceRot = InterSpaceRot
        self._InterSpaceRotIsId = InterSpaceRotIsId

        for isegm in range(self.nsegm):

            ib = self._intersegm_to_body[isegm]
            iint = self._intersegm_to_iint[isegm]
            Sym = self.intersegm_to_all[ib][iint]
            
            InterTimeRev[isegm] = Sym.TimeRev
            InterSpaceRot[isegm,:,:] = Sym.SpaceRot

            assert InterTimeRev[isegm] == 1
    
            self._InterSpaceRotIsId[isegm] = Sym.IsIdentityRot()

    def Compute_nnpr(self):
        
        n_sub_fft = np.zeros((self.nloop), dtype=np.intp)
        for il in range(self.nloop):
            
            assert  self.nint_min % self._ncoeff_min_loop[il] == 0
            assert (self.nint_min // self._ncoeff_min_loop[il]) % self._ngensegm_loop[il] == 0        
            assert (self.nint_min // (self._ncoeff_min_loop[il] * self._ngensegm_loop[il])) in [1,2]
            
            n_sub_fft[il] = (self.nint_min // (self._ncoeff_min_loop[il] * self._ngensegm_loop[il]))
            
        assert (n_sub_fft == n_sub_fft[0]).all()
        
        if n_sub_fft[0] == 1:
            self.nnpr = 2
        else:
            self.nnpr = 1
            

    @nint_fac.setter
    @cython.final
    def nint_fac(self, long nint_fac_in):
        self.nint = 2 * self.nint_min * nint_fac_in
    
    @nint.setter
    @cython.cdivision(True)
    @cython.final
    def nint(self, long nint_in):

        if (nint_in % (2 * self.nint_min)) != 0:
            raise ValueError(f"Provided nint {nint_in} should be divisible by {2 * self.nint_min}")

        self._nint = nint_in
        self._nint_fac = nint_in // (2 * self.nint_min)
        self.ncoeffs = self._nint // 2 + 1
        self.segm_size = self._nint // self.nint_min

        if self.nnpr == 1:
            self.segm_store = self.segm_size + 1
        else:
            self.segm_store = self.segm_size

        params_shapes_list = []
        ifft_shapes_list = []
        pos_slice_shapes_list = []
        for il in range(self.nloop):

            nppl = self._params_basis_shapes[il,2]
            assert self._nint % (2*self._ncoeff_min_loop[il]) == 0
            npr = self._nint //  (2*self._ncoeff_min_loop[il])
            
            params_shapes_list.append((npr, self._nnz_k_shapes[il,0], nppl))
            ifft_shapes_list.append((npr+1, self._nnz_k_shapes[il,0], nppl))
            
            if self.nnpr == 1:
                ninter = npr+1
            elif self.nnpr == 2:
                ninter = 2*npr
            else:
                raise ValueError(f'Impossible value for {nnpr = }')
            
            pos_slice_shapes_list.append((ninter, self.geodim))
            
        self._params_shapes, self._params_shifts = BundleListOfShapes(params_shapes_list)
        self._ifft_shapes, self._ifft_shifts = BundleListOfShapes(ifft_shapes_list)
        self._pos_slice_shapes, self._pos_slice_shifts = BundleListOfShapes(pos_slice_shapes_list)

        self.nparams_incl_o = self._params_shifts[self.nloop]
        self.nparams = self._params_shifts[self.nloop] - self.nrem

    @cython.final
    def AssertAllSegmGenConstraintsAreRespected(self, all_pos, eps=1e-12):

        for il in range(self.nloop):
            
            ib = self._loopgen[il] # because only loops have been computed in all_pos so far.
            
            for iint in range(self.nint_min):
                
                isegm = self._bodysegm[ib, iint]
                
                Sym = self.gensegm_to_all[ib][iint]
                
                ib_source = self._gensegm_to_body[isegm]
                iint_source = self._gensegm_to_iint[isegm]
                
                il_source = self._bodyloop[ib_source]
                assert il_source == il
                
                self.segm_size = self._nint // self.nint_min

                ibeg_source = iint_source * self.segm_size          
                iend_source = ibeg_source + self.segm_size
                assert iend_source <= self._nint
                
                # One position at a time
                for iint_s in range(ibeg_source, iend_source):
                    
                    tnum, tden = Sym.ApplyT(iint_s, self._nint)
                    
                    assert self._nint % tden == 0
                    iint_t = (tnum * (self._nint // tden) + self._nint) % self._nint

                    xs = np.matmul(Sym.SpaceRot, all_pos[il, iint_s,:])
                    xt = all_pos[il, iint_t,:]
                    dx = xs - xt

                    assert (np.linalg.norm(dx)) < eps
                    
                # All positions at once
                tnum_target, tden_target = Sym.ApplyTSegm(iint_source, self.nint_min)
                assert self.nint_min % tden_target == 0
                iint_target = (tnum_target * (self.nint_min // tden_target) + self.nint_min) % self.nint_min   
                    
                ibeg_target = iint_target * self.segm_size         
                iend_target = ibeg_target + self.segm_size
                
                # IMPORTANT !!!!
                if Sym.TimeRev == -1:
                    ibeg_target += 1
                    iend_target += 1
                
                pos_source_segm = all_pos[il,ibeg_source:iend_source,:]
                pos_target_segm = np.empty((self.segm_size, self.geodim), dtype=np.float64)

                Sym.TransformSegment(pos_source_segm, pos_target_segm)

                if iend_target <= self._nint:
                
                    assert (np.linalg.norm(pos_target_segm - all_pos[il, ibeg_target:iend_target, :])) < eps
                    
                else:
                    
                    assert iend_target == self._nint+1
                    assert (np.linalg.norm(pos_target_segm[:self.segm_size-1,:] - all_pos[il, ibeg_target:iend_target-1, :])) < eps
                    assert (np.linalg.norm(pos_target_segm[ self.segm_size-1,:] - all_pos[il, 0, :])) < eps
            
    @cython.final
    def AssertAllBodyConstraintAreRespected(self, all_pos, eps=1e-12):
        # Make sure loop constraints are respected
        
        for il, Constraints in enumerate(self.LoopGenConstraints):

            for icstr, Sym in enumerate(Constraints):

                assert (self._nint % Sym.TimeShiftDen) == 0

                ConstraintIsRespected = True

                for iint in range(self._nint):

                    tnum, tden = Sym.ApplyT(iint, self._nint)
                    jint = tnum * self._nint // tden
                    
                    err = np.linalg.norm(all_pos[il,jint,:] - np.matmul(Sym.SpaceRot, all_pos[il,iint,:]))

                    assert (err < eps)


    @cython.final
    def params_changevar(self, double[::1] params_buf_in, bint inv=False, bint transpose=False):

        cdef double[::1] params_buf_out
        
        if inv:

            if transpose:

                assert params_buf_in.shape[0] == self.nparams
                params_buf_out = np.empty((self.nparams_incl_o), dtype=np.float64)
            
                changevar_mom_invT(
                    &params_buf_in[0]       , self._params_shapes   , self._params_shifts   ,
                    self._nnz_k_buf         , self._nnz_k_shapes    , self._nnz_k_shifts    ,
                    self._co_in_buf         , self._co_in_shapes    , self._co_in_shifts    ,
                    self._ncoeff_min_loop   ,
                    self._loopnb            , self._loopmass        ,
                    &params_buf_out[0]      , 
                )   

            else:

                assert params_buf_in.shape[0] == self.nparams_incl_o
                params_buf_out = np.empty((self.nparams), dtype=np.float64)

                changevar_mom_inv(
                    &params_buf_in[0]       , self._params_shapes   , self._params_shifts   ,
                    self._nnz_k_buf         , self._nnz_k_shapes    , self._nnz_k_shifts    ,
                    self._co_in_buf         , self._co_in_shapes    , self._co_in_shifts    ,
                    self._ncoeff_min_loop   ,
                    self._loopnb            , self._loopmass        ,
                    &params_buf_out[0]      , 
                )   

        else:

            if transpose:

                assert params_buf_in.shape[0] == self.nparams_incl_o
                params_buf_out = np.empty((self.nparams), dtype=np.float64)

                changevar_mom_T(
                    &params_buf_in[0]       , self._params_shapes   , self._params_shifts   ,
                    self._nnz_k_buf         , self._nnz_k_shapes    , self._nnz_k_shifts    ,
                    self._co_in_buf         , self._co_in_shapes    , self._co_in_shifts    ,
                    self._ncoeff_min_loop   ,
                    self._loopnb            , self._loopmass        ,
                    &params_buf_out[0]      , 
                )   

            else:

                assert params_buf_in.shape[0] == self.nparams
                params_buf_out = np.empty((self.nparams_incl_o), dtype=np.float64)

                changevar_mom(
                    &params_buf_in[0]       , self._params_shapes   , self._params_shifts   ,
                    self._nnz_k_buf         , self._nnz_k_shapes    , self._nnz_k_shifts    ,
                    self._co_in_buf         , self._co_in_shapes    , self._co_in_shifts    ,
                    self._ncoeff_min_loop   ,
                    self._loopnb            , self._loopmass        ,
                    &params_buf_out[0]      , 
                )   

        return np.asarray(params_buf_out)

    @cython.final
    def all_coeffs_to_kin_nrg(self, double complex[:,:,::1] all_coeffs):

        cdef double kin = 0.
        cdef double fac, a
        cdef Py_ssize_t il, k, k2
        cdef int n = 2*self.geodim
        cdef double  *loc

        for il in range(self.nloop):
            
            fac = ctwopisq * self._loopmass[il] * self._loopnb[il]

            for k in range(1,self.ncoeffs-1):

                k2 = k*k
                a = fac*k2
                
                loc = <double*> &all_coeffs[il,k,0]

                kin += a * scipy.linalg.cython_blas.ddot(&n, loc, &int_one, loc, &int_one)

        return kin

    @cython.final
    def all_coeffs_to_kin_nrg_grad(self, double complex[:,:,::1] all_coeffs):

        cdef double complex [:,:,::1] kin_grad = np.zeros((self.nloop, self.ncoeffs, self.geodim), dtype=np.complex128)
        cdef double fac, a
        cdef Py_ssize_t il, k, k2
        cdef int n = 2*self.geodim
        cdef double *loc
        cdef double *grad_loc

        for il in range(self.nloop):
            
            fac = cfourpisq * self._loopmass[il] * self._loopnb[il]

            for k in range(1,self.ncoeffs-1):

                k2 = k*k
                a = fac*k2
                
                loc = <double*> &all_coeffs[il,k,0]
                grad_loc = <double*> &kin_grad[il,k,0]

                scipy.linalg.cython_blas.daxpy(&n, &a, loc, &int_one, grad_loc, &int_one)

        return np.asarray(kin_grad)

    @cython.final
    def params_to_kin_nrg(self, double[::1] params_mom_buf):

        return params_to_kin_nrg(
            &params_mom_buf[0]  , self._params_shapes   , self._params_shifts   ,
            self._ncor_loop     , self._nco_in_loop     ,
        )
        
    @cython.final
    def params_to_kin_nrg_grad(self, double[::1] params_mom_buf):

        cdef double[::1] grad_buf = np.empty((self.nparams), dtype=np.float64)

        params_to_kin_nrg_grad(
            &params_mom_buf[0]  , self._params_shapes   , self._params_shifts   ,
            self._ncor_loop     , self._nco_in_loop     ,
            &grad_buf[0]        ,
        )

        return np.asarray(grad_buf)

    @cython.final
    def params_to_all_coeffs_noopt(self, double[::1] params_mom_buf, bint transpose=False):

        assert params_mom_buf.shape[0] == self.nparams

        cdef np.ndarray[double, ndim=1, mode='c'] params_pos_buf = np.empty((self.nparams_incl_o), dtype=np.float64)

        if transpose:
            changevar_mom_invT(
                &params_mom_buf[0]      , self._params_shapes   , self._params_shifts   ,
                self._nnz_k_buf         , self._nnz_k_shapes    , self._nnz_k_shifts    ,
                self._co_in_buf         , self._co_in_shapes    , self._co_in_shifts    ,
                self._ncoeff_min_loop   ,
                self._loopnb            , self._loopmass        ,
                &params_pos_buf[0]      , 
            )   
        else:
            changevar_mom(
                &params_mom_buf[0]      , self._params_shapes   , self._params_shifts   ,
                self._nnz_k_buf         , self._nnz_k_shapes    , self._nnz_k_shifts    ,
                self._co_in_buf         , self._co_in_shapes    , self._co_in_shifts    ,
                self._ncoeff_min_loop   ,
                self._loopnb            , self._loopmass        ,
                &params_pos_buf[0]      , 
            )   

        all_coeffs = np.zeros((self.nloop, self.ncoeffs, self.geodim), dtype=np.complex128)
        
        cdef Py_ssize_t il

        for il in range(self.nloop):
            
            params_basis = self.params_basis(il)
            nnz_k = self.nnz_k(il)
            
            npr = (self.ncoeffs-1) //  self._ncoeff_min_loop[il]
            
            params_loop = params_pos_buf[self._params_shifts[il]:self._params_shifts[il+1]].reshape(self._params_shapes[il])

            coeffs_dense = all_coeffs[il,:(self.ncoeffs-1),:].reshape(npr, self._ncoeff_min_loop[il], self.geodim)                
            coeffs_dense[:,nnz_k,:] = np.einsum('ijk,ljk->lji', params_basis, params_loop)
            
        all_coeffs[:,0,:].imag = 0

        return all_coeffs    

    @cython.final
    def all_coeffs_to_params_noopt(self, all_coeffs, bint transpose=False):

        assert all_coeffs.shape[0] == self.nloop
        assert all_coeffs.shape[1] == self.ncoeffs
        assert all_coeffs.shape[2] == self.geodim

        params_pos_buf_np = np.empty((self.nparams_incl_o), dtype=np.float64)
        cdef double[::1] params_pos_buf = params_pos_buf_np

        cdef Py_ssize_t npr, il

        for il in range(self.nloop):

            params_basis = self.params_basis(il)
            nnz_k = self.nnz_k(il)

            npr = (self.ncoeffs-1) //  self._ncoeff_min_loop[il]

            coeffs_dense = all_coeffs[il,:(self.ncoeffs-1),:].reshape(npr, self._ncoeff_min_loop[il], self.geodim)                

            params_loop = params_pos_buf_np[self._params_shifts[il]:self._params_shifts[il+1]].reshape(self._params_shapes[il])

            params_loop[:] = np.einsum('ijk,lji->ljk', params_basis.conj(), coeffs_dense[:,nnz_k,:]).real

        params_mom_buf_np = np.empty((self.nparams), dtype=np.float64)
        cdef double[::1] params_mom_buf = params_mom_buf_np

        if transpose:
            changevar_mom_T(
                &params_pos_buf[0]      , self._params_shapes   , self._params_shifts   ,
                self._nnz_k_buf         , self._nnz_k_shapes    , self._nnz_k_shifts    ,
                self._co_in_buf         , self._co_in_shapes    , self._co_in_shifts    ,
                self._ncoeff_min_loop   ,
                self._loopnb            , self._loopmass        ,
                &params_mom_buf[0]      , 
            )   
        else:
            changevar_mom_inv(
                &params_pos_buf[0]      , self._params_shapes   , self._params_shifts   ,
                self._nnz_k_buf         , self._nnz_k_shapes    , self._nnz_k_shifts    ,
                self._co_in_buf         , self._co_in_shapes    , self._co_in_shifts    ,
                self._ncoeff_min_loop   ,
                self._loopnb            , self._loopmass        ,
                &params_mom_buf[0]      , 
            )   

        return params_mom_buf_np   

    @cython.final
    def all_pos_to_all_body_pos_noopt(self, all_pos):

        assert all_pos.shape[0] == self._nint
        
        all_body_pos = np.zeros((self.nbody, self._nint, self.geodim), dtype=np.float64)

        for ib in range(self.nbody):
            
            il = self._bodyloop[ib]
            ib_gen = self._loopgen[il]
            
            if (ib == ib_gen) :
                
                all_body_pos[ib,:,:] = all_pos[il,:,:]
                
            else:
            
                path = networkx.shortest_path(self.BodyGraph, source = ib_gen, target = ib)

                pathlen = len(path)

                TotSym = ActionSym.Identity(self.nbody, self.geodim)
                for ipath in range(1,pathlen):

                    if (path[ipath-1] > path[ipath]):
                        Sym = self.BodyGraph.edges[(path[ipath], path[ipath-1])]["SymList"][0].Inverse()
                    else:
                        Sym = self.BodyGraph.edges[(path[ipath-1], path[ipath])]["SymList"][0]

                    TotSym = Sym.Compose(TotSym)
            
                for iint_gen in range(nint):
                    
                    tnum, tden = TotSym.ApplyT(iint_gen, nint)
                    iint_target = tnum * nint // tden
                    
                    all_body_pos[ib,iint_target,:] = np.matmul(TotSym.SpaceRot, all_pos[il,iint_gen,:])

        return all_body_pos     

    @cython.final
    def all_pos_to_segmpos_noopt(self, all_pos):
        
        assert self._nint == all_pos.shape[1]
        
        cdef Py_ssize_t ib, iint, il

        allsegmpos = np.empty((self.nsegm, self.segm_store, self.geodim), dtype=np.float64)

        for isegm in range(self.nsegm):

            ib = self._gensegm_to_body[isegm]
            iint = self._gensegm_to_iint[isegm]
            il = self._bodyloop[ib]

            assert isegm == self._bodysegm[ib,iint]

            ibeg = iint * self.segm_size         
            iend = ibeg + self.segm_store
            assert iend <= self._nint

            if self._InterTimeRev[isegm] == 1:

                np.matmul(
                    all_pos[il,ibeg:iend,:]                     ,
                    np.asarray(self._InterSpaceRot[isegm,:,:]).T  ,
                    out = allsegmpos[isegm,:,:]                 ,
                )            

            else:

                allsegmpos[isegm,:,:] = np.matmul(
                    all_pos[il,ibeg:iend,:]                     ,
                    np.asarray(self._InterSpaceRot[isegm,:,:]).T  ,
                )[::-1,:]

        return allsegmpos

    @cython.final
    def segmpos_to_all_pos_noopt(self, allsegmpos):

        assert self.segm_store == allsegmpos.shape[1]

        cdef Py_ssize_t ib, iint, il

        all_pos = np.empty((self.nloop, self._nint, self.geodim), dtype=np.float64)

        for il in range(self.nloop):

            ib = self._loopgen[il]

            for iint in range(self.nint_min):

                Sym = self.gensegm_to_all[ib][iint]
                isegm = self._bodysegm[ib, iint]

                assert il == self._bodyloop[self._intersegm_to_body[isegm]]

                if Sym.TimeRev == 1:

                    ibeg = iint * self.segm_size         
                    iend = ibeg + self.segm_size
                    assert iend <= self._nint

                    np.matmul(
                        allsegmpos[isegm,:self.segm_size,:]           ,
                        Sym.SpaceRot.T                  ,
                        out = all_pos[il,ibeg:iend,:]   ,
                    )            

                else:

                    ibeg = iint * self.segm_size         
                    iend = ibeg + self.segm_size
                    assert iend <= self._nint

                    all_pos[il,ibeg:iend,:] = np.matmul(
                        allsegmpos[isegm,1:,:]              ,
                        Sym.SpaceRot.T                        ,
                    )[::-1,:]

        return all_pos
        
    @cython.final
    def params_to_segmpos(self, double[::1] params_mom_buf):

        assert params_mom_buf.shape[0] == self.nparams

        cdef double[:,:,::1] segmpos = np.empty((self.nsegm, self.segm_store, self.geodim), dtype=np.float64)

        with nogil:

            params_to_segmpos(
                params_mom_buf          , self._params_shapes       , self._params_shifts       ,
                                          self._ifft_shapes         , self._ifft_shifts         ,
                self._params_basis_buf  , self._params_basis_shapes , self._params_basis_shifts ,
                self._nnz_k_buf         , self._nnz_k_shapes        , self._nnz_k_shifts        ,
                self._co_in_buf         , self._co_in_shapes        , self._co_in_shifts        ,
                                          self._pos_slice_shapes    , self._pos_slice_shifts    ,
                self._ncoeff_min_loop   , self.nnpr                 ,
                self._loopnb            , self._loopmass            ,
                self._InterSpaceRotIsId , self._InterSpaceRot       , self._InterTimeRev        ,
                self._gensegm_to_body   , self._gensegm_to_iint     ,
                self._bodyloop          , self.segm_size            , self.segm_store           ,
                segmpos                 ,
            )

        return np.asarray(segmpos)
    
    @cython.final
    def segmpos_to_params(self, double[:,:,::1] segmpos):

        assert self.segm_store == segmpos.shape[1]

        cdef double[::1] params_mom_buf = np.empty((self.nparams), dtype=np.float64)

        with nogil:

            segmpos_to_params(
                segmpos                 , self._params_shapes       , self._params_shifts       ,
                                          self._ifft_shapes         , self._ifft_shifts         ,
                self._params_basis_buf  , self._params_basis_shapes , self._params_basis_shifts ,
                self._nnz_k_buf         , self._nnz_k_shapes        , self._nnz_k_shifts        ,
                self._co_in_buf         , self._co_in_shapes        , self._co_in_shifts        ,
                                          self._pos_slice_shapes    , self._pos_slice_shifts    ,
                self._ncoeff_min_loop   , self.nnpr                 ,
                self._loopnb            , self._loopmass            ,
                self._InterSpaceRotIsId , self._InterSpaceRot       , self._InterTimeRev        ,
                self._gensegm_to_body   , self._gensegm_to_iint     ,
                self._bodyloop          , self.segm_size            , self.segm_store           ,
                params_mom_buf          ,
            )

        return np.asarray(params_mom_buf)






    
 
@cython.cdivision(True)
cdef void changevar_mom(
    const double *params_mom_buf, long[:,::1] params_shapes     , long[::1] params_shifts   ,
    long[::1] nnz_k_buf         , long[:,::1] nnz_k_shapes      , long[::1] nnz_k_shifts    ,
    bint[::1] co_in_buf         , long[:,::1] co_in_shapes      , long[::1] co_in_shifts    ,
    long[::1] ncoeff_min_loop   ,
    long[::1] loopnb            , double[::1] loopmass          ,
    const double *params_pos_buf, 
) noexcept nogil:

    cdef double* cur_params_mom_buf = params_mom_buf
    cdef double* cur_param_pos_buf = params_pos_buf
    cdef double loopmul, mul

    cdef int nloop = params_shapes.shape[0]
    cdef Py_ssize_t il, idim, ipr, ik, iparam
    cdef long k, ko

    for il in range(nloop):

        loopmul = 1./csqrt(loopnb[il] * loopmass[il] * cfourpisq)

        # ipr = 0 treated separately
        # ik = 0 treated separately
        if nnz_k_shapes[il,0] > 0:

            k = nnz_k_buf[nnz_k_shifts[il]]
        
            if k == 0:

                for iparam in range(params_shapes[il,2]):

                    if co_in_buf[co_in_shifts[il]+iparam]:

                        cur_param_pos_buf[0] = loopmul*cur_params_mom_buf[0]

                        cur_params_mom_buf += 1
                        cur_param_pos_buf += 1
                    
                    else:
                        # DO NOT INCREMENT cur_param_mom_buf !!!
                        cur_param_pos_buf[0] = 0
                        cur_param_pos_buf += 1

            else:

                mul = loopmul/k

                for iparam in range(params_shapes[il,2]):

                    cur_param_pos_buf[0] = mul*cur_params_mom_buf[0]

                    cur_params_mom_buf += 1
                    cur_param_pos_buf += 1

        for ik in range(1, nnz_k_shapes[il,0]):

            k = nnz_k_buf[nnz_k_shifts[il]+ik]
            mul = loopmul/k

            for iparam in range(params_shapes[il,2]):

                cur_param_pos_buf[0] = mul*cur_params_mom_buf[0]

                cur_params_mom_buf += 1
                cur_param_pos_buf += 1

        for ipr in range(1,params_shapes[il,0]):

            ko = ipr * ncoeff_min_loop[il]

            for ik in range(nnz_k_shapes[il,0]):

                k = ko + nnz_k_buf[nnz_k_shifts[il]+ik]
                mul = loopmul/k

                for iparam in range(params_shapes[il,2]):

                    cur_param_pos_buf[0] = mul*cur_params_mom_buf[0]

                    cur_params_mom_buf += 1
                    cur_param_pos_buf += 1
 
@cython.cdivision(True)
cdef void changevar_mom_invT(
    const double *params_mom_buf, long[:,::1] params_shapes     , long[::1] params_shifts   ,
    long[::1] nnz_k_buf         , long[:,::1] nnz_k_shapes      , long[::1] nnz_k_shifts    ,
    bint[::1] co_in_buf         , long[:,::1] co_in_shapes      , long[::1] co_in_shifts    ,
    long[::1] ncoeff_min_loop   ,
    long[::1] loopnb            , double[::1] loopmass          ,
    const double *params_pos_buf, 
) noexcept nogil:

    cdef double* cur_params_mom_buf = params_mom_buf
    cdef double* cur_param_pos_buf = params_pos_buf
    cdef double loopmul, mul

    cdef int nloop = params_shapes.shape[0]
    cdef Py_ssize_t il, idim, ipr, ik, iparam
    cdef long k, ko

    for il in range(nloop):

        loopmul = csqrt(loopnb[il] * loopmass[il] * cfourpisq)

        # ipr = 0 treated separately
        # ik = 0 treated separately
        if nnz_k_shapes[il,0] > 0:

            k = nnz_k_buf[nnz_k_shifts[il]]
        
            if k == 0:

                for iparam in range(params_shapes[il,2]):

                    if co_in_buf[co_in_shifts[il]+iparam]:

                        cur_param_pos_buf[0] = loopmul*cur_params_mom_buf[0]

                        cur_params_mom_buf += 1
                        cur_param_pos_buf += 1
                    
                    else:
                        # DO NOT INCREMENT cur_param_mom_buf !!!
                        cur_param_pos_buf[0] = 0
                        cur_param_pos_buf += 1

            else:

                mul = loopmul*k

                for iparam in range(params_shapes[il,2]):

                    cur_param_pos_buf[0] = mul*cur_params_mom_buf[0]

                    cur_params_mom_buf += 1
                    cur_param_pos_buf += 1

        for ik in range(1, nnz_k_shapes[il,0]):

            k = nnz_k_buf[nnz_k_shifts[il]+ik]
            mul = loopmul*k

            for iparam in range(params_shapes[il,2]):

                cur_param_pos_buf[0] = mul*cur_params_mom_buf[0]

                cur_params_mom_buf += 1
                cur_param_pos_buf += 1

        for ipr in range(1,params_shapes[il,0]):

            ko = ipr * ncoeff_min_loop[il]

            for ik in range(nnz_k_shapes[il,0]):

                k = ko + nnz_k_buf[nnz_k_shifts[il]+ik]
                mul = loopmul*k

                for iparam in range(params_shapes[il,2]):

                    cur_param_pos_buf[0] = mul*cur_params_mom_buf[0]

                    cur_params_mom_buf += 1
                    cur_param_pos_buf += 1

cdef void changevar_mom_inv(
    const double *params_pos_buf, long[:,::1] params_shapes     , long[::1] params_shifts   ,
    long[::1] nnz_k_buf         , long[:,::1] nnz_k_shapes      , long[::1] nnz_k_shifts    ,
    bint[::1] co_in_buf         , long[:,::1] co_in_shapes      , long[::1] co_in_shifts    ,
    long[::1] ncoeff_min_loop   ,
    long[::1] loopnb            , double[::1] loopmass          ,
    const double *params_mom_buf, 
) noexcept nogil:

    cdef double* cur_param_pos_buf = params_pos_buf
    cdef double* cur_params_mom_buf = params_mom_buf
    cdef double loopmul, mul

    cdef int nloop = params_shapes.shape[0]
    cdef Py_ssize_t il, idim, ipr, ik, iparam
    cdef long k, ko

    for il in range(nloop):

        loopmul = csqrt(loopnb[il] * loopmass[il] * cfourpisq)

        # ipr = 0 treated separately
        # ik = 0 treated separately
        if nnz_k_shapes[il,0] > 0:

            k = nnz_k_buf[nnz_k_shifts[il]]
        
            if k == 0:

                for iparam in range(params_shapes[il,2]):

                    if co_in_buf[co_in_shifts[il]+iparam]:

                        cur_params_mom_buf[0] = loopmul*cur_param_pos_buf[0]

                        cur_params_mom_buf += 1
                        cur_param_pos_buf += 1

                    else:
                        # DO NOT INCREMENT cur_params_mom_buf !!!
                        cur_param_pos_buf += 1

            else:

                mul = loopmul * k

                for iparam in range(params_shapes[il,2]):

                    cur_params_mom_buf[0] = mul*cur_param_pos_buf[0]

                    cur_params_mom_buf += 1
                    cur_param_pos_buf += 1

        for ik in range(1, nnz_k_shapes[il,0]):

            k = nnz_k_buf[nnz_k_shifts[il]+ik]
            mul = loopmul * k

            for iparam in range(params_shapes[il,2]):

                cur_params_mom_buf[0] = mul*cur_param_pos_buf[0]

                cur_params_mom_buf += 1
                cur_param_pos_buf += 1

        for ipr in range(1,params_shapes[il,0]):

            ko = ipr * ncoeff_min_loop[il]

            for ik in range(nnz_k_shapes[il,0]):

                k = ko + nnz_k_buf[nnz_k_shifts[il]+ik]
                mul = loopmul * k

                for iparam in range(params_shapes[il,2]):

                    cur_params_mom_buf[0] = mul*cur_param_pos_buf[0]

                    cur_params_mom_buf += 1
                    cur_param_pos_buf += 1

cdef void changevar_mom_T(
    const double *params_pos_buf, long[:,::1] params_shapes     , long[::1] params_shifts   ,
    long[::1] nnz_k_buf         , long[:,::1] nnz_k_shapes      , long[::1] nnz_k_shifts    ,
    bint[::1] co_in_buf         , long[:,::1] co_in_shapes      , long[::1] co_in_shifts    ,
    long[::1] ncoeff_min_loop   ,
    long[::1] loopnb            , double[::1] loopmass          ,
    const double *params_mom_buf, 
) noexcept nogil:

    cdef double* cur_param_pos_buf = params_pos_buf
    cdef double* cur_params_mom_buf = params_mom_buf
    cdef double loopmul, mul

    cdef int nloop = params_shapes.shape[0]
    cdef Py_ssize_t il, idim, ipr, ik, iparam
    cdef long k, ko

    for il in range(nloop):

        loopmul = 1./csqrt(loopnb[il] * loopmass[il] * cfourpisq)

        # ipr = 0 treated separately
        # ik = 0 treated separately
        if nnz_k_shapes[il,0] > 0:

            k = nnz_k_buf[nnz_k_shifts[il]]
        
            if k == 0:

                for iparam in range(params_shapes[il,2]):

                    if co_in_buf[co_in_shifts[il]+iparam]:

                        cur_params_mom_buf[0] = loopmul*cur_param_pos_buf[0]

                        cur_params_mom_buf += 1
                        cur_param_pos_buf += 1

                    else:
                        # DO NOT INCREMENT cur_params_mom_buf !!!
                        cur_param_pos_buf += 1

            else:

                mul = loopmul / k

                for iparam in range(params_shapes[il,2]):

                    cur_params_mom_buf[0] = mul*cur_param_pos_buf[0]

                    cur_params_mom_buf += 1
                    cur_param_pos_buf += 1

        for ik in range(1, nnz_k_shapes[il,0]):

            k = nnz_k_buf[nnz_k_shifts[il]+ik]
            mul = loopmul / k

            for iparam in range(params_shapes[il,2]):

                cur_params_mom_buf[0] = mul*cur_param_pos_buf[0]

                cur_params_mom_buf += 1
                cur_param_pos_buf += 1

        for ipr in range(1,params_shapes[il,0]):

            ko = ipr * ncoeff_min_loop[il]

            for ik in range(nnz_k_shapes[il,0]):

                k = ko + nnz_k_buf[nnz_k_shifts[il]+ik]
                mul = loopmul / k

                for iparam in range(params_shapes[il,2]):

                    cur_params_mom_buf[0] = mul*cur_param_pos_buf[0]

                    cur_params_mom_buf += 1
                    cur_param_pos_buf += 1
 
@cython.cdivision(True)
cdef double params_to_kin_nrg(
    const double *params_mom_buf, long[:,::1] params_shapes , long[::1] params_shifts   ,
    long[::1] ncor_loop         , long[::1] nco_in_loop     ,
) noexcept nogil:

    cdef double* loc = params_mom_buf
    cdef double loopmul

    cdef int nloop = params_shapes.shape[0]
    cdef Py_ssize_t il

    cdef double kin = 0
    cdef int beg
    cdef int n
    cdef int ncor_loop_cum = 0

    for il in range(nloop):

        loc += ncor_loop[il]
        n = params_shifts[il+1] - (params_shifts[il] + ncor_loop[il] + nco_in_loop[il])
        
        kin += scipy.linalg.cython_blas.ddot(&n, loc, &int_one, loc, &int_one)

        loc += n

    kin *= 0.5

    return kin
 
@cython.cdivision(True)
cdef void params_to_kin_nrg_grad(
    const double *params_mom_buf, long[:,::1] params_shapes , long[::1] params_shifts   ,
    long[::1] ncor_loop         , long[::1] nco_in_loop     ,
    const double *grad_buf      ,
) noexcept nogil:

    cdef double* loc = params_mom_buf
    cdef double* grad_loc = grad_buf
    cdef double loopmul

    cdef int nloop = params_shapes.shape[0]
    cdef Py_ssize_t il

    cdef double kin = 0
    cdef int beg
    cdef int n
    cdef int ncor_loop_cum = 0

    for il in range(nloop):

        memset(grad_loc, 0, sizeof(double)*ncor_loop[il])

        loc += ncor_loop[il]
        grad_loc += ncor_loop[il]
        n = params_shifts[il+1] - (params_shifts[il] + ncor_loop[il] + nco_in_loop[il])
        
        scipy.linalg.cython_blas.dcopy(&n, loc, &int_one, grad_loc, &int_one)

        loc += n
        grad_loc += n

@cython.cdivision(True)
cdef void inplace_twiddle(
    const double complex* const_ifft    ,
    const long* nnz_k       ,
    long nint               ,
    int n_inter             ,
    int ncoeff_min_loop_nnz ,
    int nppl                ,
    int direction           , # -1 or 1
) noexcept nogil:

    cdef double complex w, wo, winter
    cdef double complex w_pow[16] # minimum size of int on all machines.

    cdef int ibit
    cdef int nbit = 1
    cdef long twopow = 1
    cdef bint *nnz_bin 

    cdef double complex* ifft = <double complex*> const_ifft

    cdef Py_ssize_t m, j, i, k

    if ncoeff_min_loop_nnz > 0:

        if nnz_k[ncoeff_min_loop_nnz-1] > 0:

            wo =  cexp(direction * citwopi / nint)
            winter = 1.

            while (twopow < nnz_k[ncoeff_min_loop_nnz-1]) :
                twopow *= 2
                nbit += 1

            nnz_bin  = <bint*> malloc(sizeof(bint)*nbit*ncoeff_min_loop_nnz)
            for j in range(ncoeff_min_loop_nnz):
                for ibit in range(nbit):
                    nnz_bin[ibit + j*nbit] = ((nnz_k[j] >> (ibit)) & 1) # tests if the ibit-th bit of nnz_k[j] is one 

            for m in range(n_inter):

                w_pow[0] = winter
                for ibit in range(nbit-1):
                    w_pow[ibit+1] = w_pow[ibit] * w_pow[ibit] 

                for j in range(ncoeff_min_loop_nnz):
                    
                    w = 1.
                    for ibit in range(nbit):
                        if nnz_bin[ibit + j*nbit]:
                            w *= w_pow[ibit] 

                    # w = cexp(direction * citwopi * m * nnz_k[j] / nint)

                    for i in range(nppl):
                        ifft[0] *= w
                        ifft += 1

                winter *= wo

            free(nnz_bin)

@cython.cdivision(True)
cdef void partial_fft_to_pos_slice_1npr(
    const double complex* const_ifft        ,
    const double complex* params_basis      ,  
    const long* nnz_k                       ,
    const double* pos_slice                 ,
    int npr                                 ,
    int ncoeff_min_loop_nnz                 ,
    int ncoeff_min_loop                     ,
    int geodim                              ,
    int nppl                                ,
) noexcept nogil:
 
    cdef int n_inter = npr+1
    cdef long nint = 2*ncoeff_min_loop*npr

    cdef double dfac

    # Casting complex double to double array
    cdef double* params_basis_r = <double*> params_basis
    cdef double* ifft_r = <double*> const_ifft

    cdef int ndcom = 2*ncoeff_min_loop_nnz*nppl

    inplace_twiddle(const_ifft, nnz_k, nint, n_inter, ncoeff_min_loop_nnz, nppl, -1)

    dfac = 1./(npr * ncoeff_min_loop)

    # Computes a.real * b.real.T + a.imag * b.imag.T using clever memory arrangement and a single gemm call
    scipy.linalg.cython_blas.dgemm(transt, transn, &geodim, &n_inter, &ndcom, &dfac, params_basis_r, &ndcom, ifft_r, &ndcom, &zero_double, pos_slice, &geodim)



@cython.cdivision(True)
cdef void pos_slice_to_partial_fft_1npr(
    const double* pos_slice                 ,
    const double complex* params_basis      ,  
    const long* nnz_k                       ,
    const double complex* const_ifft        ,
    int npr                                 ,
    int ncoeff_min_loop_nnz                 ,
    int ncoeff_min_loop                     ,
    int geodim                              ,
    int nppl                                ,
) noexcept nogil:
 
    cdef int n_inter = npr+1
    cdef long nint = 2*ncoeff_min_loop*npr

    cdef double dfac

    # Casting complex double to double array
    cdef double* params_basis_r = <double*> params_basis
    cdef double* ifft_r = <double*> const_ifft

    cdef int ndcom = 2*ncoeff_min_loop_nnz*nppl

    dfac = 2*(npr * ncoeff_min_loop)

    # Again with the clever memory arrangement and a single gemm call
    scipy.linalg.cython_blas.dgemm(transn, transn, &ndcom, &n_inter, &geodim, &dfac, params_basis_r, &ndcom, pos_slice, &geodim, &zero_double, ifft_r, &ndcom)

    inplace_twiddle(const_ifft, nnz_k, nint, n_inter, ncoeff_min_loop_nnz, nppl, 1)



@cython.cdivision(True)
cdef void partial_fft_to_pos_slice_2npr(
    const double complex* const_ifft        ,
    const double complex* params_basis      ,
    const long* nnz_k                       ,
    const double* const_pos_slice           ,
    int npr                                 ,
    int ncoeff_min_loop_nnz                 ,
    int ncoeff_min_loop                     ,
    int geodim                              ,
    int nppl                                ,
) noexcept nogil:

    cdef int n_inter = npr+1
    cdef long nint = 2*ncoeff_min_loop*npr

    cdef double dfac

    # Casting complex double to double array
    cdef double* params_basis_r = <double*> params_basis
    cdef double complex* ifft = const_ifft
    cdef double* ifft_r = <double*> const_ifft
    cdef double* pos_slice = const_pos_slice
    
    cdef int nzcom = ncoeff_min_loop_nnz*nppl
    cdef int ndcom = 2*nzcom
    cdef int nconj

    cdef double complex w
    cdef Py_ssize_t m, j, i

    inplace_twiddle(const_ifft, nnz_k, nint, n_inter, ncoeff_min_loop_nnz, nppl, -1)

    dfac = 1./(npr * ncoeff_min_loop)

    # Computes a.real * b.real.T + a.imag * b.imag.T using clever memory arrangement and a single gemm call
    scipy.linalg.cython_blas.dgemm(transt, transn, &geodim, &n_inter, &ndcom, &dfac, params_basis_r, &ndcom, ifft_r, &ndcom, &zero_double, pos_slice, &geodim)


    n_inter = npr-1
    ifft += nzcom
    for j in range(ncoeff_min_loop_nnz):
        w = cexp(citwopi*nnz_k[j]/ncoeff_min_loop)
        for i in range(nppl):
            scipy.linalg.cython_blas.zscal(&n_inter,&w,ifft,&nzcom)
            ifft += 1

    # Inplace conjugaison
    ifft_r += 1 + ndcom
    nconj = n_inter*nzcom
    scipy.linalg.cython_blas.dscal(&nconj,&minusone_double,ifft_r,&int_two)

    cdef double complex *ztmp = <double complex*> malloc(sizeof(double complex) * nconj)
    cdef double *dtmp = (<double*> ztmp) + n_inter*ndcom

    ifft_r -= 1
    for i in range(n_inter):
        dtmp -= ndcom
        scipy.linalg.cython_blas.dcopy(&ndcom,ifft_r,&int_one,dtmp,&int_one)
        ifft_r += ndcom

    pos_slice += (npr+1)*geodim
    scipy.linalg.cython_blas.dgemm(transt, transn, &geodim, &n_inter, &ndcom, &dfac, params_basis_r, &ndcom, dtmp, &ndcom, &zero_double, pos_slice, &geodim)

    free(ztmp)
    
@cython.cdivision(True)
cdef void pos_slice_to_partial_fft_2npr(
    const double* const_pos_slice           ,
    const double complex* params_basis      ,
    const long* nnz_k                       ,
    const double complex* const_ifft        ,
    int npr                                 ,
    int ncoeff_min_loop_nnz                 ,
    int ncoeff_min_loop                     ,
    int geodim                              ,
    int nppl                                ,
) noexcept nogil:

    cdef int n_inter = npr+1
    cdef long nint = 2*ncoeff_min_loop*npr

    cdef double dfac

    # Casting complex double to double array
    cdef double* params_basis_r = <double*> params_basis
    cdef double complex* ifft = const_ifft
    cdef double* ifft_r = <double*> const_ifft
    cdef double* pos_slice
    
    cdef int nzcom = ncoeff_min_loop_nnz*nppl
    cdef int ndcom = 2*nzcom
    cdef int nconj

    cdef double complex w
    cdef Py_ssize_t m, j, i

    dfac = (npr * ncoeff_min_loop)
    n_inter = npr-1
    nconj = n_inter*nzcom

    cdef double complex *ztmp = <double complex*> malloc(sizeof(double complex) * nconj)
    cdef double *dtmp = (<double*> ztmp)

    pos_slice = const_pos_slice + (npr+1)*geodim
    # Again with the clever memory arrangement and a single gemm call
    scipy.linalg.cython_blas.dgemm(transn, transn, &ndcom, &n_inter, &geodim, &dfac, params_basis_r, &ndcom, pos_slice, &geodim, &zero_double, dtmp, &ndcom)

    dtmp += n_inter*ndcom
    ifft_r = (<double*> const_ifft) + ndcom
    for i in range(n_inter):
        dtmp -= ndcom
        scipy.linalg.cython_blas.dcopy(&ndcom,dtmp,&int_one,ifft_r,&int_one)
        ifft_r += ndcom

    free(ztmp)

    # Inplace conjugaison
    ifft_r = (<double*> const_ifft) + 1 + ndcom
    scipy.linalg.cython_blas.dscal(&nconj,&minusone_double,ifft_r,&int_two)    

    ifft += nzcom
    for j in range(ncoeff_min_loop_nnz):
        w = cexp(cminusitwopi*nnz_k[j]/ncoeff_min_loop)
        for i in range(nppl):
            scipy.linalg.cython_blas.zscal(&n_inter,&w,ifft,&nzcom)
            ifft += 1

    # Initialize start and end of ifft_r
    ifft_r = (<double*> const_ifft)
    memset(ifft_r, 0, sizeof(double)*ndcom)
    ifft_r = (<double*> const_ifft) + npr*ndcom
    memset(ifft_r, 0, sizeof(double)*ndcom)

    pos_slice = const_pos_slice
    n_inter = npr+1
    ifft_r = (<double*> const_ifft) 
    # Again with the clever memory arrangement and a single gemm call
    scipy.linalg.cython_blas.dgemm(transn, transn, &ndcom, &n_inter, &geodim, &dfac, params_basis_r, &ndcom, pos_slice, &geodim, &one_double, ifft_r, &ndcom)

    # Double start and end of ifft_r
    scipy.linalg.cython_blas.dscal(&ndcom,&two_double,ifft_r,&int_one)
    ifft_r += npr*ndcom
    scipy.linalg.cython_blas.dscal(&ndcom,&two_double,ifft_r,&int_one)

    inplace_twiddle(const_ifft, nnz_k, nint, n_inter, ncoeff_min_loop_nnz, nppl, 1)

cdef void params_to_ifft(
    double* params_buf              , long[:,::1] params_shapes     , long[::1] params_shifts   ,
    long[::1] nnz_k_buf             , long[:,::1] nnz_k_shapes      , long[::1] nnz_k_shifts    ,
    double complex *ifft_buf_ptr    , long[:,::1] ifft_shapes       , long[::1] ifft_shifts     ,
) noexcept nogil:

    cdef double [:,:,::1] params
    cdef long[::1] nnz_k
    cdef double complex[:,:,::1] ifft

    cdef int nloop = params_shapes.shape[0]
    cdef int n
    cdef double complex * dest
    cdef Py_ssize_t il, i

    for il in range(nloop):

        if params_shapes[il,1] > 0:

            with gil:

                params = <double[:params_shapes[il,0],:params_shapes[il,1],:params_shapes[il,2]:1]> &params_buf[params_shifts[il]]
                nnz_k = <long[:nnz_k_shapes[il,0]:1]> &nnz_k_buf[nnz_k_shifts[il]]

                if nnz_k.shape[0] > 0:
                    if nnz_k[0] == 0:
                        for i in range(params.shape[2]):
                            params[0,0,i] *= 0.5

                ifft = scipy.fft.rfft(params, axis=0, n=2*params.shape[0])

                # TODO : remove this once validation is done
                if nnz_k.shape[0] > 0:
                    if nnz_k[0] == 0:
                        for i in range(params.shape[2]):
                            params[0,0,i] *= 2


            dest = ifft_buf_ptr + ifft_shifts[il]
            n = ifft_shifts[il+1] - ifft_shifts[il]
            scipy.linalg.cython_blas.zcopy(&n,&ifft[0,0,0],&int_one,dest,&int_one)

cdef void ifft_to_params(
    double complex *ifft_buf_ptr    , long[:,::1] ifft_shapes       , long[::1] ifft_shifts     ,
    long[::1] nnz_k_buf             , long[:,::1] nnz_k_shapes      , long[::1] nnz_k_shifts    ,
    double* params_buf              , long[:,::1] params_shapes     , long[::1] params_shifts   ,
) noexcept nogil:

    cdef double [:,:,::1] params
    cdef long[::1] nnz_k
    cdef double complex[:,:,::1] ifft

    cdef int nloop = params_shapes.shape[0]
    cdef int n
    cdef double* dest
    cdef Py_ssize_t il, i

    for il in range(nloop):

        if params_shapes[il,1] > 0:

            with gil:

                nnz_k = <long[:nnz_k_shapes[il,0]:1]> &nnz_k_buf[nnz_k_shifts[il]]
                ifft = <double complex[:ifft_shapes[il,0],:ifft_shapes[il,1],:ifft_shapes[il,2]:1]> &ifft_buf_ptr[ifft_shifts[il]]

                params = scipy.fft.irfft(ifft, axis=0)

            dest = params_buf + params_shifts[il]
            n = params_shifts[il+1] - params_shifts[il]
            scipy.linalg.cython_blas.dcopy(&n,&params[0,0,0],&int_one,dest,&int_one)

cdef void ifft_to_pos_slice(
    const double complex *ifft_buf_ptr            , long[:,::1] ifft_shapes           , long[::1] ifft_shifts         ,
    const double complex *params_basis_buf_ptr    , long[:,::1] params_basis_shapes   , long[::1] params_basis_shifts ,
    const long* nnz_k_buf_ptr                     , long[:,::1] nnz_k_shapes          , long[::1] nnz_k_shifts        ,
    const double* pos_slice_buf_ptr               , long[:,::1] pos_slice_shapes      , long[::1] pos_slice_shifts    ,
    long[::1] ncoeff_min_loop, long nnpr,
) noexcept nogil:

    cdef double complex* ifft
    cdef double complex* params_basis
    cdef long* nnz_k
    cdef double* pos_slice

    cdef int nloop = ncoeff_min_loop.shape[0]
    cdef Py_ssize_t il, i

    cdef int npr
    cdef int ncoeff_min_loop_il
    cdef int ncoeff_min_loop_nnz
    cdef int geodim
    cdef int nppl

    for il in range(nloop):

        if params_basis_shapes[il,1] > 0:

            ifft = ifft_buf_ptr + ifft_shifts[il]
            params_basis = params_basis_buf_ptr + params_basis_shifts[il]
            nnz_k = nnz_k_buf_ptr + nnz_k_shifts[il]
            pos_slice = pos_slice_buf_ptr + pos_slice_shifts[il]

            npr = ifft_shapes[il,0] - 1
            ncoeff_min_loop_nnz = nnz_k_shapes[il,0]
            ncoeff_min_loop_il = ncoeff_min_loop[il]
            geodim = params_basis_shapes[il,0]
            nppl = ifft_shapes[il,2] 

            if nnpr == 1:
                partial_fft_to_pos_slice_1npr(
                    ifft, params_basis, nnz_k, pos_slice,
                    npr, ncoeff_min_loop_nnz, ncoeff_min_loop_il, geodim, nppl,
                )
            else:
                partial_fft_to_pos_slice_2npr(
                    ifft, params_basis, nnz_k, pos_slice,
                    npr, ncoeff_min_loop_nnz, ncoeff_min_loop_il, geodim, nppl,
                )

cdef void pos_slice_to_ifft(
    const double* pos_slice_buf_ptr               , long[:,::1] pos_slice_shapes      , long[::1] pos_slice_shifts    ,
    const double complex *params_basis_buf_ptr    , long[:,::1] params_basis_shapes   , long[::1] params_basis_shifts ,
    const long* nnz_k_buf_ptr                     , long[:,::1] nnz_k_shapes          , long[::1] nnz_k_shifts        ,
    const double complex *ifft_buf_ptr            , long[:,::1] ifft_shapes           , long[::1] ifft_shifts         ,
    long[::1] ncoeff_min_loop, long nnpr,
) noexcept nogil:

    cdef double complex* ifft
    cdef double complex* params_basis
    cdef long* nnz_k
    cdef double* pos_slice

    cdef int nloop = ncoeff_min_loop.shape[0]
    cdef Py_ssize_t il, i

    cdef int npr
    cdef int ncoeff_min_loop_il
    cdef int ncoeff_min_loop_nnz
    cdef int geodim
    cdef int nppl


    for il in range(nloop):

        if params_basis_shapes[il,1] > 0:

            ifft = ifft_buf_ptr + ifft_shifts[il]
            params_basis = params_basis_buf_ptr + params_basis_shifts[il]
            nnz_k = nnz_k_buf_ptr + nnz_k_shifts[il]
            pos_slice = pos_slice_buf_ptr + pos_slice_shifts[il]

            npr = ifft_shapes[il,0] - 1
            ncoeff_min_loop_nnz = nnz_k_shapes[il,0]
            ncoeff_min_loop_il = ncoeff_min_loop[il]
            geodim = params_basis_shapes[il,0]
            nppl = ifft_shapes[il,2] 

            if nnpr == 1:
                pos_slice_to_partial_fft_1npr(
                    pos_slice, params_basis, nnz_k, ifft,
                    npr, ncoeff_min_loop_nnz, ncoeff_min_loop_il, geodim, nppl,
                )
            else:
                pos_slice_to_partial_fft_2npr(
                    pos_slice, params_basis, nnz_k, ifft,
                    npr, ncoeff_min_loop_nnz, ncoeff_min_loop_il, geodim, nppl,
                )

cdef void pos_slice_to_segmpos(
    const double* pos_slice_buf_ptr , long[:,::1] pos_slice_shapes  , long[::1] pos_slice_shifts    ,
    const double* segmpos_buf_ptr   ,
    bint[::1] InterSpaceRotIsId     ,
    double[:,:,::1] InterSpaceRot   ,
    long[::1] InterTimeRev          ,
    long[::1] gensegm_to_body     ,
    long[::1] gensegm_to_iint     ,
    long[::1] BodyLoop              ,
    long segm_size                  ,
    long segm_store                 ,
) noexcept nogil:

    cdef int nsegm = gensegm_to_body.shape[0]
    cdef double* pos_slice
    cdef double* segmpos
    cdef double* tmp_loc
    cdef double* tmp

    cdef int geodim = InterSpaceRot.shape[1]
    # cdef int segm_size_int = segm_size
    cdef int segm_store_int = segm_store
    cdef int nitems_size = segm_size*geodim
    cdef int nitems_store = segm_store*geodim
    cdef Py_ssize_t isegm, ib, il, iint
    cdef Py_ssize_t i, idim

    cdef bint NeedsAllocate = False

    for isegm in range(nsegm):
        NeedsAllocate = (NeedsAllocate or ((InterTimeRev[isegm] < 0) and not(InterSpaceRotIsId[isegm])))

    if NeedsAllocate:
        tmp_loc = <double*> malloc(sizeof(double)*nitems_store)

    for isegm in range(nsegm):

        ib = gensegm_to_body[isegm]
        iint = gensegm_to_iint[isegm]
        il = BodyLoop[ib]

        if InterTimeRev[isegm] == 1:

            pos_slice = pos_slice_buf_ptr + pos_slice_shifts[il] + nitems_size*iint
            segmpos = segmpos_buf_ptr + nitems_store*isegm

            if InterSpaceRotIsId[isegm]:
                scipy.linalg.cython_blas.dcopy(&nitems_store,pos_slice,&int_one,segmpos,&int_one)
            else:
                scipy.linalg.cython_blas.dgemm(transt, transn, &geodim, &segm_store_int, &geodim, &one_double, &InterSpaceRot[isegm,0,0], &geodim, pos_slice, &geodim, &zero_double, segmpos, &geodim)

        else:

            pos_slice = pos_slice_buf_ptr + pos_slice_shifts[il] + nitems_size*iint + geodim
            segmpos = segmpos_buf_ptr + nitems_store*(isegm+1) - geodim

            if InterSpaceRotIsId[isegm]:

                for i in range(segm_store):
                    for idim in range(geodim):
                        segmpos[idim] = pos_slice[idim]
                    segmpos -= geodim
                    pos_slice += geodim
                            
            else:
                
                tmp = tmp_loc
                scipy.linalg.cython_blas.dgemm(transt, transn, &geodim, &segm_store_int, &geodim, &one_double, &InterSpaceRot[isegm,0,0], &geodim, pos_slice, &geodim, &zero_double, tmp, &geodim)

                for i in range(segm_store):
                    for idim in range(geodim):
                        segmpos[idim] = tmp[idim]
                    segmpos -= geodim
                    tmp += geodim

    if NeedsAllocate:
        free(tmp_loc)

cdef void segmpos_to_pos_slice(
    const double* segmpos_buf_ptr   ,
    const double* pos_slice_buf_ptr , long[:,::1] pos_slice_shapes  , long[::1] pos_slice_shifts    ,
    bint[::1] InterSpaceRotIsId     ,
    double[:,:,::1] InterSpaceRot   ,
    long[::1] InterTimeRev          ,
    long[::1] gensegm_to_body     ,
    long[::1] gensegm_to_iint     ,
    long[::1] BodyLoop              ,
    long segm_size                  ,
    long segm_store                 ,
) noexcept nogil:

    cdef int nsegm = gensegm_to_body.shape[0]
    cdef double* pos_slice
    cdef double* segmpos
    cdef double* tmp_loc
    cdef double* tmp

    cdef int geodim = InterSpaceRot.shape[1]
    cdef int segm_store_int = segm_store
    cdef int nitems_size = segm_size*geodim
    cdef int nitems_store = segm_store*geodim
    cdef Py_ssize_t isegm, ib, il, iint
    cdef Py_ssize_t i, idim

    cdef bint NeedsAllocate = False

    for isegm in range(nsegm):
        NeedsAllocate = (NeedsAllocate or ((InterTimeRev[isegm] < 0) and not(InterSpaceRotIsId[isegm])))

    if NeedsAllocate:
        tmp_loc = <double*> malloc(sizeof(double)*nitems_store)

    for isegm in range(nsegm):

        ib = gensegm_to_body[isegm]
        iint = gensegm_to_iint[isegm]
        il = BodyLoop[ib]

        if InterTimeRev[isegm] == 1:

            pos_slice = pos_slice_buf_ptr + pos_slice_shifts[il] + nitems_size*iint
            segmpos = segmpos_buf_ptr + nitems_store*isegm

            if InterSpaceRotIsId[isegm]:
                scipy.linalg.cython_blas.dcopy(&nitems_store,segmpos,&int_one,pos_slice,&int_one)
            else:
                scipy.linalg.cython_blas.dgemm(transt, transn, &geodim, &segm_store_int, &geodim, &one_double, &InterSpaceRot[isegm,0,0], &geodim, pos_slice, &geodim, &zero_double, segmpos, &geodim)

        else:

            pos_slice = pos_slice_buf_ptr + pos_slice_shifts[il] + nitems_size*iint + geodim
            segmpos = segmpos_buf_ptr + nitems_store*(isegm+1) - geodim

            if InterSpaceRotIsId[isegm]:

                for i in range(segm_store):
                    for idim in range(geodim):
                        segmpos[idim] = pos_slice[idim]
                    segmpos -= geodim
                    pos_slice += geodim
                            
            else:
                
                tmp = tmp_loc
                scipy.linalg.cython_blas.dgemm(transt, transn, &geodim, &segm_store_int, &geodim, &one_double, &InterSpaceRot[isegm,0,0], &geodim, pos_slice, &geodim, &zero_double, tmp, &geodim)

                for i in range(segm_store):
                    for idim in range(geodim):
                        segmpos[idim] = tmp[idim]
                    segmpos -= geodim
                    tmp += geodim

    if NeedsAllocate:
        free(tmp_loc)

@cython.cdivision(True)
cdef void params_to_segmpos(
    double[::1] params_mom_buf              , long[:,::1] params_shapes         , long[::1] params_shifts       ,
                                              long[:,::1] ifft_shapes           , long[::1] ifft_shifts         ,
    double complex[::1] params_basis_buf    , long[:,::1] params_basis_shapes   , long[::1] params_basis_shifts ,
    long[::1] nnz_k_buf                     , long[:,::1] nnz_k_shapes          , long[::1] nnz_k_shifts        ,
    bint[::1] co_in_buf                     , long[:,::1] co_in_shapes          , long[::1] co_in_shifts        ,
                                              long[:,::1] pos_slice_shapes      , long[::1] pos_slice_shifts    ,
    long[::1] ncoeff_min_loop               , long nnpr                         ,
    long[::1] loopnb                        , double[::1] loopmass              ,
    bint[::1] InterSpaceRotIsId             , double[:,:,::1] InterSpaceRot     , long[::1] InterTimeRev        ,
    long[::1] gensegm_to_body       ,
    long[::1] gensegm_to_iint       ,
    long[::1] BodyLoop              ,
    long segm_size                  ,
    long segm_store                 ,
    double[:,:,::1] segmpos         ,
) noexcept nogil:

    cdef double *params_pos_buf
    cdef double complex *ifft_buf_ptr
    cdef double *pos_slice_buf_ptr

    cdef int nsegm = gensegm_to_body.shape[0]
    cdef int geodim = InterSpaceRot.shape[1]
    cdef int size

    params_pos_buf = <double*> malloc(sizeof(double)*params_shifts[params_shapes.shape[0]])

    changevar_mom(
        &params_mom_buf[0]  , params_shapes , params_shifts ,
        nnz_k_buf           , nnz_k_shapes  , nnz_k_shifts  ,
        co_in_buf           , co_in_shapes  , co_in_shifts  ,
        ncoeff_min_loop     ,
        loopnb              , loopmass      ,
        params_pos_buf      , 
    )   

    # TODO PYFFTW allocate here if needed.
    ifft_buf_ptr = <double complex *> malloc(sizeof(double complex)*ifft_shifts[ifft_shapes.shape[0]])

    params_to_ifft(
        params_pos_buf  , params_shapes , params_shifts ,
        nnz_k_buf       , nnz_k_shapes  , nnz_k_shifts  ,
        ifft_buf_ptr    , ifft_shapes   , ifft_shifts   ,
    )
 
    free(params_pos_buf)

    size = pos_slice_shifts[pos_slice_shapes.shape[0]]
    pos_slice_buf_ptr = <double *> malloc(sizeof(double)*size)
    memset(pos_slice_buf_ptr, 0, sizeof(double)*size)

    ifft_to_pos_slice(
        ifft_buf_ptr        , ifft_shapes           , ifft_shifts           ,
        &params_basis_buf[0], params_basis_shapes   , params_basis_shifts   ,
        &nnz_k_buf[0]       , nnz_k_shapes          , nnz_k_shifts          ,
        pos_slice_buf_ptr   , pos_slice_shapes      , pos_slice_shifts      ,
        ncoeff_min_loop     , nnpr                  ,
    )

    # PYFFTW free ???
    free(ifft_buf_ptr)

    pos_slice_to_segmpos(
        pos_slice_buf_ptr   , pos_slice_shapes  , pos_slice_shifts ,
        &segmpos[0,0,0] ,
        InterSpaceRotIsId,
        InterSpaceRot     ,
        InterTimeRev      ,
        gensegm_to_body ,
        gensegm_to_iint ,
        BodyLoop        ,
        segm_size       ,
        segm_store      ,
    )

    free(pos_slice_buf_ptr)

@cython.cdivision(True)
cdef void segmpos_to_params(
    double[:,:,::1] segmpos                 , long[:,::1] params_shapes         , long[::1] params_shifts       ,
                                              long[:,::1] ifft_shapes           , long[::1] ifft_shifts         ,
    double complex[::1] params_basis_buf    , long[:,::1] params_basis_shapes   , long[::1] params_basis_shifts ,
    long[::1] nnz_k_buf                     , long[:,::1] nnz_k_shapes          , long[::1] nnz_k_shifts        ,
    bint[::1] co_in_buf                     , long[:,::1] co_in_shapes          , long[::1] co_in_shifts        ,
                                              long[:,::1] pos_slice_shapes      , long[::1] pos_slice_shifts    ,
    long[::1] ncoeff_min_loop               , long nnpr                         ,
    long[::1] loopnb                        , double[::1] loopmass              ,
    bint[::1] InterSpaceRotIsId             , double[:,:,::1] InterSpaceRot     , long[::1] InterTimeRev        ,
    long[::1] gensegm_to_body       ,
    long[::1] gensegm_to_iint       ,
    long[::1] BodyLoop              ,
    long segm_size                  ,
    long segm_store                 ,
    double[::1] params_mom_buf      ,
) noexcept nogil:

    cdef double *params_pos_buf
    cdef double complex *ifft_buf_ptr
    cdef double *pos_slice_buf_ptr

    cdef int nsegm = gensegm_to_body.shape[0]
    cdef int geodim = InterSpaceRot.shape[1]
    cdef int size

    size = pos_slice_shifts[pos_slice_shapes.shape[0]]
    pos_slice_buf_ptr = <double *> malloc(sizeof(double)*size)

    segmpos_to_pos_slice(
        &segmpos[0,0,0] ,
        pos_slice_buf_ptr   , pos_slice_shapes  , pos_slice_shifts ,
        InterSpaceRotIsId,
        InterSpaceRot     ,
        InterTimeRev      ,
        gensegm_to_body ,
        gensegm_to_iint ,
        BodyLoop        ,
        segm_size       ,
        segm_store      ,
    )

    # TODO PYFFTW allocate here if needed.
    ifft_buf_ptr = <double complex *> malloc(sizeof(double complex)*ifft_shifts[ifft_shapes.shape[0]])

    pos_slice_to_ifft(
        pos_slice_buf_ptr   , pos_slice_shapes      , pos_slice_shifts      ,
        &params_basis_buf[0], params_basis_shapes   , params_basis_shifts   ,
        &nnz_k_buf[0]       , nnz_k_shapes          , nnz_k_shifts          ,
        ifft_buf_ptr        , ifft_shapes           , ifft_shifts           ,
        ncoeff_min_loop     , nnpr                  ,
    )

    free(pos_slice_buf_ptr)

    params_pos_buf = <double*> malloc(sizeof(double)*params_shifts[params_shapes.shape[0]])

    ifft_to_params(
        ifft_buf_ptr    , ifft_shapes   , ifft_shifts   ,
        nnz_k_buf       , nnz_k_shapes  , nnz_k_shifts  ,
        params_pos_buf  , params_shapes , params_shifts ,
    )

    # PYFFTW free ??
    free(ifft_buf_ptr)

    changevar_mom_inv(
        params_pos_buf      , params_shapes , params_shifts ,
        nnz_k_buf           , nnz_k_shapes  , nnz_k_shifts  ,
        co_in_buf           , co_in_shapes  , co_in_shifts  ,
        ncoeff_min_loop     ,
        loopnb              , loopmass      ,
        &params_mom_buf[0]  , 
    )   

    free(params_pos_buf)

