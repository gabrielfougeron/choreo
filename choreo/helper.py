import os

import sys,argparse
import random
import numpy as np
import math as m

import copy
import shutil
import time
import builtins

from choreo.scipy_plus import *
from choreo.funs import *

def Make2DChoreoSym(SymType,ib_list):
    r"""
    Defines symmetries of a 2-D system of bodies as classfied in [1] 
    
    Classification :
    C(n,k,l) with k and l relative primes
    D(n,k,l) with k and l relative primes
    Cp(n,2,#) 
    Dp(n,1,#) 
    Dp(n,2,#) 
    Those are exhaustive for 2-D purely choreographic symmetries (i.e. 1 loop with 1 path)
    
    I also added p and q for space rotation. This might however not be exhaustive.
    
    SymType  => Dictionary containing the following keys :
        'name'
        'n'
        'm'
        'l'
        'k'
        'p'
        'q'
        
    [1] : https://arxiv.org/abs/1305.0470
    """

    SymGens = []
    
    if (SymType['name'] in ['C','D','Cp','Dp']):
        
        rot_angle =  twopi * SymType['p'] /  SymType['q']
        s = 1
        
        for ib_rel in range(len(ib_list)-1):
            SymGens.append(ChoreoSym(
                LoopTarget=ib_list[ib_rel+1],
                LoopSource=ib_list[ib_rel  ],
                SpaceRot = np.array([[s*np.cos(rot_angle),-s*np.sin(rot_angle)],[np.sin(rot_angle),np.cos(rot_angle)]],dtype=np.float64),
                TimeRev=1,
                TimeShift=fractions.Fraction(numerator=-SymType['m'],denominator=SymType['n'])
            ))

    if ((SymType['name'] == 'C') or (SymType['name'] == 'D')):
        
        rot_angle = twopi * SymType['l'] /  SymType['k']
        s = 1

        Sym = ChoreoSym(
            LoopTarget=ib_list[0],
            LoopSource=ib_list[0],
            SpaceRot = np.array([[s*np.cos(rot_angle),-s*np.sin(rot_angle)],[np.sin(rot_angle),np.cos(rot_angle)]],dtype=np.float64),
            TimeRev=1,
            TimeShift=fractions.Fraction(numerator=1,denominator=SymType['k'])
        )

        if not(Sym.IsIdentity()):
            SymGens.append(Sym)

    if (SymType['name'] == 'D'):
        
        rot_angle = 0
        s = -1

        SymGens.append(ChoreoSym(
            LoopTarget=ib_list[0],
            LoopSource=ib_list[0],
            SpaceRot = np.array([[s*np.cos(rot_angle),-s*np.sin(rot_angle)],[np.sin(rot_angle),np.cos(rot_angle)]],dtype=np.float64),
            TimeRev=-1,
            TimeShift=fractions.Fraction(numerator=0,denominator=1)
            ))
        
    if ((SymType['name'] == 'Cp') or ((SymType['name'] == 'Dp') and (SymType['k'] == 2))):
        
        rot_angle = 0
        s = -1

        SymGens.append(ChoreoSym(
            LoopTarget=ib_list[0],
            LoopSource=ib_list[0],
            SpaceRot = np.array([[s*np.cos(rot_angle),-s*np.sin(rot_angle)],[np.sin(rot_angle),np.cos(rot_angle)]],dtype=np.float64),
            TimeRev=1,
            TimeShift=fractions.Fraction(numerator=1,denominator=2)
            ))

    if (SymType['name'] == 'Dp'):
        
        rot_angle =  np.pi
        s = 1

        SymGens.append(ChoreoSym(
            LoopTarget=ib_list[0],
            LoopSource=ib_list[0],
            SpaceRot = np.array([[s*np.cos(rot_angle),-s*np.sin(rot_angle)],[np.sin(rot_angle),np.cos(rot_angle)]],dtype=np.float64),
            TimeRev=-1,
            TimeShift=fractions.Fraction(numerator=0,denominator=1)
            ))
    
    return SymGens

def Make2DChoreoSymManyLoops(nloop=None,nbpl=None,SymName=None,SymType=None):

    if nloop is None :
        if nbpl is None :
            raise ValueError("1")
        else:
            if isinstance(nbpl,list):
                nloop = len(nbpl)
            else:
                raise ValueError("2")
                
    else:
        if nbpl is None :
            raise ValueError("3")
        else:
            if isinstance(nbpl,int):
                nbpl = [ nbpl for il in range(nloop) ]
            elif isinstance(nbpl,list):
                    if nloop != len(nbpl):
                        raise ValueError("4")
            else:
                raise ValueError("5")

    the_lcm = m.lcm(*nbpl)

    if (SymType is None):
        
        SymType = []

        if (SymName is None):
            
            for il in range(nloop):

                SymType.append({
                    'name'  : 'C',
                    'n'     : the_lcm,
                    'm'     : 1,
                    'k'     : 1,
                    'l'     : 0 ,
                    'p'     : 0 ,
                    'q'     : 1 ,
                })

        elif isinstance(SymName,str):
            
            for il in range(nloop):
                    
                SymType.append({
                    'name'  : SymName,
                    'n'     : the_lcm,
                    'm'     : 1,
                    'k'     : 1,
                    'l'     : 0 ,
                    'p'     : 0 ,
                    'q'     : 1 ,
                })                    

        elif isinstance(SymName,list):
            
            for il in range(nloop):
                
                SymType.append({
                    'name'  : SymName[il],
                    'n'     : the_lcm,
                    'm'     : 1,
                    'k'     : 1,
                    'l'     : 0 ,
                    'p'     : 0 ,
                    'q'     : 1 ,
                })
        else:
            raise ValueError("6")

    elif (isinstance(SymType,dict)):
        
        SymType = [{
            'name'  : SymType['name'],
            'n'     : the_lcm,
            'm'     : SymType['m'],
            'k'     : SymType['k'],
            'l'     : SymType['l'] ,
            'p'     : SymType['p'] ,
            'q'     : SymType['q'],
        }
        for il in range(nloop)]

    elif (isinstance(SymType,list)):
        
        for il in range(nloop):
            SymType[il]['n'] = the_lcm
            
    else:
        raise ValueError("7")

    SymGens = []

    istart = 0
    for il in range(nloop):

        SymGens.extend(Make2DChoreoSym(SymType[il],[(i+istart) for i in range(nbpl[il])]))

        if (the_lcm//nbpl[il] != 1):

            SymGens.append(ChoreoSym(
                LoopTarget=istart,
                LoopSource=istart,
                SpaceRot = np.identity(2,dtype=np.float64),
                TimeRev=1,
                TimeShift=fractions.Fraction(numerator=-1,denominator=the_lcm//nbpl[il])
            ))
            
        istart += nbpl[il]

    nbody = istart
        
    return SymGens,nbody

def MakeSymFromGlobalTransform(Transform_list,Permutation_list,mass):

    assert (len(Transform_list) == len(Permutation_list))

    SymGens = []

    for i_transform in range(len(Transform_list)):

        for ibody in range(len(Permutation_list[i_transform])):

            if (mass[Permutation_list[i_transform][ibody]] != mass[ibody]):

                print("Warning: Mass is not uniform within loop")
            
            SymGens.append(
                ChoreoSym(       
                    LoopTarget=Permutation_list[i_transform][ibody],
                    LoopSource=ibody,
                    SpaceRot = Transform_list[i_transform].SpaceRot,
                    TimeRev=Transform_list[i_transform].TimeRev,
                    TimeShift=Transform_list[i_transform].TimeShift
                )
            )

    return SymGens

def MakeLoopEarSymList(n_main_loop,n_ears,m_main_loop=1,m_ears=1,SelfReflMain=False,SelfReflEar=False):
    
    Permutation_list = []
    Transform_list = []

    nbody = n_main_loop* (1 + n_ears)

    mass = np.array( [ (m_main_loop if i < n_main_loop else m_ears) for i in range(nbody)] , dtype=np.float64 )

    # --------------------------------------------
    # Loop symmetry

    the_lcm = m.lcm(n_main_loop,n_ears)
    delta_i_main_loop = the_lcm//n_ears
    delta_i_ears = the_lcm//n_main_loop

    perm = np.zeros((nbody),dtype=np.intp)
    for ibody in range(n_main_loop):
        perm[ibody] = ( (ibody+1) % n_main_loop) 

    for il in range(n_main_loop):
        for iear in range(n_ears):

            ibody = n_main_loop + il*n_ears +   iear
            jbody = n_main_loop + il*n_ears + ((iear + 1 ) % n_ears)

            perm[ibody] = jbody


    Permutation_list.append(perm)

    rot_angle = 2*np.pi * 0
    s = 1

    SpaceRot = np.array([[s*np.cos(rot_angle),-s*np.sin(rot_angle)],[np.sin(rot_angle),np.cos(rot_angle)]],dtype=np.float64)
    TimeRev = 1
    TimeShift=fractions.Fraction(numerator=1,denominator=the_lcm)

    Transform_list.append(
        ChoreoSym(
            SpaceRot=SpaceRot,
            TimeRev=TimeRev,
            TimeShift=TimeShift
        )
    )

    # --------------------------------------------
    # Reflexion symmetry

    if SelfReflMain:
        db_main = 0
    else:
        db_main = 1

    if SelfReflEar:
        db_ear = 0
    else:
        db_ear = 1        

    perm = np.zeros((nbody),dtype=np.intp)
    for ibody in range(n_main_loop):
        perm[ibody] = ( (n_main_loop + db_main - ibody) % n_main_loop) 

    for il in range(n_main_loop):
        for iear in range(n_ears):

            ibody = n_main_loop + il*n_ears + iear
          
            jl   = ( n_main_loop + db_main - il  ) % n_main_loop
            jear = ( n_ears      + db_ear  - iear) % n_ears      


            jbody = n_main_loop + jl*n_ears + jear

            perm[ibody] = jbody

    Permutation_list.append(perm)

    rot_angle = 2*np.pi * 0
    s = -1

    SpaceRot = np.array([[s*np.cos(rot_angle),-s*np.sin(rot_angle)],[np.sin(rot_angle),np.cos(rot_angle)]],dtype=np.float64)
    TimeRev = -1
    TimeShift=fractions.Fraction(numerator=0,denominator=1)

    Transform_list.append(
        ChoreoSym(
            SpaceRot=SpaceRot,
            TimeRev=TimeRev,
            TimeShift=TimeShift
        )
    )

    # --------------------------------------------
    # Rotational symmetry

    perm = np.zeros((nbody),dtype=np.intp)
    for ibody in range(n_main_loop):
        perm[ibody] = ( (ibody+1) % n_main_loop) 

    for il in range(n_main_loop):
        for iear in range(n_ears):

            ibody = n_main_loop + il*n_ears + iear
            jbody = n_main_loop + ( (il+1) % n_main_loop) * n_ears + iear

            perm[ibody] = jbody


    Permutation_list.append(perm)

    rot_angle = 2*np.pi * 1./ n_main_loop
    s = 1

    SpaceRot = np.array([[s*np.cos(rot_angle),-s*np.sin(rot_angle)],[np.sin(rot_angle),np.cos(rot_angle)]],dtype=np.float64)
    TimeRev = 1
    TimeShift=fractions.Fraction(numerator=0,denominator=1)

    Transform_list.append(
        ChoreoSym(
            SpaceRot=SpaceRot,
            TimeRev=TimeRev,
            TimeShift=TimeShift
        )
    )

    Sym_list = MakeSymFromGlobalTransform(Transform_list,Permutation_list,mass=mass)

    return Sym_list,nbody,mass

def Make_SymList_From_InfoDict(InfoDict,Transform_Sym=None):

    if Transform_Sym is None:
        Transform_Sym = ChoreoSym()


    SymList = []

    for il in range(InfoDict['nloop']):

        Transform_Sym.LoopTarget=InfoDict["Targets"][il][0]
        Transform_Sym.LoopSource=InfoDict["Targets"][il][0]

        Sym_ref = Transform_Sym.Compose(ChoreoSym(
            LoopTarget = InfoDict["Targets"][il][0],
            LoopSource = -1,
            SpaceRot = np.array(InfoDict["SpaceRotsUn"][il][0]),
            TimeRev = InfoDict["TimeRevsUn"][il][0],
            TimeShift=fractions.Fraction(numerator=InfoDict["TimeShiftNumUn"][il][0],denominator=InfoDict["TimeShiftDenUn"][il][0])
        )).Inverse()


        for ibl in range(InfoDict["loopnb"][il]):
            
            Sym_rel = ChoreoSym(
                LoopTarget = InfoDict["Targets"][il][ibl],
                LoopSource = -1,
                SpaceRot = np.array(InfoDict["SpaceRotsUn"][il][ibl]),
                TimeRev = InfoDict["TimeRevsUn"][il][ibl],
                TimeShift=fractions.Fraction(numerator=InfoDict["TimeShiftNumUn"][il][ibl],denominator=InfoDict["TimeShiftDenUn"][il][ibl])
            ).Compose(Sym_ref)

                    
            Transform_Sym.LoopTarget=InfoDict["Targets"][il][ibl]
            Transform_Sym.LoopSource=InfoDict["Targets"][il][ibl]

            SymList.append(Transform_Sym.Compose(Sym_rel))

    return SymList

def MakeTargetsSyms(Info_dict_slow,Info_dict_fast_list):
    # In targets, a loop is created for each slow loop and fast body in it

    assert Info_dict_slow['nloop'] == len(Info_dict_fast_list)

    SymList_slow = Make_SymList_From_InfoDict(Info_dict_slow)

    SymList_Target = []
    mass_Target = []

    ibody_targ = 0

    il_slow_source = []
    ibl_slow_source = []
    il_fast_source = []
    ibl_fast_source = []

    for il_slow in range(Info_dict_slow['nloop']):

        Info_dict_fast = Info_dict_fast_list[il_slow]
        SymList_fast = Make_SymList_From_InfoDict(Info_dict_fast)
        nbody_fast = Info_dict_fast['nbody']

        ibody_targ_ref = ibody_targ

        for ibl_slow in range(Info_dict_slow["loopnb"][il_slow]):

            ib_slow = Info_dict_slow["Targets"][il_slow][ibl_slow]
            Sym_Slow = SymList_slow[ib_slow]

            mass_slow = Info_dict_slow["mass"][ib_slow]

            mass_fast_list = []
            
            for il_fast in range(Info_dict_fast['nloop']):

                for ibl_fast in range(Info_dict_fast["loopnb"][il_fast]):

                    ib_fast = Info_dict_fast["Targets"][il_fast][ibl_fast]
                    mass_fast = Info_dict_fast["mass"][ib_fast]

                    mass_fast_list.append(mass_fast)

                    Sym_targ = ChoreoSym(
                        LoopTarget = ibody_targ ,
                        LoopSource = ibody_targ_ref + ib_fast,
                        SpaceRot = Sym_Slow.SpaceRot,
                        TimeRev = Sym_Slow.TimeRev,
                        TimeShift = Sym_Slow.TimeShift,
                    )

                    SymList_Target.append(Sym_targ)
                    
                    il_slow_source.append(il_slow)
                    ibl_slow_source.append(ibl_slow)
                    il_fast_source.append(il_fast)
                    ibl_fast_source.append(ibl_fast)

                    ibody_targ += 1

            mass_fast_tot = sum(mass_fast_list)

            for ib_fast in range(nbody_fast):
                mass_fast_list[ib_fast] = mass_fast_list[ib_fast] * (mass_slow / mass_fast_tot)

            mass_Target.extend(mass_fast_list)

    mass_Target = np.array(mass_Target)
    il_slow_source = np.array(il_slow_source)
    ibl_slow_source = np.array(ibl_slow_source)
    il_fast_source = np.array(il_fast_source)
    ibl_fast_source = np.array(ibl_fast_source)

    return SymList_Target, mass_Target,il_slow_source,ibl_slow_source,il_fast_source,ibl_fast_source
