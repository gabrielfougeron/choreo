import os

import sys,argparse
import random
import numpy as np
import math as m

import copy
import shutil
import time
import builtins

from choreo.Choreo_scipy_plus import *
from choreo.Choreo_funs import *
from choreo.Choreo_plot import *


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
        
        SymGens.append(ChoreoSym(
            LoopTarget=ib_list[0],
            LoopSource=ib_list[0],
            SpaceRot = np.array([[s*np.cos(rot_angle),-s*np.sin(rot_angle)],[np.sin(rot_angle),np.cos(rot_angle)]],dtype=np.float64),
            TimeRev=1,
            TimeShift=fractions.Fraction(numerator=1,denominator=SymType['k'])
        ))

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
                
        SymGens.append(ChoreoSym(
            LoopTarget=istart,
            LoopSource=istart,
            SpaceRot = np.identity(ndim,dtype=np.float64),
            TimeRev=1,
            TimeShift=fractions.Fraction(numerator=-1,denominator=the_lcm//nbpl[il])
        ))
        
        istart += nbpl[il]
        
    nbody = istart
        
    return SymGens,nbody

def MakeSymFromGlobalTransform(Transform_list,Permutation_list,mass = None):

    assert (len(Transform_list) == len(Permutation_list))
    
    if mass is None:
        mass = np.ones((len(Transform_list)),dtype=np.float64)

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

def MakeLoopEarSymList(n_main_loop,n_ears,m_main_loop=1,m_ears=1):


    Permutation_list = []
    Transform_list = []

    nbody = n_main_loop* (1 + n_ears)

    mass = np.array( [ (m_main_loop if i < n_main_loop else m_ears) for i in range(nbody)] , dtype=np.float64 )

    # --------------------------------------------
    # Loop symmetry

    the_lcm = m.lcm(n_main_loop,n_ears)
    delta_i_main_loop = the_lcm//n_ears
    delta_i_ears = the_lcm//n_main_loop

    perm = np.zeros((nbody),dtype=int)
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

    db = 1

    perm = np.zeros((nbody),dtype=int)
    for ibody in range(n_main_loop):
        perm[ibody] = ( (n_main_loop + db - ibody) % n_main_loop) 

    for il in range(n_main_loop):
        for iear in range(n_ears):

            ibody = n_main_loop + il*n_ears + iear
          
            jl = (n_main_loop + db - il) % n_main_loop
            jear = ( (n_ears     - iear) % n_ears)


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

    perm = np.zeros((nbody),dtype=int)
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