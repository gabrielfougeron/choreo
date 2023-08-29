'''
ODE.py : Defines ODE-related things I designed I feel ought to be in scipy.

'''

import math as m
import numpy as np

from choreo.scipy_plus.cython.SegmQuad import QuadFormula
from choreo.scipy_plus.cython.ODE import ExplicitSymplecticRKTable


#####################
# EXPLICIT RK STUFF #
#####################

SymplecticEuler = ExplicitSymplecticRKTable(
    c_table = np.array([1.])    ,
    d_table = np.array([1.])    ,
    th_cvg_rate = 1             ,
)

StormerVerlet = ExplicitSymplecticRKTable(
    c_table = np.array([0.    ,1.      ])   ,
    d_table = np.array([1./2  ,1./2    ])   ,
    th_cvg_rate = 2                         ,
)


Ruth3 = ExplicitSymplecticRKTable(
    c_table = np.array([1.        ,-2./3  ,2/3    ])    ,
    d_table = np.array([-1./24    , 3./4  ,7./24  ])    ,
    th_cvg_rate = 3                                     ,
)

McAte3 = ExplicitSymplecticRKTable(
    c_table = np.array([0.2683300957817599  ,-0.18799161879915982  , 0.9196615230173999     ])  ,
    d_table = np.array([0.9196615230173999 , -0.18799161879915982   ,0.2683300957817599     ])  ,
    th_cvg_rate = 3                                                                             ,
)

curt2 = m.pow(2,1./3)
Ruth4 = ExplicitSymplecticRKTable(
    c_table = np.array([1./(2*(2-curt2))  ,(1-curt2)/(2*(2-curt2))    ,(1-curt2)/(2*(2-curt2))    ,1./(2*(2-curt2))   ])    ,
    d_table = np.array([1./(2-curt2)      ,-curt2/(2-curt2)           ,1./(2-curt2)               ,0.                 ])    ,
    th_cvg_rate = 4                                                                                                         ,
)

Ruth4Rat = ExplicitSymplecticRKTable(
    c_table = np.array([0.     , 1./3  , -1./3     , 1.        , -1./3 , 1./3  ])   ,
    d_table = np.array([7./48  , 3./8  , -1./48    , -1./48    ,  3./8 , 7./48 ])   ,
    th_cvg_rate = 4                                                                 ,
)

McAte4 = ExplicitSymplecticRKTable(
    c_table = np.array([0.128846158365384185    ,  0.441583023616466524 ,-0.085782019412973646  , 0.515352837431122936  ])  ,
    d_table = np.array([ 0.334003603286321425   , 0.756320000515668291  , -0.224819803079420806 ,0.134496199277431089   ])  ,
    th_cvg_rate = 4                                                                                                         ,
)
