'''Pre-computed tables for ODE solvers

Order 1
=======

.. autosummary::
    :toctree: _generated/
    :caption: Order 1
    :nosignatures:

    SymplecticEuler
    
Order 2
=======
    
.. autosummary::
    :toctree: _generated/
    :caption: Order 2
    :nosignatures:

    StormerVerlet
    McAte2    
    
Order 3
=======
    
.. autosummary::
    :toctree: _generated/
    :caption: Order 3
    :nosignatures:
    
    Ruth3
    McAte3
    
Order 4
=======
    
.. autosummary::
    :toctree: _generated/
    :caption: Order 4
    :nosignatures:
    
    Ruth4
    Ruth4Rat
    McAte4
    CalvoSanz4
    
Order 5
=======
    
.. autosummary::
    :toctree: _generated/
    :caption: Order 5
    :nosignatures:
    
    McAte5
    
Order 6
=======
    
.. autosummary::
    :toctree: _generated/
    :caption: Order 5
    :nosignatures:
    
    Yoshida6A
    Yoshida6B
    Yoshida6C
    KahanLi6  
    
Order 8
=======
    
.. autosummary::
    :toctree: _generated/
    :caption: Order 8
    :nosignatures:
    
    McLahan8
    KahanLi8
    Yoshida8A
    Yoshida8B
    Yoshida8C
    Yoshida8D
    Yoshida8E
  
Order 10
========
    
.. autosummary::
    :toctree: _generated/
    :caption: Order 10
    :nosignatures:
    
    SofSpa10

'''

import math as m
import numpy as np

from choreo.segm.cython.quad        import QuadTable
from choreo.segm.cython.ODE         import ExplicitSymplecticRKTable
from choreo.segm.multiprec_tables   import Yoshida_w_to_cd, Yoshida_w_to_cd_reduced

#####################
# EXPLICIT RK STUFF #
#####################

# Order 1

SymplecticEuler = ExplicitSymplecticRKTable(
    c_table = np.array([1.])    ,
    d_table = np.array([1.])    ,
    th_cvg_rate = 1             ,
)
"""Symplectic Euler"""

# Order 2

StormerVerlet = ExplicitSymplecticRKTable(
    c_table = np.array([0.    ,1.      ])   ,
    d_table = np.array([1./2  ,1./2    ])   ,
    th_cvg_rate = 2                         ,
)
"""Störmer-Verlet """

StormerVerlet_noopt = ExplicitSymplecticRKTable(
    c_table = np.array([0.    ,1.      ])   ,
    d_table = np.array([1./2  ,1./2    ])   ,
    th_cvg_rate = 2                         ,
    OptimizeFGunCalls = False               ,
)
"""Störmer-Verlet """

sq2s2 = m.sqrt(2)/2
McAte2 = ExplicitSymplecticRKTable(
    c_table = np.array([1. - sq2s2  , sq2s2     ])  ,
    d_table = np.array([sq2s2       , 1.-sq2s2  ])  ,
    th_cvg_rate = 2  ,
)
"""McLachlan-Atela method of order 2

Optimal method with 2 stages by McLachlan and Atela :footcite:`mclachlan1992accuracy`.

:cited:
.. footbibliography::

"""

# Order 3

Ruth3 = ExplicitSymplecticRKTable(
    c_table = np.array([1.        ,-2./3  ,2/3    ])    ,
    d_table = np.array([-1./24    , 3./4  ,7./24  ])    ,
    th_cvg_rate = 3                                     ,
)
""" Ruth method of order 3

Cf :footcite:`ruth1983canonical`.

:cited:
.. footbibliography::

"""

McAte3 = ExplicitSymplecticRKTable(
    c_table = np.array([0.2683300957817599  ,-0.18799161879915982  , 0.9196615230173999     ])  ,
    d_table = np.array([0.9196615230173999 , -0.18799161879915982   ,0.2683300957817599     ])  ,
    th_cvg_rate = 3                                                                             ,
)
"""McLachlan-Atela method of order 3

Optimal method with 3 stages by McLachlan and Atela :footcite:`mclachlan1992accuracy`.

:cited:
.. footbibliography::

"""

# Order 4

curt2 = m.pow(2,1./3)
Ruth4 = ExplicitSymplecticRKTable(
    c_table = np.array([1./(2*(2-curt2))  ,(1-curt2)/(2*(2-curt2))    ,(1-curt2)/(2*(2-curt2))    ,1./(2*(2-curt2))   ])    ,
    d_table = np.array([1./(2-curt2)      ,-curt2/(2-curt2)           ,1./(2-curt2)               ,0.                 ])    ,
    th_cvg_rate = 4                                                                                                         ,
)
"""Ruth and Forrest method of order 4

Cf :footcite:`forest1990fourth`.

:cited:
.. footbibliography::

"""

Ruth4Rat = ExplicitSymplecticRKTable(
    c_table = np.array([0.     , 1./3  , -1./3     , 1.        , -1./3 , 1./3  ])   ,
    d_table = np.array([7./48  , 3./8  , -1./48    , -1./48    ,  3./8 , 7./48 ])   ,
    th_cvg_rate = 4                                                                 ,
)
""" Rational Ruth method of order 4"""

McAte4 = ExplicitSymplecticRKTable(
    c_table = np.array([0.128846158365384185    ,  0.441583023616466524 ,-0.085782019412973646  , 0.515352837431122936  ])  ,
    d_table = np.array([ 0.334003603286321425   , 0.756320000515668291  , -0.224819803079420806 ,0.134496199277431089   ])  ,
  
    th_cvg_rate = 4                                                                                                         ,
)
"""McLachlan-Atela method of order 4

Optimal method with 4 stages by McLachlan and Atela :footcite:`mclachlan1992accuracy`.

:cited:
.. footbibliography::
"""

CalvoSanz4 = ExplicitSymplecticRKTable(
    c_table = np.array([0.0             , 0.512721933192410 ,-0.12092087633891  ,0.403021281604210  , 0.205177661542290 ])  ,
    d_table = np.array([0.12501982279453,-0.14054801465937  , 0.61479130717558  ,0.33897802655364   , 0.061758858135626 ])  ,
    th_cvg_rate = 4                                                                                                         ,
)
"""Calvo and Sanz-Serna method of order 4

Cf :footcite:`sanz1993symplectic`.

:cited:
.. footbibliography::
"""

# Order 5

McAte5 = ExplicitSymplecticRKTable(
    c_table = np.array([
        0.4423637942197494587   ,
        0.3235807965546976394   ,
        -0.603039356536491888   ,
        0.5858564768259621188   ,
        -0.088601336903027329   ,
        0.339839625839110000    ,
    ]),
    d_table = np.array([
        -0.0589796254980311632  ,
        0.0107050818482359840   ,
        0.4012695022513534480   ,
        -0.1713123582716007754  ,
        0.6989273703824752308   ,
        0.1193900292875672758   ,
    ]),
    th_cvg_rate = 5             ,
)
"""McLachlan-Atela method of order 5

Optimal method with 6 stages by McLachlan and Atela :footcite:`mclachlan1992accuracy`.

:cited:
.. footbibliography::
"""

# Order 6

Yoshida6A = Yoshida_w_to_cd(np.array([
    -1.17767998417887   ,
    0.235573213359      ,
    0.784513610477      ,
]),
    th_cvg_rate = 6     ,
)
"""Method A of order 6 by Yoshida

First published in :footcite:`yoshida1990construction`.

:cited:
.. footbibliography::
"""

Yoshida6B = Yoshida_w_to_cd(np.array([
    -2.13228522200144   ,
    0.426068187079180e-2,
    1.4398481679767     ,
]),
    th_cvg_rate = 6     ,
)
"""Method B of order 6 by Yoshida

First published in :footcite:`yoshida1990construction`.

:cited:
.. footbibliography::
"""
    
Yoshida6C = Yoshida_w_to_cd(np.array([
    0.152886228424922e-2    ,
    - 2.14403531630539      ,
    1.44778256239930        ,
]),
    th_cvg_rate = 6         ,
)
"""Method C of order 6 by Yoshida

First published in :footcite:`yoshida1990construction`.

:cited:
.. footbibliography::
"""
    
KahanLi6 = Yoshida_w_to_cd(np.array([
    0.79854399093482996339895035    , 
    0.08221359629355080023149045    ,
    -0.70624617255763935980996482   ,
    0.33259913678935943859974864    ,
    0.39216144400731413927925056    ,
]),
    th_cvg_rate = 6     ,
).symmetric_adjoint()
""" Method s9odr6a by Kahan and Li

cf :footcite:`kahan1997composition`.

:cited:
.. footbibliography::
"""    

# Order 8

McLahan8 = Yoshida_w_to_cd_reduced(np.array([
    -0.79688793935291635401978884   ,    
    0.31529309239676659663205666    ,
    0.33462491824529818378495798    ,
    0.29906418130365592384446354    ,
    -0.57386247111608226665638773   ,
    0.19075471029623837995387626    ,
    -0.40910082580003159399730010   ,
    0.74167036435061295344822780    ,
]),
    th_cvg_rate = 8                 ,
).symmetric_adjoint()
""" McLachlan SS method of order 8 and error 0.14

cf :footcite:`mclachlan1995numerical`.

:cited:
.. footbibliography::
"""

KahanLi8 = Yoshida_w_to_cd_reduced(np.array([
    -0.60550853383003451169892108   ,    
    0.29501172360931029887096624    ,    
    0.25837438768632204729397911    ,    
    0.18453964097831570709183254    ,    
    -0.39590389413323757733623154   ,    
    0.15884190655515560089621075    ,    
    -0.38947496264484728640807860   ,
    0.56116298177510838456196441    ,        
    0.13020248308889008087881763    ,
]),
    th_cvg_rate = 8                     ,
).symmetric_adjoint()
""" Method s17odr8a by Kahan and Li

cf :footcite:`kahan1997composition`.

:cited:
.. footbibliography::
"""

Yoshida8A =  Yoshida_w_to_cd(np.array([
    -0.161582374150097e1    ,
    -0.244699182370524e1    ,
    -0.716989419708120e-2   ,
    0.244002732616735e1     ,
    0.157739928123617e0     ,
    0.182020630970714e1     ,
    0.104242620869991e1     ,
]),
    th_cvg_rate = 8         ,
)
"""Method A of order 8 by Yoshida

First published in :footcite:`yoshida1990construction`.

:cited:
.. footbibliography::
"""

Yoshida8B =  Yoshida_w_to_cd(np.array([
    -0.169248587770116e-2   ,
    0.289195744315849e1     ,
    0.378039588360192e-2    ,
    -0.289688250328827e1    ,
    0.289105148970595e1     ,
    -0.233864815101035e1    ,
    0.148819229202922e1     ,
]),
    th_cvg_rate = 8         ,
)
"""Method B of order 8 by Yoshida

First published in :footcite:`yoshida1990construction`.

:cited:
.. footbibliography::
"""

Yoshida8C =  Yoshida_w_to_cd(np.array([
    0.311790812418427       ,
    -0.155946803821447e1    ,
    -0.167896928259640e1    ,
    0.166335809963315e1     ,
    -0.106458714789183e1    ,
    0.136934946416871e1     ,
    0.629030650210433e0     ,
]),
    th_cvg_rate = 8         ,
)
"""Method C of order 8 by Yoshida

First published in :footcite:`yoshida1990construction`.

:cited:
.. footbibliography::
"""

Yoshida8D =  Yoshida_w_to_cd(np.array([
    0.102799849391985E0     ,
    -0.196061023297549e1    ,
    0.193813913762276e1     ,   
    -0.158240635368243e0    ,
    -0.144485223686048e1    ,
    0.253693336566229e0     ,   
    0.914844246229740e0     ,
]),
    th_cvg_rate = 8         ,
)
"""Method D of order 8 by Yoshida

First published in :footcite:`yoshida1990construction`.

:cited:
.. footbibliography::
"""

Yoshida8E =  Yoshida_w_to_cd(np.array([
    0.227738840094906e-1    ,
    0.252778927322839e1     ,
    -0.719180053552772e-1   ,
    0.536018921307285e-2    ,
    -0.204809795887393e1    ,
    0.107990467703699e0     , 
    0.130300165760014e1     ,
]),
    th_cvg_rate = 8         ,
)
"""Method E of order 8 by Yoshida

First published in :footcite:`yoshida1990construction`.

:cited:
.. footbibliography::
"""

# Order 10

SofSpa10 = Yoshida_w_to_cd_reduced(np.array([
    0.04931773575959453791768001   ,
    0.04967437063972987905456880   ,
    0.05066509075992449633587434   ,
    0.05194250296244964703718290   ,
    -0.39203335370863990644808194  ,
    -0.00486636058313526176219566  ,
    0.41143087395589023782070412   ,
    0.10308739852747107731580277   ,
    -0.39910563013603589787862981  ,
    0.36613344954622675119314812   ,
    0.11199342399981020488957508   ,
    0.07497334315589143566613711   ,
    -0.26973340565451071434460973  ,
    0.13096206107716486317465686   ,
    -0.22959284159390709415121340  ,
    0.02791838323507806610952027   ,
    0.31309610341510852776481247   ,
    0.07879572252168641926390768   ,   
]),
    th_cvg_rate = 10                ,
).symmetric_adjoint()
"""Sofroniou and Spaletta method of order 10

Cf :footcite:`sofroniou2005derivation`

:cited:
.. footbibliography::
"""