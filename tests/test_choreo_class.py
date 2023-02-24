import os

import shutil
import random
import time
import math as m
import numpy as np
import scipy.linalg
import sys
import fractions


__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(__PROJECT_ROOT__)

import choreo 

a_list = ['a','b','c','d']
b_list = ['e','f','g','h']
not_a_list_ = 123

test_action = choreo.ChoreoAction(a_cvg_lvl_list=a_list,not_a_list_=not_a_list_,current_cvg_lvl=0)

test_action.current_cvg_lvl = 2

print(test_action.a)

test_action_2 = choreo.ChoreoAction(a_cvg_lvl_list=b_list,not_a_list_=not_a_list_,current_cvg_lvl=0)


print(test_action_2.a)
