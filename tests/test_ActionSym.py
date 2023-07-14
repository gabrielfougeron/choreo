import os
import sys

__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(__PROJECT_ROOT__)

import pytest
from test_config import *

import choreo

def test_small(float64_tols):
    assert 0 < float64_tols.atol

# def test_big(float64_tols):
#     assert 1 < float64_tols.atol
