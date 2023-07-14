import attrs
import pytest
import inspect

@attrs.define
class float_tol:
    atol: float
    rtol: float

@attrs.define
class repeat:
    small:  int
    medium: int
    big:    int

@attrs.define
class likelyhood():
    probable:       float
    not_unlikely:   float
    uncommon:       float
    unlikely:       float
    unbelievable:   float
    impossible:     float

@pytest.fixture
def float64_tols():
    return float_tol(
        atol = 1e-12,
        rtol = 1e-10,
    )

@pytest.fixture
def float32_tols():
    return float_tol(
        atol = 1e-5,
        rtol = 1e-3,
    )

@pytest.fixture
def nonstiff_float64_likelyhood():
    return likelyhood(
        probable        = 1e-1 ,
        not_unlikely    = 1e-3 ,
        uncommon        = 1e-5 ,
        unlikely        = 1e-8 ,
        unbelievable    = 1e-12,
        impossible      = 1e-16,
    )

