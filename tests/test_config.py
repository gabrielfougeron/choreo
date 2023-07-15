import attrs
import pytest
import inspect
import typing

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

@attrs.define
class dimension:
    all_geodims:    list[int]

@attrs.define
class nbody:
    all_nbodies:    list[int]

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

@pytest.fixture
def TwoD_only():
    return dimension(
        all_geodims = [2] ,
    )

@pytest.fixture
def Physical_dims():
    return dimension(
        all_geodims = [2, 3] ,
    )


@pytest.fixture
def Few_bodies():
    return nbody(
        all_nbodies = [2, 3, 4, 5] ,
    )
