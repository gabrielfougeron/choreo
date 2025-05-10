import os
import sys
import choreo_GUI
import pytest
import threadpoolctl
import warnings

from pytest_timeout import _get_item_settings, SESSION_TIMEOUT_KEY # BAAAAD !!! Works wit timeout-2.3.1, might not work with other version

from . import test_config
import choreo

def pytest_sessionstart(session):
    choreo_GUI.install_official_gallery()
    threadpoolctl.threadpool_limits(limits=1).__enter__()
    
    try:
        choreo.segm.cython.test.AssertFalse()
        warnings.warn("The package choreo was compiled with flag CYTHON_WITHOUT_ASSERTIONS", stacklevel=2)
    except AssertionError:
        pass
    

def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")

def robust_min(*args):
    
    res = sys.float_info.max
    for item in args:
        
        try:
            if res > item:
                res = item
        except:
            pass
    
    return res

def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return

    for item in items:
        if "slow" in item.keywords:
            
            timeout_settings = _get_item_settings(item)
            timeout_config = item.session.config.stash[SESSION_TIMEOUT_KEY]
            if timeout_config == 0:
                timeout_config = sys.float_info.max
            
            timeout = robust_min(timeout_settings.timeout, timeout_config)
            
            slow_marker = item.get_closest_marker(name="slow")
            required_time = slow_marker.kwargs.get("required_time", sys.float_info.max)

            if required_time > timeout:
                item.add_marker(pytest.mark.skip(reason=f"Test marked slow with {required_time = }, but {timeout = }. Run with --runslow CLI option or increase --timeout."))
