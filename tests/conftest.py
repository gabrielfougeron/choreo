import os
import choreo_GUI
import test_config

def pytest_sessionstart(session):

    choreo_GUI.install_official_gallery()
