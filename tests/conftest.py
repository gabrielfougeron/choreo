import os
import choreo
import test_config

def pytest_sessionstart(session):
    
    if not os.path.isdir(test_config.gallery_dir):
        choreo.GUI.install_official_gallery(test_config.gallery_root)
