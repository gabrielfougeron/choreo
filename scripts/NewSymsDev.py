import os
import sys
__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(__PROJECT_ROOT__)

import choreo 

# choreo.run.GUI_in_CLI(['-f', os.path.join('.', 'NewSym_data', '3')])
# choreo.run.GUI_in_CLI(['-f', os.path.join('.', 'NewSym_data', '3D')])
# choreo.run.GUI_in_CLI(['-f', os.path.join('.', 'NewSym_data', '3q')])
# choreo.run.GUI_in_CLI(['-f', os.path.join('.', 'NewSym_data', '3Dl1q2')])
# choreo.run.GUI_in_CLI(['-f', os.path.join('.', 'NewSym_data', '2C3C')])
# choreo.run.GUI_in_CLI(['-f', os.path.join('.', 'NewSym_data', '2D1')])
# choreo.run.GUI_in_CLI(['-f', os.path.join('.', 'NewSym_data', 'test')])
# choreo.run.GUI_in_CLI(['-f', os.path.join('.', 'NewSym_data', '4C5k')])
# choreo.run.GUI_in_CLI(['-f', os.path.join('.', 'NewSym_data', '4D3k')])
# choreo.run.GUI_in_CLI(['-f', os.path.join('.', 'NewSym_data', '4C')])
# choreo.run.GUI_in_CLI(['-f', os.path.join('.', 'NewSym_data', '4D')])
# choreo.run.GUI_in_CLI(['-f', os.path.join('.', 'NewSym_data', '6Dk5')])
choreo.run.GUI_in_CLI(['-f', os.path.join('.', 'NewSym_data', 'overconstrained')])