import os
import sys
__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(__PROJECT_ROOT__)

import pstats

prof_file = os.path.join(__PROJECT_ROOT__, 'prof.out')

p = pstats.Stats(prof_file)
p.sort_stats('cumulative').print_stats(100)