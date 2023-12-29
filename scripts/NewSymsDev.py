import os
import sys
__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(__PROJECT_ROOT__)

import choreo 
import time


import cProfile
import pstats

def main():
        
    all_tests = [
        '3q',
        # '2C3C',
        # '2C3C5k',
        # '2D3D',
        # '2D3D5k',
        # '2D1',
        # 'test',
        # '4C5k',
        # '4D3k',
        # '4C',
        # '4D',
        # '3',
        # '3D',
        # '3C2k',
        # '3D2k',
        # '3C4k',
        # '3D4k',
        # '3C5k',
        # '3D5k',
        # '3C101k',
        # '3D101k',
        # 'test_3D5k',
        # '3C7k2',
        # '3D7k2',
        # '6C',
        # '6D',
        # '6Ck5',
        # '6Dk5',
        # '5Dq',
        # '2C3C5C',
    ]

    tbeg_loop = time.perf_counter()
    for test in all_tests:
        print()
        print("  OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO  ")
        print()
        print(test)
        print()

        tbeg = time.perf_counter()

        choreo.run.GUI_in_CLI(['-f', os.path.join('.', 'NewSym_data', test)])
        
        tend = time.perf_counter()

        print(f'Elapsed time : {tend-tbeg} s')
        
    tend_loop = time.perf_counter()
    
    print()
    print(f'Total elapsed time : {tend_loop-tbeg_loop} s')
# 

if __name__ == "__main__":
    main()


# 
# if __name__ == "__main__":
# 
#     with cProfile.Profile() as pr:
# 
#         main()
#         
#         stats = pstats.Stats(pr)
#         stats.sort_stats(pstats.SortKey.TIME)
#         # stats.print_stats()
#         stats.dump_stats(filename='prof.prof')
