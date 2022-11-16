import Speed_test

import pstats, cProfile
 
cProfile.runctx("Speed_test.main()", globals(), locals(), "Profile.prof")
#
s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()



