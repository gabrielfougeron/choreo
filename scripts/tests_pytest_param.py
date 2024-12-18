import os
import sys

__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(__PROJECT_ROOT__)

import tests_cp

# tests.test_integration_tables.test_ImplicitSymmetricPairs
# 

fun = tests_cp.test_ODE.test_Implicit_ODE
ptm = fun.pytestmark

# print(dir(fun))

print(fun.__doc__)

print()
print()
print("--------------------------------------------------------------------")
print()
print()


print(dir())