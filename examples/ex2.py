"""
Test
=========================

Sphinx-Gallery is capable of transforming Python files into reST files
with a notebook structure. For this to be used you need to respect some syntax
rules. This example demonstrates how to alternate text and code blocks and some
edge cases. It was designed to be compared with the"""

import numpy as np

print(np.identity(3))


np.identity(4)

# import matplotlib.pyplot as plt
# fig=plt.figure()
# _ = plt.plot(np.array(range(10)))
# plt.tight_layout()
# plt.show()