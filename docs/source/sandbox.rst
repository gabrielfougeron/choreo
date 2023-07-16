Sandbox
=======



.. nbplot::

    >>> import networkx as nx
    >>> import matplotlib.pyplot as plt

To test if the import of `~networkx.drawing.nx_pylab` was successful draw ``G``
using one of

.. nbplot::

    >>> G = nx.petersen_graph()
    >>> subax1 = plt.subplot(121)
    >>> nx.draw(G, with_labels=True, font_weight='bold')
    >>> subax2 = plt.subplot(122)
    >>> nx.draw_shell(G, nlist=[range(5, 10), range(5)], with_labels=True, font_weight='bold')

when drawing to an interactive display.  Note that you may need to issue a
Matplotlib

>>> plt.show()