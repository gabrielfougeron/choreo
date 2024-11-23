import os
import sys

import numpy as np
import math as m
import json
import choreo
import itertools
import pyquickbench
import networkx
from matplotlib import pyplot as plt


__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.append(__PROJECT_ROOT__)

Workspace_folder = os.path.join(__PROJECT_ROOT__, "Sniff_all_sym")

params_filename = os.path.join(Workspace_folder, "choreo_config.json")
with open(params_filename) as jsonFile:
    params_dict = json.load(jsonFile)

geodim = params_dict['Phys_Gen']['geodim']
nbody = params_dict["Phys_Bodies"]["nbody"]
Sym_list = choreo.find.ChoreoLoadSymList(params_dict)

# Sym_list.pop()

print(Sym_list)

Graph = choreo.BuildCayleyGraph(nbody, geodim, GeneratorList = Sym_list)

print(Graph)


fig, ax = plt.subplots()
pos = networkx.spring_layout(Graph)

networkx.draw(Graph, pos)
networkx.draw_networkx_labels(Graph, pos)



plt.axis('off')
fig.tight_layout()

plt.savefig(os.path.join(Workspace_folder,"Cayley_Graph.pdf"))
plt.close()