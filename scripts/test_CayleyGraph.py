import os
import sys

import numpy as np
import math as m
import json
import choreo
import itertools
import pyquickbench
import networkx

import matplotlib
# matplotlib.rcParams['backend'] = 'WXAgg' 

from matplotlib import pyplot as plt
import matplotlib.animation as animation 


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
pos = networkx.spring_layout(Graph, dim=2)
# pos = networkx.spectral_layout(Graph, dim=2)
networkx.draw(Graph, pos)
networkx.draw_networkx_labels(Graph, pos)

plt.axis('off')
fig.tight_layout()

plt.savefig(os.path.join(Workspace_folder,"Cayley_Graph.pdf"))
plt.close()




from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.text import Annotation

class Annotation3D(Annotation):
    '''Annotate the point xyz with text s'''

    def __init__(self, s, xyz, *args, **kwargs):
        Annotation.__init__(self,s, xy=(0,0), *args, **kwargs)
        self._verts3d = xyz        

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.xy=(xs,ys)
        Annotation.draw(self, renderer)
        
def annotate3D(ax, s, *args, **kwargs):

    tag = Annotation3D(s, *args, **kwargs)
    ax.add_artist(tag)

pos = networkx.spring_layout(Graph, dim=3, iterations=1000)

labels = [v for v in Graph]
nodes = np.array([pos[v] for v in Graph])
edges = np.array([(pos[u], pos[v]) for u, v in Graph.edges()])

def init():
    ax.scatter(*nodes.T, alpha=0.2, s=100, color="blue")
    for vizedge in edges:
        ax.plot(*vizedge.T, color="gray")
        
    for j, xyz_ in enumerate(nodes): 
        annotate3D(ax, s=labels[j], xyz=xyz_, fontsize=10, xytext=(-3,3),
                textcoords='offset points', ha='right',va='bottom') 
    ax.grid(False)
    ax.set_axis_off()
    plt.tight_layout()
    return

def _frame_update(index):
    ax.view_init(index * 0.2, index * 0.5)
    return


nx = 1920
ny = 1080
dpi = 100
figsize = (nx/dpi, ny/dpi)

fig = plt.figure(figsize=figsize, dpi=dpi)
ax = fig.add_subplot(111, projection="3d")

framerate = 60

anim = animation.FuncAnimation(
    fig,
    _frame_update,
    init_func=init,
    interval=1000./framerate,
    cache_frame_data=False,
    frames=1000,
)

writervideo = animation.FFMpegWriter(fps=framerate) 
anim.save(os.path.join(Workspace_folder,'Cayley_Graph_3D.mp4'), writer=writervideo) 
plt.close() 