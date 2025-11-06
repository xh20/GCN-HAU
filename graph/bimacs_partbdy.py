import numpy as np
import sys

sys.path.extend(['../'])
from graph import tools
import networkx as nx
import matplotlib.pyplot as plt

# Node index:
    # Body joint nodes:
    # {0,  "Nose"}
    # {1,  "Neck"},
    # {2,  "RShoulder"},
    # {3,  "RElbow"},
    # {4,  "RWrist"},
    # {5,  "MidHip"}
    # {6,  "RHip"},
    # {7,  "LHip"},
    # {8,  "REye"},
    # {9,  "LEye"},
    # {10, "REar"},
    # {11, "LEar"},
    # Object nodes:
    # {12, "bowl"},
    # {13, "knife"},
    # {14, "screwdriver"},
    # {15, "cuttingboard"},
    # {16, "whisk"},
    # {17, "hammer"},
    # {18, "bottle"},
    # {19, "cup"},
    # {20, "banana"},
    # {21, "cereals"},
    # {22, "sponge"},
    # {23, "woodenwedge"},
    # {24, "saw"},
    # {25, "harddrive"},

# Edge format: (origin, neighbor)
num_joint = 12
num_object = 14

inward = [(4, 3), (3, 2), (2, 1), (0, 1), (5, 1), (6, 5), (7, 5), (10, 8), (8, 0), (11, 9), (9, 0)]

# ob2hand = [(i, 4) for i in range(num_joint, num_node)] #right hands & objects
# inward += ob2hand

#for i in range(num_joint, num_node):
    #for j in range(num_joint):
        #inward.append((i, j))

outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class Graph:
    def __init__(self, labeling_mode='spatial', num_node=None):
        self.num_node = num_node
        self.self_link = [(i, i) for i in range(num_node)]

        self.A = self.get_adjacency_matrix(labeling_mode)
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(self.num_node, self.self_link, inward, outward)
            # A += 1e-6
        else:
            raise ValueError()
        return A


if __name__ == '__main__':
    A = Graph('spatial').get_adjacency_matrix()
    for i in range(3):
        plt.matshow(A[i])
        plt.savefig('../A/initial_bimacs_partbody_{}.png'.format(i))
    # f = './A0.npy'
    # print(A.shape)
    # np.save(f, A)
