import sys

sys.path.extend(['../'])
from graph import tools
import matplotlib.pyplot as plt

# num_node = 24
num_node = 20
num_objects = 7
num_human = 13
# Node index:
    # Body joint nodes:
    # {0,  "Nose"}
    # {1,  "LEye"},
    # {2,  "REye"},
    # {3,  "LEar"},
    # {4,  "REar"},
    # {5,  "LShoulder"},
    # {6,  "RShoulder"},
    # {7,  "LElbow"},
    # {8,  "RElbow"}
    # {9,  "LWirst"},
    # {10, "RWirst"},
    # {11, "LHip"},
    # {12, "RHip"},
    # Objects:
    # {13, "table_top"},
    # {14, "leg"},
    # {15, "shelf"},
    # {16, "side_panel"},
    # {17, "front_panel"},
    # {18, "bottom_panel"},
    # {19, "rear_panel"}

self_link = [(i, i) for i in range(num_node)]
inward = [(0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
          (5, 11), (6, 12), (11, 12), (9, 13), (10, 13), (9, 14), (10, 14), (9, 15), (10, 15),
          (9, 16), (10, 16), (9, 17), (10, 17), (9, 18), (10, 18), (9, 19), (10, 19)]

outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
            # A += 1e-6
        else:
            raise ValueError()
        return A


if __name__ == '__main__':

    A = Graph('spatial').get_adjacency_matrix()
    # for i in range(3):
    #     plt.matshow(A[i])
    #     plt.savefig('../A/initial_{}.png'.format(i))
    for i in A:
        plt.imshow(i, cmap='gray')
        plt.show()
    print(A)
    # f = './A0.npy'
    # print(A.shape)
    # np.save(f, A)
