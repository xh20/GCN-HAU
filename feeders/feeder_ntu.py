import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
import sys

sys.path.extend(['../'])
from feeders import tools_ntu


class Feeder(Dataset):
    def __init__(self, data_path, label_path, test_a=False, test_b=False, is_empty=False, add_noise=False,
                 noise_intensity=0.1, noise_type='gaussian', ignore_labels=None,
                 timer_label_path=None,
                 random_choose=False, random_shift=False, random_move=False,
                 window_size=-1, normalization=False, debug=False, num_point=26, use_mmap=True):
        """
        
        :param data_path: 
        :param label_path: 
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the beginning or end of sequence
        :param random_move: 
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        """

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        # for original bimacs
        self.test_a = test_a
        # for ikea dataset, it test_b is True
        self.test_b = test_b
        # for empty dataset
        self.is_empty = is_empty
        self.ignore_labels = ignore_labels
        # the selected actions of ikea dataset
        self.selected_actoion = [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
        # for noised dataset
        self.add_noise = add_noise
        self.noise_intensity = noise_intensity
        self.noise_type = noise_type

        self.timer_label_path = timer_label_path
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.num_point = num_point
        self.use_mmap = use_mmap
        self.load_data()
        self.timer = None
        if normalization:
            self.get_mean_map()
        if self.add_noise:
            self.add_noise_to_data()

    def add_noise_to_data(self):
        N, C, T, V, M = self.data.shape
        # np.random.seed(0)
        if self.noise_type == 'gaussian':
            var_x = np.var(self.data[:, 0, :, :, :])
            var_y = np.var(self.data[:, 1, :, :, :])
            var_z = np.var(self.data[:, 2, :, :, :])
            mean_ = [0, 0, 0]
            cov_ = np.dot([var_x, var_y, var_z], self.noise_intensity)
            covM = [[cov_[0], 0, 0], [0, cov_[1], 0], [0, 0, cov_[2]]]
            noise = np.random.multivariate_normal(mean_, covM, size=(N, V, T, M)).transpose(0, 4, 2, 1, 3)
            self.data = noise + self.data
        elif self.noise_type == 'pepper':
            data = self.data.copy()
            size = int(self.noise_intensity * V)
            for b in range(N):
                for t in range(T):
                    joint_index = np.random.default_rng().choice(V, size, replace=False)
                    data[b, :, t, joint_index, :] = 0
            self.data = data
        elif self.noise_type == 'poisson':
            noise = np.random.poisson(lam=self.noise_intensity * 1000, size=self.data.shape)
            self.data = noise + self.data

    def load_data(self):
        # data: N C V T M
        try:
            with open(self.label_path) as f:
                sample_name, label = pickle.load(f)
        except:
            # for pickle file from python2
            with open(self.label_path, 'rb') as f:
                sample_name, label = pickle.load(f)
        try:
            self.label = np.array(label).astype(int)
            self.sample_name = sample_name
        except:
            self.label = np.array(sample_name).astype(int)
            self.sample_name = label
        if len(self.label.shape) < 2:
            self.label = self.label[None, :]
        self.label = self.label.tolist()

        if self.ignore_labels is not None:
            label_array = np.array(self.label)
            for i_label in self.ignore_labels:
                label_array[label_array == i_label] = -1
            self.label = list(label_array)

        if self.test_b:
            try:
                label = np.array(self.label).astype(int)
            except:
                label = np.array(self.sample_name).astype(int)
            if len(label.shape) < 2:
                label = label[None, :]
            label_b = np.zeros(label.shape, dtype=int) - 1
            index_i = 0
            for c in self.selected_actoion:
                label_b[label == c] = index_i
                index_i += 1
            if len(label_b.shape) <= 1:
                label_b = label_b[None, :]
            self.label = label_b.tolist()

        if self.timer_label_path is not None:
            try:
                with open(self.timer_label_path) as f:
                    self.sample_name, self.timer = pickle.load(f)
            except:
                # for pickle file from python2
                with open(self.timer_label_path, 'rb') as f:
                    self.sample_name, self.timer = pickle.load(f, encoding='latin1')

        # load data
        if self.use_mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

        if len(self.data.shape) == 3:
            self.data = self.data[None, :, :, :, None]

        data = self.data.copy()
        B, C, T, V, M = self.data.shape

        if self.test_b and V != self.num_point and not self.is_empty and not self.test_a:
            self.data = np.zeros((B, 3, T, self.num_point, M))
            # mid_joint = 0.5*data[:, :, :, 5, :] + 0.5*data[:, :, :, 6, :]
            # data = data - mid_joint[:, :, :, None, :]
            if C <= 3:
                self.data[:, :C, :, :V, :] = data
            else:
                self.data[:, :3, :, :V, :] = data[:, :3, :, :, :]
            if len(self.label) != T:
                return -1
        elif self.is_empty and not self.test_b and not self.test_a:
            self.data = np.zeros((B, 3, T, self.num_point, M))
        elif self.test_a and not self.test_b and not self.is_empty:
            pass

        if self.debug:
            if self.timer is not None:
                self.timer = self.timer[0:100]
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        timer = None
        if self.timer is not None:
            timer = self.timer[index]
        data_numpy = np.array(data_numpy)
        label_numpy = np.array(label)

        if self.normalization:
            data_numpy = (data_numpy - self.mean_map) / self.std_map
        if self.random_shift:
            data_numpy = tools_ntu.random_shift(data_numpy)
        if self.random_choose:
            data_numpy = tools_ntu.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools_ntu.auto_padding(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools_ntu.random_move(data_numpy)
        if timer is not None:
            return data_numpy, label, timer, index
        else:
            return data_numpy, label, index

    def top_k_ntu(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)

    def acc_seg(self, prediction, topk=1, label=None):
        if label is None:
            label = np.array(self.label, dtype=int).squeeze()

        if isinstance(label, list):
            label = np.array(label, dtype=int).squeeze()
        if isinstance(prediction, list):
            prediction = np.array(prediction, dtype=int).squeeze()
        else:
            prediction = prediction.squeeze()
        if len(prediction) != len(label):
            print("the length of prediction is unequal to labels")
            exit(-1)
        pred_correct = np.sum((prediction == label))
        pred_count = np.sum(label >= 0)
        if pred_count == 0:
            return 0.0
        # pred_count = np.sum(prediction >= 0)
        # hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        # return sum(hit_top_k) * 1.0 / len(hit_top_k)
        return pred_correct * 1.0 / pred_count

    def acc_time(self, time_pre):
        time_pre = np.squeeze(time_pre)
        pred_time_correct = np.sum((time_pre == self.timer))
        pred_count = np.sum(self.timer >= 0)
        return pred_time_correct * 1.0 / pred_count


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def test(data_path, label_path, vid=None, graph=None, is_3d=False):
    '''
    vis the samples using matplotlib
    :param data_path: 
    :param label_path: 
    :param vid: the id of sample
    :param graph: 
    :param is_3d: when vis NTU, set it True
    :return: 
    '''
    import matplotlib.pyplot as plt
    loader = torch.utils.data.DataLoader(
        dataset=Feeder(data_path, label_path),
        batch_size=64,
        shuffle=False,
        num_workers=2)

    if vid is not None:
        sample_name = loader.dataset.sample_name
        sample_id = [name.split('.')[0] for name in sample_name]
        index = sample_id.index(vid)
        data, label, index = loader.dataset[index]
        data = data.reshape((1,) + data.shape)

        # for batch_idx, (data, label) in enumerate(loader):
        N, C, T, V, M = data.shape

        plt.ion()
        fig = plt.figure()
        if is_3d:
            from mpl_toolkits.mplot3d import Axes3D
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)

        if graph is None:
            p_type = ['b.', 'g.', 'r.', 'c.', 'm.', 'y.', 'k.', 'k.', 'k.', 'k.']
            pose = [
                ax.plot(np.zeros(V), np.zeros(V), p_type[m])[0] for m in range(M)
            ]
            ax.axis([-1, 1, -1, 1])
            for t in range(T):
                for m in range(M):
                    pose[m].set_xdata(data[0, 0, t, :, m])
                    pose[m].set_ydata(data[0, 1, t, :, m])
                fig.canvas.draw()
                plt.pause(0.001)
        else:
            p_type = ['b-', 'g-', 'r-', 'c-', 'm-', 'y-', 'k-', 'k-', 'k-', 'k-']
            import sys
            from os import path
            sys.path.append(
                path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
            G = import_class(graph)()
            edge = G.inward
            pose = []
            for m in range(M):
                a = []
                for i in range(len(edge)):
                    if is_3d:
                        a.append(ax.plot(np.zeros(3), np.zeros(3), p_type[m])[0])
                    else:
                        a.append(ax.plot(np.zeros(2), np.zeros(2), p_type[m])[0])
                pose.append(a)
            ax.axis([-1, 1, -1, 1])
            if is_3d:
                ax.set_zlim3d(-1, 1)
            for t in range(T):
                for m in range(M):
                    for i, (v1, v2) in enumerate(edge):
                        x1 = data[0, :2, t, v1, m]
                        x2 = data[0, :2, t, v2, m]
                        if (x1.sum() != 0 and x2.sum() != 0) or v1 == 1 or v2 == 1:
                            pose[m][i].set_xdata(data[0, 0, t, [v1, v2], m])
                            pose[m][i].set_ydata(data[0, 1, t, [v1, v2], m])
                            if is_3d:
                                pose[m][i].set_3d_properties(data[0, 2, t, [v1, v2], m])
                fig.canvas.draw()
                # plt.savefig('/home/lshi/Desktop/skeleton_sequence/' + str(t) + '.jpg')
                plt.pause(0.01)


if __name__ == '__main__':
    import os

    os.environ['DISPLAY'] = 'localhost:10.0'
    data_path = "../data/ntu/xview/val_data_joint.npy"
    label_path = "../data/ntu/xview/val_label.pkl"
    graph = 'graph.ntu_rgb_d.Graph'
    test(data_path, label_path, vid='S004C001P003R001A032', graph=graph, is_3d=True)
    # data_path = "../data/kinetics/val_data.npy"
    # label_path = "../data/kinetics/val_label.pkl"
    # graph = 'graph.Kinetics'
    # test(data_path, label_path, vid='UOD7oll3Kqo', graph=graph)
