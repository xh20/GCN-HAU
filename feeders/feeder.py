import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
import sys
from feeders import tools
from pathlib import Path
import os
from tqdm import tqdm
from scipy.ndimage import zoom
import cv2
sys.path.extend(['../'])
from utils.visualize_skeleton import plot_3d_skeleton
from PIL import Image
import json
import torch.nn.functional as F


class Feeder(Dataset):
    def __init__(self, data_path, image_path=None, label_path=None, add_noise=False,
                 noise_intensity=0.1, noise_type='gaussian', ignore_labels=None,
                 split=None, dataset="behave", num_class=14,
                 smooth_label=False, kernel=None, kernel_size=None, image_stream=True, image_difference=False,
                 optical_flow=False, downsample_factor=1,
                 random_choose=False, random_shift=False, random_move=False, use_ram=False,
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
        self.color = {}
        self.debug = debug
        self.data_path = data_path
        self.image_path = image_path
        self.label_path = label_path
        self.split = split
        self.ignore_labels = ignore_labels
        self.dataset = dataset
        self.downsample_factor = downsample_factor
        self.use_ram = use_ram
        # for noised dataset
        self.add_noise = add_noise
        self.noise_intensity = noise_intensity
        self.noise_type = noise_type
        self.num_class = num_class
        self.smooth_label = smooth_label
        self.kernel = kernel
        self.kernel_size = kernel_size
        # for image stream
        self.image_stream = image_stream
        self.image_difference = image_difference
        self.optical_flow = optical_flow

        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.num_point = num_point
        self.use_mmap = use_mmap
        self.image_dirs = []
        self.path = []
        self.load_data()
        self.timer = None

        if normalization:
            self.get_mean_map()
        if self.add_noise:
            if self.split == "test":
                self.add_noise_to_testset()
            else:
                self.add_noise_to_data()

    def add_noise_to_testset(self):
        for i, data in enumerate(self.data):
            C, T, V, M = data.shape
            size = int(self.noise_intensity * V)
            if self.noise_type == 'pepper':
                for t in range(T):
                    joint_index = np.random.default_rng().choice(V, size, replace=False)
                    data[:, t, joint_index, :] = 0
                self.data[i] = data

    def add_noise_to_data(self):
        if isinstance(self.data, np.ndarray):
            data = self.data
        elif isinstance(self.data, np.lib.npyio.NpzFile):
            data = np.array(list(self.data.values()))
        else:
            data = self.data
        N, C, T, V, M = data.shape
        if self.noise_type == 'gaussian':
            var_x = np.var(data[:, 0, :, :, :])
            var_y = np.var(data[:, 1, :, :, :])
            var_z = np.var(data[:, 2, :, :, :])
            mean_ = [0, 0, 0]
            cov_ = np.dot([var_x, var_y, var_z], self.noise_intensity)
            covM = [[cov_[0], 0, 0], [0, cov_[1], 0], [0, 0, cov_[2]]]
            noise = np.random.multivariate_normal(mean_, covM, size=(N, V, T, M)).transpose(0, 4, 2, 1, 3)
            self.data = noise + data
        elif self.noise_type == 'pepper':
            size = int(self.noise_intensity * V)
            for b in range(N):
                for t in range(T):
                    joint_index = np.random.default_rng().choice(V, size, replace=False)
                    data[b, :, t, joint_index, :] = 0
            self.data = data
        elif self.noise_type == 'poisson':
            noise = np.random.poisson(lam=self.noise_intensity * 1000, size=self.data.shape)
            self.data = noise + self.data

    def smooth_label_with_kernel(self, label, kernel, kernel_size=5, sigma=2):
        if isinstance(label, list):
            label = np.array(label, dtype=int)
        elif isinstance(label, np.ndarray):
            label = label.astype(int)
        label = label.squeeze()
        B, T = label.shape
        label_tensor = torch.from_numpy(label)
        # B x T x C -> B x C x T
        label_onehot = F.one_hot(label_tensor, num_classes=self.num_class).float().permute(0, 2, 1)
        smoothed_label = torch.clone(label_onehot)
        if kernel == "linear":
            kernel = torch.ones(self.num_class, 1, kernel_size) / kernel_size  # Moving average of size k
            smoothed_label = F.conv1d(label_onehot, kernel, padding=kernel_size // 2, groups=self.num_class)
        elif kernel == "gaussian":
            kernel_size = int(2 * np.ceil(3 * sigma) + 1)
            kernel = torch.arange(kernel_size).float() - (kernel_size - 1) / 2
            kernel = torch.exp(-kernel ** 2 / (2 * sigma ** 2))
            kernel = kernel / kernel.sum()  # Normalize kernel
            kernel = kernel.unsqueeze(0).unsqueeze(0)  # Shape (1, 1, kernel_size)
            kernel = kernel.expand(self.num_class, 1, -1)  # Expand to (num_channels, 1, kernel_size)
            smoothed_label = F.conv1d(label_onehot, kernel, padding=kernel_size // 2, groups=self.num_class)

        return smoothed_label

    def read_images_from_list(self, paths):
        images = []
        if isinstance(paths, list):
            for path in paths:
                if self.dataset == "ikea":
                    image = Image.open(Path(self.image_path) / Path(path))
                    W, H = image.size
                    image = np.array(image.resize((W//4, H//4)))  # 4, 4
                elif self.dataset == "bimacts":
                    image = np.array(Image.open(Path(path)))
                image = image[:, :, :3]
                images.append(image)
        else:
            if self.dataset == "ikea":
                image = np.array(Image.open(Path(self.image_path) / Path(paths)))
            elif self.dataset == "bimacts":
                image = np.array(Image.open(Path(paths)))
            image = image[:, :, :3]
            images.append(image)

        # Images: C x T x H x W
        try:
            images = np.stack(images, axis=0).transpose(3, 0, 1, 2)
            return images
        except:
            raise ("error paths \n", paths)

    def load_batch_images(self, batches):
        image_batch = []
        for paths in tqdm(batches):
            images = self.read_images_from_list(paths)
            image_batch.append(images)
        image_batch = np.stack(image_batch, axis=0)
        return image_batch

    def load_train_val_data(self):
        print("loading bimacts train and val data and label ...")
        # load label
        if "npz" in self.data_path:
            try:
                data_all = np.load(self.data_path, allow_pickle=True)
                self.keys = list(data_all.keys())
                # B T V C 1 => B C T V 1
                self.data = data_all["data"]
                if self.data.shape[1] != 3 and self.data.shape[1] != 2:
                    self.data = self.data[:, :, :, :3, None].transpose(0, 3, 1, 2, 4)
                N, C, T, V, _ = self.data.shape

                label = data_all["label"]
                self.label = np.array(label).squeeze()

                if "path" in self.keys and self.image_stream:
                    step = self.downsample_factor
                    half_step = step // 2
                    self.path = data_all["path"][:, half_step:T:step]
                    if self.use_ram:
                        self.color[f"{self.split}"] = self.load_batch_images(self.path.tolist())

                sample_name = None
            except:
                raise ValueError("cannot load bimacts data")

        if self.ignore_labels is not None:
            if isinstance(self.label, list) or isinstance(self.label, np.ndarray):
                label_array = np.array(self.label)
                for i_label in self.ignore_labels:
                    label_array[label_array == i_label] = -1
                # self.label = list(label_array)
                self.label = label_array
            elif isinstance(self.label, np.lib.npyio.NpzFile):
                for key, label in self.label.items():
                    if label in self.ignore_labels:
                        self.label[key] = -1

        if self.smooth_label and self.kernel is not None and self.kernel_size is not None:
            self.label = self.smooth_label_with_kernel(self.label, self.kernel, self.kernel_size)

    def load_bimacts_test_data(self):
        self.data = []
        self.label = []
        self.names = []
        self.path = []
        self.color[f"{self.split}"] = []

        all_files = []
        if not Path(self.data_path).exists():
            raise FileNotFoundError(self.data_path)
        all_files.extend(Path(self.data_path).glob("*.json"))
        for file in tqdm(all_files, desc="loading bimacts test data"):
            with open(file, "r") as f:
                data = json.load(f)
            self.keys = list(data.keys())
            joints = np.array(data["nodes"]).transpose(2, 0, 1)[:3, :, :, None]
            label = data["label"]
            self.data.append(joints)
            self.label.append(label)
            self.names.append(file.name)
            if "color_path" in self.keys and self.image_stream:
                C, T, V, _ = joints.shape
                step = self.downsample_factor
                half_step = step // 2
                path = data["color_path"][half_step:T:step]
                if not isinstance(path, list):
                    path = path.tolist()
                self.path.append(path)
                if self.use_ram:
                    self.color[f"{self.split}"].append(self.read_images_from_list(path))

    def load_ikea_test_data(self):
        self.data = []
        self.label = []
        self.image_dirs = []
        self.names = []
        self.path = []
        self.color[f"{self.split}"] = []

        root_dir = Path(self.data_path)
        if not root_dir.exists():
            raise FileNotFoundError(self.data_path)
        for npz_files in tqdm(root_dir.rglob("data.npz"), desc="loading ikea test data"):
            with np.load(npz_files) as data:
                self.keys = list(data.keys())
                # 1 x C x T x V x 1
                joints = np.array(data["data"]).squeeze(0)
                # T,
                label = data["label"]
                label_name = data["label_name"]
                self.data.append(joints)
                self.label.append(label)
                self.names.append(npz_files.parent.name)
                if "path" in self.keys and self.image_stream:
                    C, T, V, _ = joints.shape
                    step = self.downsample_factor
                    half_step = step // 2
                    path = data["path"][half_step:T:step]
                    if not isinstance(path, list):
                        path = path.tolist()
                    self.path.append(path)
                    if self.use_ram:
                        # C x T x H x W
                        self.color[f"{self.split}"].append(self.read_images_from_list(path))
                        # print("done")

    def load_data(self):
        if self.split == "train" or self.split == "val":
            self.load_train_val_data()
        elif self.split == "test":
            if self.dataset == "bimacts":
                self.load_bimacts_test_data()
            elif self.dataset == "ikea":
                self.load_ikea_test_data()
            else:
                raise ValueError("dataset not supported")
        else:
            raise ValueError("split not supported")

    def get_mean_map(self):
        if isinstance(self.data, np.ndarray):
            data = self.data
        elif isinstance(self.data, np.lib.npyio.NpzFile):
            data = np.array(list(self.data.values()))
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        if isinstance(self.label, list) or isinstance(self.label, np.ndarray) or isinstance(self.label, torch.Tensor):
            return len(self.label)
        elif isinstance(self.label, np.lib.npyio.NpzFile):
            return len(self.keys)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        output = {}
        if self.split == "train":
            data_numpy = self.data[index]
            label = self.label[index]
            timer = None
            images = None
            masked_images = None
            flow = None
            if self.timer is not None:
                timer = self.timer[index]
            # load image
            if self.image_stream:
                if not self.use_ram:
                    if "color" in self.keys:
                        images = self.color[f"{self.split}"][index]
                        masked_images = None
                    elif "path" in self.keys:
                        images = self.read_images_from_list(self.path[index].tolist())
                else:
                    images = self.color[f"{self.split}"][index]

            data_numpy = np.array(data_numpy)
            C, T, V, _ = data_numpy.shape
            if self.normalization:
                data_numpy = (data_numpy - self.mean_map) / (self.std_map + 1e-5)
            if self.random_shift:
                data_numpy = tools.random_shift(data_numpy)
            if self.random_choose:
                data_numpy = tools.random_choose(data_numpy, self.window_size)
            elif self.window_size:
                data_numpy = tools.auto_padding(data_numpy, self.window_size)
            if self.random_move:
                data_numpy = tools.random_move(data_numpy)
            output = {"motion": data_numpy,
                      "label": label,
                      "index": index,
                      }
            if timer is not None:
                output["time"] = timer
            if images is not None:
                output["image"] = images
            if masked_images is not None:
                output["masked_image"] = masked_images

        elif self.split == "val":
            data_numpy = self.data[index]
            label = self.label[index]
            output = {"motion": data_numpy,
                      "label": label,
                      "index": index}
            # load image
            if self.image_stream:
                if not self.use_ram:
                    if "path" in self.keys:
                        image_paths = [str(image_i) for image_i in self.path[index]]  # List of image paths
                        images = self.read_images_from_list(image_paths)
                        output["image"] = images
                else:
                    output["image"] = self.color[f"{self.split}"][index]

        elif self.split == "test":
            data_numpy = self.data[index]
            label = self.label[index]
            name = self.names[index]
            output = {"motion": data_numpy,
                      "label": label,
                      "name": name,
                      "index": index}
            if self.image_stream:
                if not self.use_ram:
                    if self.image_dirs:
                        color = np.load(self.image_dirs[index], allow_pickle=True)
                        images, masked_images = color["images_"], color["masked_images_"]
                        output["image"] = images
                        output["masked_image"] = masked_images
                    elif "path" in self.keys or "color_path" in self.keys:
                        image_paths = [str(image_i) for image_i in self.path[index]]  # List of image paths
                        output["image"] = self.read_images_from_list(image_paths)
                else:
                    output["image"] = self.color[f"{self.split}"][index]

        return output

    def top_k_ntu(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)

    def acc_seg(self, predictions, topk=1, label=None):
        if label is None:
            label = self.label
        if isinstance(label, list):
            label = np.concatenate(label, axis=0).squeeze()
        if isinstance(predictions, list):
            prediction = np.concatenate(predictions, axis=0).squeeze()
        else:
            prediction = predictions.squeeze()
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

    ''' 
    
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
    '''


if __name__ == '__main__':

    bimacs_train_feeders = {
        "split": "train",
        "num_class": 14,
        "dataset": "bimacts",
        "smooth_label": True,
        "image_stream": True,
        "use_ram": False,
        "kernel": "gaussian",
        "kernel_size": 5,
        "downsample_factor": 30,
        "random_choose": True,
        "random_move": True,
        "data_path": "/media/hao/data_base/BimanualActions/generated_data/frameseg_120_30_5/train.npz",
        "debug": False,
    }

    bimacs_val_feeders = {
        "split": "val",
        "dataset": "bimacts",
        "smooth_label": True,
        "image_stream": True,
        "use_ram": False,
        "kernel": "gaussian",
        "kernel_size": 5,
        "downsample_factor": 30,
        "data_path": "/media/hao/data_base/BimanualActions/generated_data/frameseg_120_30_1/val.npz",
        "debug": False
    }

    bimacs_test_feeders = {
        "split": "test",
        "dataset": "bimacts",
        "image_stream": True,
        "use_ram": False,
        "downsample_factor": 30,
        "add_noise": True,
        "noise_type": "pepper",
        "noise_intensity": 0.05,
        "data_path": "/media/hao/data_base/BimanualActions/generated_data/subject_1",
        "debug": False
    }

    ikea_train_feeders = {
        "split": "train",
        "num_class": 32,
        "dataset": "ikea",
        "smooth_label": False,
        "image_stream": True,
        "use_ram": False,
        "kernel": "gaussian",
        "kernel_size": 5,
        "downsample_factor": 30,
        "random_choose": True,
        "random_move": True,
        "image_path": "/media/hao/data_base/IKEA/ikea_asm_dataset_RGB_top/ANU_ikea_dataset_video/",
        "data_path": "/media/hao/data_base/IKEA/processed/xenv_train_2d_data_joint.npz",
        "debug": False,
    }

    ikea_val_feeders = {
        "split": "val",
        "dataset": "ikea",
        "smooth_label": False,
        "use_ram": False,
        "kernel": "gaussian",
        "kernel_size": 5,
        "image_stream": True,
        "downsample_factor": 30,
        "image_path": "/media/hao/data_base/IKEA/ikea_asm_dataset_RGB_top/ANU_ikea_dataset_video/",
        "data_path": "/media/hao/data_base/IKEA/processed/xenv_test_2d_data_joint.npz",
        "debug": False,
    }

    ikea_test_feeders = {
        "split": "test",
        "dataset": "ikea",
        "image_stream": True,
        "use_ram": False,
        "downsample_factor": 30,
        "add_noise": False,
        "noise_type": "pepper",
        "noise_intensity": 0.00,
        "image_path": "/media/hao/data_base/IKEA/ikea_asm_dataset_RGB_top/ANU_ikea_dataset_video/",
        "data_path": "/media/hao/data_base/IKEA/processed/test/",
        "debug": False
    }

    loader = torch.utils.data.DataLoader(
        dataset=Feeder(**bimacs_test_feeders),
        batch_size=1,
        shuffle=False,
        num_workers=12,
    )

    process = tqdm(loader)
    ## Behave dataset
    # num_joint = 22
    # edges = [(0, 1), (0, 2), (0, 3), (1, 4), (4, 7), (7, 10), (2, 5), (5, 8), (8, 11), (3, 6), (6, 9), (9, 12),
    #          (12, 15), (9, 13), (13, 16), (16, 18), (18, 20), (9, 14), (14, 17), (17, 19), (19, 21)]
    ## Bimacs:
    edges = [(4, 3), (3, 2), (2, 1), (0, 1), (5, 1), (6, 5), (7, 5), (10, 8), (8, 0), (11, 9), (9, 0)]
    num_joint = 12
    num_object = 14
    num_node = 26
    for batch_idx, data in enumerate(process):
        # plot_3d_skeleton(data["motion"], num_joint, edges, save_dir=None)
        print(data["motion"].shape)
        print(data["image"].shape)
        # print(data["name"])
        print(batch_idx)

