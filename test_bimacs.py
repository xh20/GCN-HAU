#!/usr/bin/env python
from __future__ import print_function

import argparse
import inspect
import itertools
import os
from pathlib import Path
import pickle
import random
import shutil
import math
import time
import json
from collections import OrderedDict
from sklearn.metrics import accuracy_score
# for plotting
import matplotlib.pyplot as plt
import seaborn as sb
from matplotlib import rc

import numpy as np
from sklearn.metrics import f1_score

# torch
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from torchsummary import summary
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.optim.lr_scheduler import LRScheduler
from tqdm import tqdm
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from utils.framewise_loss import SegmentationLoss, SmoothLoss


def init_seed(_):
    torch.cuda.manual_seed_all(1)
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='Spatial Temporal Graph Convolution Network')
    parser.add_argument(
        '--work-dir',
        default='./work_dir/temp',
        help='the work folder for storing results')
    parser.add_argument(
        '--results-dir',
        default='./result',
        help='the work folder for storing results')
    parser.add_argument(
        '--log_name',
        default='log.json',
        help='the log save name')
    parser.add_argument('-result_save_name', default='')
    parser.add_argument(
        '--config',
        default='./config/nturgbd-cross-view/test_bone.yaml',
        help='path to the configuration file')
    parser.add_argument(
        '--phase', default='test', help='must be test')
    parser.add_argument(
        '--show-topk',
        type=int,
        default=[1],
        nargs='+',
        help='which Top K accuracy will be shown')
    parser.add_argument(
        '--dataset',
        default='bimacs',
        help='which dataset, ikea or bimacs')
    parser.add_argument(
        '--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument(
        '--test-feeder-args',
        default=dict(),
        help='the arguments of data loader for test')
    parser.add_argument(
        '--train-feeder-args',
        default=dict(),
        help='the arguments of data loader for test')
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument(
        '--model-args',
        type=dict,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--weight_dir',
        default=None,
        help='the folder of weights')
    parser.add_argument(
        '--weights',
        type=str,
        default=None,
        help='the weights for network initialization')
    parser.add_argument(
        '--weights_all',
        type=list,
        default=None,
        help='the weights for network initialization')
    parser.add_argument(
        '--gauss_model_dir',
        type=str,
        default='/media/dataset/Human_Object_Interaction_Segmentation/segmentation/gauss_model/',
        help='gaussian model dir')
    parser.add_argument(
        '--gauss_mean',
        type=str,
        default='gauss_mean.npy',
        help='the gaussian model mean array')
    parser.add_argument(
        '--gauss_cov',
        type=str,
        default='gauss_cov.npy',
        help='the gaussian model covariance array')
    parser.add_argument(
        '--test_noise',
        type=bool,
        default=False,
        help='test noised data?')
    parser.add_argument(
        '--noise_type',
        type=str,
        default='pepper',
        help='the type of noise')
    parser.add_argument(
        '--noise_intensity',
        type=float,
        default=0.5,
        help='the type of noise')
    parser.add_argument(
        '--test_empty',
        type=bool,
        default=False,
        help='test empty data?')
    parser.add_argument(
        '--test_ikea',
        type=bool,
        default=False,
        help='test ikea data?')

    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')
    parser.add_argument(
        '--batch-size', type=int, default=1, help='test batch size')
    parser.add_argument(
        '--test-batch-size', type=int, default=1, help='test batch size')
    parser.add_argument(
        '--subject', type=int, default=1, help='test subject')
    parser.add_argument(
        '--num_moment', type=int, default=2, help='the number of moments')
    parser.add_argument(
        '--runs', type=int, default=1, help='the number of moments')
    parser.add_argument('--model_type', type=str, default='gauss', help='the model type, ensemble or not')
    return parser


class seg_cache():
    def __init__(self):
        self.label = 0
        self.start = 0
        self.end = 0

    def toseg(self):
        return ([self.label, self.start, self.end])


def framewiselabel2seglabel(seg_list):
    if isinstance(seg_list, torch.Tensor):
        seg_list = list(seg_list.squeeze().tolist())
    seg = []
    cache = seg_cache()
    cache.label = seg_list[0]
    cache.start = 0
    frame_index = 0
    for i in seg_list:
        if i != cache.label:
            cache.end = frame_index
            seg.append(cache.toseg())
            cache.label = i
            cache.start = frame_index
        elif frame_index == len(seg_list) - 1:
            cache.end = frame_index
            seg.append(cache.toseg())
        frame_index += 1
    return seg


def f1_macro_weighted(gt_segs, pr_segs):
    gt_list = np.concatenate(gt_segs, axis=0)
    pr_list = np.concatenate(pr_segs, axis=0)
    # gt = torch.cat(gt_segs, dim=1)
    # pr = torch.cat(pr_segs, dim=1)
    return f1_score(gt_list, pr_list, average='weighted')


def f1_overlap(ratio, n_class, gt_segs, pr_segs):

    TP = np.zeros(n_class, np.float32)
    FP = np.zeros(n_class, np.float32)
    FN = np.zeros(n_class, np.float32)

    for a in range(n_class):
        n_true = 0
        for pr, gt in zip(pr_segs, gt_segs):
            for seg in gt:
                if seg[0] != a:
                    continue
                else:
                    n_true += 1
                    gt_start = seg[1]
                    gt_end = seg[2]
                    tp_flag = 0
                    for segp in pr:
                        pr_label = segp[0]
                        pr_start = segp[1]
                        pr_end = segp[2]
                        if pr_end < gt_start:
                            continue
                        if pr_start > gt_end:
                            break
                        if pr_label == a:
                            space = [gt_start, gt_end, pr_start, pr_end]
                            space.sort()
                            iou = (space[2]-space[1]+1)/(space[3]-space[0]+1)
                            if iou >= ratio and tp_flag == 0:
                                TP[a] += 1
                                tp_flag = 1
                            else:
                                FP[a] += 1
        FN[a] = n_true - TP[a]

    TP_sum = np.sum(TP).item()
    FP_sum = np.sum(FP).item()
    FN_sum = np.sum(FN).item()
    if TP_sum + FP_sum == 0.0:
        p = 0.0
    else:
        p = TP_sum / (TP_sum + FP_sum)

    if TP_sum + FN_sum == 0.0:
        r = 0.0
    else:
        r = TP_sum / (TP_sum + FN_sum)

    if (p+r) == 0.0:
        f = 0.0
    else:
        f = 2*p*r/(p+r)
    print('f1@{0:.2f}: {1:.3%}'.format(ratio, f))
    return f


class Processor():
    """ 
        Processor for Skeleton-based Action Recgnition
        embeddings and labels save all features and labels that are used to build gaussian model in gmm_fit function
    """

    def __init__(self, arg):
        self.arg = arg
        self.load_model()
        self.data_loader = dict()
        self.Feeder = import_class(self.arg.feeder)
        self.load_data()
        self.num_class = 0
        self.f1_updated = False

        self.initialize_segs()
        self.initialize_f1_score()
        self.initialize_accuracies()

    def initialize_accuracies(self):
        # Bimacs
        self.a_tested = False
        self.accuracies = []

    def initialize_f1_score(self):
        # Bimacs
        self.f1_10 = []
        self.f1_25 = []
        self.f1_50 = []
        self.f1_macro = []

    def initialize_segs(self):
        # dataset A
        self.gt_segs = []
        self.pr_segs = []
        self.labels = []
        self.predictions = []

    def load_data(self):
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=self.Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
            worker_init_fn=init_seed)

    def load_model(self):
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.output_device = output_device
        Model = import_class(self.arg.model)
        self.model = Model(**self.arg.model_args).cuda(output_device)
        # summary(self.model, (3, 1000, self.arg.model_args['num_point']))
        print("Number of parameters:", sum([param.nelement() for param in self.model.parameters()]))
        print("Trainable number of parameters:", sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        self.loss_motion = SegmentationLoss().cuda(output_device)
        self.loss_smooth = SmoothLoss().cuda(output_device)

        if self.arg.weights:
            weights = torch.load(self.arg.weights)
            weights = OrderedDict([[k.split('module.')[-1], v.cuda(output_device)] for k, v in weights.items()])
            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state, strict=False)

        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.arg.device,
                    output_device=output_device)

    def update_f1_results(self):
        self.f1_10.append(f1_overlap(0.1, self.num_class, self.gt_segs, self.pr_segs))
        self.f1_25.append(f1_overlap(0.25, self.num_class, self.gt_segs, self.pr_segs))
        self.f1_50.append(f1_overlap(0.5, self.num_class, self.gt_segs, self.pr_segs))
        self.f1_macro.append(f1_macro_weighted(self.labels, self.predictions))

        if not self.f1_updated:
            self.f1_updated = True
        self.initialize_segs()

    def get_metrics_results(self):
        # First dataset
        if len(self.accuracies) == 0:
            mean_accuracy = self.accuracies[0]
            std_accuracy = 0
        else:
            accuracy = np.array(self.accuracies)
            mean_accuracy = np.mean(accuracy)
            std_accuracy = np.std(accuracy) / math.sqrt(accuracy.shape[0])

        # F1 score
        std_f1_10 = 0
        std_f1_25 = 0
        std_f1_50 = 0
        std_f1_macro = 0
        if len(self.f1_10) == 1:
            f1_10_mean = self.f1_10[0]
            f1_25_mean = self.f1_25[0]
            f1_50_mean = self.f1_50[0]
            f1_macro_mean = self.f1_macro[0]
        else:
            f1_10 = np.array(self.f1_10)
            f1_25 = np.array(self.f1_25)
            f1_50 = np.array(self.f1_50)
            f1_macro = np.array(self.f1_macro)
            f1_10_mean = np.mean(f1_10)
            f1_25_mean = np.mean(f1_25)
            f1_50_mean = np.mean(f1_50)
            f1_macro_mean = np.mean(f1_macro)

            std_f1_10 = np.std(f1_10) / math.sqrt(f1_10.shape[0])
            std_f1_25 = np.std(f1_25) / math.sqrt(f1_25.shape[0])
            std_f1_50 = np.std(f1_50) / math.sqrt(f1_50.shape[0])
            std_f1_macro = np.std(f1_macro) / math.sqrt(f1_macro.shape[0])

        res_dict = {}
        res_dict["mean"] = {}
        res_dict["mean"]["acc, f1_macro, f1_10_25_50"] = [
            mean_accuracy,
            f1_macro_mean,
            f1_10_mean,
            f1_25_mean,
            f1_50_mean
        ]

        res_dict["std"] = {}
        res_dict["std"]["acc, fa_macro, f1_10_25_50"] = [
            std_accuracy,
            std_f1_macro,
            std_f1_10,
            std_f1_25,
            std_f1_50
        ]

        res_dict["info"] = str(self.arg.config)
        results_dir = Path(self.arg.results_dir)
        if not results_dir.exists():
            results_dir.mkdir()
        result_save_name = results_dir/ Path(str(self.arg.result_save_name)) / Path("result_test_acc.json")
        with open(result_save_name, "w") as f:
            json.dump(res_dict, f)
        print(res_dict)

    def eval(self):
        res = {}
        self.num_class = self.arg.model_args['num_class']
        self.model.eval()
        loss_smooth_value = []
        loss_motion_value = []
        score_frag = []
        process = tqdm(self.data_loader['test'])
        duration = 0.0
        total_frames = 0.0
        for batch_idx, data in enumerate(process):
            with torch.no_grad():
                data["motion"] = Variable(data["motion"].float().cuda(self.output_device), requires_grad=False)
                if "image" in data:
                    data["image"] = Variable(data["image"].float().cuda(self.output_device),
                                             requires_grad=False) / 255.0
                label = data["label"]
                index = data["index"]
                name = data["name"]
                if isinstance(label, list):
                    label = torch.stack(label).cuda(device=self.output_device)
                    label = label.transpose(1, 0)
                elif isinstance(label, torch.Tensor):
                    label = label.cuda(device=self.output_device)
                start_time = time.time()
                output = self.model(data)
                torch.cuda.synchronize()
                duration += (time.time() - start_time)
                total_frames += data["motion"].shape[2]
                if isinstance(output, tuple):
                    output, l1 = output
                    l1 = l1.mean()
                else:
                    l1 = 0
                    # label = label.squeeze()
                loss_motion = self.loss_motion(output["pred_motion"], label.to(torch.long)) + l1
                loss_smooth = self.loss_smooth(output["pred_motion"])
                value, prediction = torch.max(output["pred_motion"].data, 1)
                label = label.squeeze().cpu().to(torch.long)
                prediction = prediction.squeeze().cpu()
                accuracy = accuracy_score(label, prediction)
                gt_seg = framewiselabel2seglabel(label)
                pr_seg = framewiselabel2seglabel(prediction)

                score_frag.append(prediction)
                loss_motion_value.append(loss_motion.data.item())
                loss_smooth_value.append(loss_smooth.data.item())

                # self.accuracies.append(accuracy)
                self.gt_segs.append(gt_seg)
                self.pr_segs.append(pr_seg)
                self.predictions.append(prediction)
                self.labels.append(label)
                ares = {}
                ares['predict'] = prediction.tolist()
                ares['ground_truth'] = label.tolist()
                acc = self.data_loader["test"].dataset.acc_seg(prediction.numpy(), 1, label.numpy())
                ares['accuracy'] = acc
                ares['loss'] = loss_motion.item()
            res[f"{name}"] = ares
        loss_motion = np.mean(loss_motion_value)
        accuracy = self.data_loader["test"].dataset.acc_seg(self.predictions, 1, label=self.labels)
        self.accuracies.append(accuracy)
        fps = total_frames / duration
        print('\nAccuracy: ', accuracy)
        print(f'\nFPS: {fps:.1f}')
        # print('\nMean Accuracy: ', np.mean(np.array(self.accuracies)))
        save_dir = os.path.join("./result", arg.result_save_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        resfile = os.path.join(save_dir, "results_test_segments.json")
        print("save name: ", resfile)
        with open(resfile, 'wt') as f:
            json.dump(res, f)
        print('Done.')

    def start(self):
        if self.arg.phase == 'test':
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.eval()


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])  # import return model
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


if __name__ == '__main__':
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.safe_load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                exit(-1)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    init_seed(0)

    res = {}
    num_class = arg.model_args['num_class']

    print("model:", arg.model.split(".")[1])
    print("results name: ", arg.result_save_name)
    print("weight dir: ", Path(arg.weights).parent)

    # initialize processor and update gaussian model with train data
    processor = Processor(arg)
    processor.start()

    processor.update_f1_results()
    processor.get_metrics_results()

    print('Done.')
