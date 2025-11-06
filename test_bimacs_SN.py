#!/usr/bin/env python
from __future__ import print_function

import argparse
import inspect
import itertools
import os
import pickle
import random
import shutil
import math
import time
import json
from collections import OrderedDict
# for plotting
import matplotlib.pyplot as plt
import seaborn as sb
from matplotlib import rc
from pathlib import Path

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
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor

from utils.framewise_loss import SegmentationLoss
from metrics.classification_metrics import (
    test_classification_net,
    test_classification_net_softmax,
    test_classification_net_logits,
    test_classification_net_ensemble
)
from utils.temperature_scaling import ModelWithTemperature
from utils.gauss_utils import get_embeddings, gmm_evaluate, gmm_fit, get_gmm
from utils.ensemble_utils import load_ensemble, ensemble_forward_pass
from metrics.ood_metrics import get_roc_auc, get_roc_auc_logits, get_roc_auc_ensemble
from metrics.calibration_metrics import expected_calibration_error
from metrics.uncertainty_confidence import entropy, logsumexp
from metrics.calibr_metrics import TACELoss, ACELoss, SCELoss, ECELoss


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
        default='/media/hao/data_base/Human_Object_Interaction_Segmentation/segmentation/gauss_model/',
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
        '--gaussian_process',
        type=bool,
        default=False,
        help='use the GP?')
    parser.add_argument(
        '--test_a',
        type=bool,
        default=False,
        help='test noised data?')
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
    gt_list = torch.cat(gt_segs, dim=1).squeeze().to('cpu').tolist()
    pr_list = list(itertools.chain.from_iterable(pr_segs))
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
        # self.load_model()
        self.data_loader = dict()
        self.Feeder = import_class(self.arg.feeder)
        self.load_train_data()
        self.num_class = 0
        self.f1_updated = False

        self.initialize_segs()
        self.initialize_features()
        self.initialize_f1_score()
        self.initialize_calibration_errors()
        self.initialize_auroc()
        self.initialize_accuracies()
        self.initialized_logits()

    def initialized_logits(self):
        # Bimacs
        self.logits = []
        self.logitsSoft = []

        # Noised Bimacs
        self.logitsNoised = []
        self.logitsSoftNoised = []

        # IKEA
        self.logitsB = []
        self.logitsSoftB = []

        # Empty Bimacs
        self.logitsEmpty = []
        self.logitsSoftEmpty = []

    def initialize_accuracies(self):
        # Bimacs
        self.a_tested = False
        self.accuracies = []
        self.accuraciesGP = []

        # Noised Bimacs
        self.add_noise = False
        self.noise_intensity = None
        self.noise_type = None
        self.noised = False
        self.accuraciesNoised = []
        self.accuraciesNoisedGP = []

        # IKEA
        self.test_data_b = False
        self.b_tested = False
        self.accuraciesB = []
        self.accuraciesBGP = []

        # empty dataset
        self.emptied = False
        self.accuraciesEmpty = []
        self.accuraciesEmptyGP = []

    def initialize_features(self):
        # Logits for bimacs, ikea, noised bimacs, empty bimacs.

        # feature space
        self.features = []
        self.featuresB = []
        self.featuresNoised = []
        self.featuresEmpty = []

        # feature

        # # model output
        # self.logitsSoft = []
        # self.logitsSoftB = []
        # self.logitsSoftNoised = []
        # self.logitsSoftEmpty = []

    def initialize_f1_score(self):
        # Bimacs
        self.f1_10 = []
        self.f1_25 = []
        self.f1_50 = []
        self.f1_10GP = []
        self.f1_25GP = []
        self.f1_50GP = []
        self.f1_macro = []
        self.f1_macroGP = []
        # IKEA
        self.f1_10B = []
        self.f1_25B = []
        self.f1_50B = []
        self.f1_10BGP = []
        self.f1_25BGP = []
        self.f1_50BGP = []
        self.f1_macroB = []
        self.f1_macroBGP = []
        # Noised Bimacs
        self.f1_10Noised = []
        self.f1_25Noised = []
        self.f1_50Noised = []
        self.f1_10NoisedGP = []
        self.f1_25NoisedGP = []
        self.f1_50NoisedGP = []
        self.f1_macroNoised = []
        self.f1_macroNoisedGP = []
        # Empty bimacs
        self.f1_10Empty = []
        self.f1_25Empty = []
        self.f1_50Empty = []
        self.f1_10EmptyGP = []
        self.f1_25EmptyGP = []
        self.f1_50EmptyGP = []
        self.f1_macroEmpty = []
        self.f1_macroEmptyGP = []

    def initialize_calibration_errors(self):
        self.eceLoss = ECELoss()
        self.aceLoss = ACELoss()
        self.taceLoss = TACELoss()
        self.sceLoss = SCELoss()
        # Bimacs
        self.eces = []
        self.ecesGP = []
        self.aces = []
        self.acesGP = []
        self.taces = []
        self.tacesGP = []
        self.sces = []
        self.scesGP = []
        self.t_eces = []
        # Noised Bimacs
        self.ecesNoised = []
        self.ecesNoisedGP = []
        self.acesNoised = []
        self.acesNoisedGP = []
        self.tacesNoised = []
        self.tacesNoisedGP = []
        self.scesNoised = []
        self.scesNoisedGP = []
        self.t_ecesNoised = []
        # Empty Bimacs
        self.ecesEmpty = []
        self.ecesEmptyGP = []
        self.acesEmpty = []
        self.acesEmptyGP = []
        self.tacesEmpty = []
        self.tacesEmptyGP = []
        self.scesEmpty = []
        self.scesEmptyGP = []
        # IKEA
        self.ecesB = []
        self.ecesBGP = []
        self.acesB = []
        self.acesBGP = []
        self.tacesB = []
        self.tacesBGP = []
        self.scesB = []
        self.scesBGP = []
        self.t_ecesB = []

    def initialize_auroc(self):
        # IKEA dataset as OoD
        ## evaluation metric softmax
        self.m1_aurocs = []
        self.m1_auprcs = []
        self.m2_aurocs = []
        self.m2_auprcs = []
        ## evaluation metric GP
        self.m1_aurocsGP = []
        self.m1_auprcsGP = []
        self.m2_aurocsGP = []
        self.m2_auprcsGP = []

        # Noised Bimacs as OoD
        ## evaluation metric softmax
        self.m1_aurocsNoised = []
        self.m1_auprcsNoised = []
        self.m2_aurocsNoised = []
        self.m2_auprcsNoised = []
        ## evaluation metric GP
        self.m1_aurocsNoisedGP = []
        self.m1_auprcsNoisedGP = []
        self.m2_aurocsNoisedGP = []
        self.m2_auprcsNoisedGP = []

        # Empty Bimacs as OoD
        ## evaluation metric softmax
        self.m1_aurocsEmpty = []
        self.m1_auprcsEmpty = []
        self.m2_aurocsEmpty = []
        self.m2_auprcsEmpty = []
        ## evaluation metric GP
        self.m1_aurocsEmptyGP = []
        self.m1_auprcsEmptyGP = []
        self.m2_aurocsEmptyGP = []
        self.m2_auprcsEmptyGP = []

        # Post temperature scaling
        self.t_m1_aurocs = []
        self.t_m1_auprcs = []
        self.t_m2_aurocs = []
        self.t_m2_auprcs = []

    def initialize_segs(self):
        # dataset A
        self.gt_segs = []
        self.pr_segs = []
        self.pr_segsGP = []
        self.labels = []
        self.predictions = []
        self.predictionsGP = []

        # IKEA dataset
        self.gt_segsB = []
        self.pr_segsB = []
        self.pr_segsBGP = []
        self.labelsB = []
        self.predictionsB = []
        self.predictionsBGP = []
        # Noised Bimacs
        self.gt_segsNoised = []
        self.pr_segsNoised = []
        self.pr_segsNoisedGP = []
        self.labelsNoised = []
        self.predictionsNoised = []
        self.predictionsNoisedGP = []

        # Empty Bimacs
        self.gt_segsEmpty = []
        self.pr_segsEmpty = []
        self.pr_segsEmptyGP = []
        self.labelsEmpty = []
        self.predictionsEmpty = []
        self.predictionsEmptyGP = []

    def load_train_data(self):
        self.data_loader['train'] = torch.utils.data.DataLoader(
            dataset=self.Feeder(**self.arg.train_feeder_args),
            batch_size=self.arg.batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
            worker_init_fn=init_seed)

    def load_val_data(self, data_path, label_path, test_a=False,
                      test_b=False, is_empty=False, add_noise=False, noise_intensity=0.1, noise_type='gaussian'):
        self.test_a = test_a
        self.add_noise = add_noise
        self.test_data_b = test_b
        self.is_empty = is_empty

        if test_a and not self.a_tested:
            self.a_tested = True
        elif add_noise and not self.noised:
            self.noised = True
        elif test_b and not self.b_tested:
            self.b_tested = True
        elif is_empty and not self.emptied:
            self.emptied = True

        if add_noise:
            self.noise_intensity = noise_intensity
            self.noise_type = noise_type

        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=self.Feeder(data_path=data_path, label_path=label_path, test_b=test_b, is_empty=is_empty,
                                add_noise=add_noise, noise_intensity=noise_intensity, noise_type=noise_type),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
            worker_init_fn=init_seed)

    def load_model(self, weight_i):
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.output_device = output_device
        Model = import_class(self.arg.model)
        work_dir = os.path.join(self.arg.work_dir, self.arg.model.split(".")[-2])
        shutil.copy2(inspect.getfile(Model), work_dir)
        self.model = Model(**self.arg.model_args).cuda(output_device)
        # summary(self.model, (3, 1000, self.arg.model_args['num_point']))
        print("Number of parameters:", sum([param.nelement() for param in self.model.parameters()]))
        print("Trainable number of parameters:", sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        self.loss = SegmentationLoss().cuda(output_device)

        if self.arg.weights:
            # if len(self.arg.weights_all) > weight_i:
            weights = torch.load(os.path.join(self.arg.weight_dir, self.arg.weights))
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

    def plotting_results(self, separate_ID=False):
        if self.a_tested:
            logitsSoft = torch.cat(self.logitsSoft, dim=0).to('cpu')
            logits = torch.cat(self.logits, dim=0).to('cpu')
            # logits = torch.cat(self.features, dim=0).to('cpu')
            entropy_bimacs = entropy(logitsSoft)
            feature_density_bimacs = logsumexp(logits)
            # feature_density_bimacs = torch.max(logits, dim=1)[0]

        if self.emptied:
            logitsSoftEmpty = torch.cat(self.logitsSoftEmpty, dim=0).to('cpu')
            logitsEmpty = torch.cat(self.logitsEmpty, dim=0).to('cpu')
            # logitsEmpty = torch.cat(self.featuresEmpty, dim=0).to('cpu')
            entropy_bimacsEmpty = entropy(logitsSoftEmpty)
            feature_density_bimacsEmpty = logsumexp(logitsEmpty)
            # feature_density_bimacsEmpty = torch.max(logitsEmpty, dim=1)[0]

        if self.noised:
            logitsSoftNoised = torch.cat(self.logitsSoftNoised, dim=0).to('cpu')
            logitsNoised = torch.cat(self.logitsNoised, dim=0).to('cpu')
            # logitsNoised = torch.cat(self.featuresNoised, dim=0).to('cpu')
            entropy_bimacs_noised = entropy(logitsSoftNoised)
            feature_density_bimacs_noised = logsumexp(logitsNoised)
            # feature_density_bimacs_noised = torch.max(logitsNoised, dim=1)[0]

        if self.b_tested:
            logitsSoftB = torch.cat(self.logitsSoftB, dim=0).to('cpu')
            logitsB = torch.cat(self.logitsB, dim=0).to('cpu')
            # logitsB = torch.cat(self.featuresB, dim=0).to('cpu')
            entropy_ikea = entropy(logitsSoftB)
            feature_density_ikea = logsumexp(logitsB)
            # feature_density_ikea = torch.max(logitsB, dim=1)[0]

        sb.set_style('whitegrid')
        sb.set_context("paper", font_scale=1, rc={"lines.linewidth": 2.5})
        rc('text', usetex=False)

        common_kwargs = dict(stat='probability', kde=False, bins=50, binrange=[0, 5], label="dummy", legend=False,
                             element="step", alpha=0.7)

        # plotting entropy
        plt.figure(figsize=(2.5, 2.5 / 1.6))
        plt.tight_layout()

        if self.a_tested:
            sb.histplot(data=entropy_bimacs, color=sb.color_palette("tab10")[0], **common_kwargs)
        if self.b_tested:
            sb.histplot(entropy_ikea, color=sb.color_palette("tab10")[3], **common_kwargs)
        if self.noised:
            sb.histplot(entropy_bimacs_noised, color=sb.color_palette("tab10")[4], **common_kwargs)
        if self.emptied:
            sb.histplot(entropy_bimacsEmpty, color=sb.color_palette("tab10")[5], **common_kwargs)

        plt.xlabel('Entropy', fontsize=12)
        plt.ylabel('Fraction', fontsize=12)
        figure_save_name = self.arg.result_save_name + ".png"
        entropy_save_name = os.path.join(self.arg.results_dir, "entropy_" + figure_save_name)
        plt.savefig(entropy_save_name, bbox_inches='tight', dpi=300)

        # plotting feature density
        plt.figure(figsize=(2.5, 2.5 / 1.6))
        plt.tight_layout()

        # range_config = dict(bins=50, element="step", binrange=[-50, 100], fill=True, alpha=0.7)
        range_config = dict(bins=50, element="step", fill=True, alpha=0.7)
        if self.a_tested:
            sb.histplot(data=feature_density_bimacs, color=sb.color_palette("tab10")[0],
                        stat='probability', kde=False, **range_config, label="dummy",
                        legend=False)
        if self.b_tested:
            sb.histplot(feature_density_ikea, color=sb.color_palette("tab10")[3],
                        stat='probability', kde=False, **range_config, label="dummy",
                        legend=False)
        if self.noised:
            sb.histplot(feature_density_bimacs_noised, color=sb.color_palette("tab10")[4],
                        stat='probability', kde=False, **range_config, label="dummy",
                        legend=False)
        if self.emptied:
            sb.histplot(data=feature_density_bimacsEmpty, color=sb.color_palette("tab10")[5],
                        stat='probability', kde=False, **range_config, label="dummy",
                        legend=False)

        plt.xlabel('Log Density', fontsize=12)
        plt.ylabel('Fraction', fontsize=12)
        density_save_name = os.path.join(self.arg.results_dir, "feature_" + figure_save_name)
        plt.savefig(density_save_name, bbox_inches='tight', dpi=300)
        plt.close('all')

    def update_f1_results(self):
        if self.test_a and not self.add_noise:
            self.f1_10.append(f1_overlap(0.1, self.num_class, self.gt_segs, self.pr_segs))
            self.f1_25.append(f1_overlap(0.25, self.num_class, self.gt_segs, self.pr_segs))
            self.f1_50.append(f1_overlap(0.5, self.num_class, self.gt_segs, self.pr_segs))
            self.f1_macro.append(f1_macro_weighted(self.labels, self.predictions))
            if self.arg.gaussian_process:
            	self.f1_10GP.append(f1_overlap(0.1, self.num_class, self.gt_segs, self.pr_segsGP))
            	self.f1_25GP.append(f1_overlap(0.25, self.num_class, self.gt_segs, self.pr_segsGP))
            	self.f1_50GP.append(f1_overlap(0.5, self.num_class, self.gt_segs, self.pr_segsGP))
            	self.f1_macroGP.append(f1_macro_weighted(self.labels, self.predictionsGP))

        if self.add_noise:
            self.f1_10Noised.append(f1_overlap(0.1, self.num_class, self.gt_segsNoised, self.pr_segsNoised))
            self.f1_25Noised.append(f1_overlap(0.25, self.num_class, self.gt_segsNoised, self.pr_segsNoised))
            self.f1_50Noised.append(f1_overlap(0.5, self.num_class, self.gt_segsNoised, self.pr_segsNoised))
            # print(self.labelsNoised)
            self.f1_macroNoised.append(f1_macro_weighted(self.labelsNoised, self.predictionsNoised))
            if self.arg.gaussian_process:
            	self.f1_10NoisedGP.append(f1_overlap(0.1, self.num_class, self.gt_segsNoised, self.pr_segsNoisedGP))
            	self.f1_25NoisedGP.append(f1_overlap(0.25, self.num_class, self.gt_segsNoised, self.pr_segsNoisedGP))
            	self.f1_50NoisedGP.append(f1_overlap(0.5, self.num_class, self.gt_segsNoised, self.pr_segsNoisedGP))
            	self.f1_macroNoisedGP.append(f1_macro_weighted(self.labelsNoised, self.predictionsNoisedGP))
        if self.is_empty:
            self.f1_10Empty.append(f1_overlap(0.1, self.num_class, self.gt_segsEmpty, self.pr_segsEmpty))
            self.f1_25Empty.append(f1_overlap(0.25, self.num_class, self.gt_segsEmpty, self.pr_segsEmpty))
            self.f1_50Empty.append(f1_overlap(0.5, self.num_class, self.gt_segsEmpty, self.pr_segsEmpty))
            self.f1_macroEmpty.append(f1_macro_weighted(self.labelsEmpty, self.predictionsEmpty))
            if self.arg.gaussian_process:
            	self.f1_10EmptyGP.append(f1_overlap(0.1, self.num_class, self.gt_segsEmpty, self.pr_segsEmptyGP))
            	self.f1_25EmptyGP.append(f1_overlap(0.25, self.num_class, self.gt_segsEmpty, self.pr_segsEmptyGP))
            	self.f1_50EmptyGP.append(f1_overlap(0.5, self.num_class, self.gt_segsEmpty, self.pr_segsEmptyGP))
            	self.f1_macroEmptyGP.append(f1_macro_weighted(self.labelsEmpty, self.predictionsEmptyGP))
        if not self.f1_updated:
            self.f1_updated = True
        self.initialize_segs()

    def get_metrics_results(self):
        # First dataset
        accuracy = np.array(self.accuracies)
        accuracyGP = np.array(self.accuraciesGP)
        ece = np.array(self.eces)
        eceGP = np.array(self.ecesGP)
        ace = np.array(self.aces)
        aceGP = np.array(self.acesGP)
        tace = np.array(self.taces)
        taceGP = np.array(self.tacesGP)
        sce = np.array(self.sces)
        sceGP = np.array(self.scesGP)
        # t_ece = np.array(self.t_eces)
        mean_accuracy = np.mean(accuracy)
        mean_accuracyGP = np.mean(accuracyGP)
        mean_ece = np.mean(ece)
        mean_eceGP = np.mean(eceGP)
        mean_ace = np.mean(ace)
        mean_aceGP = np.mean(aceGP)
        mean_tace = np.mean(tace)
        mean_taceGP = np.mean(taceGP)
        mean_sce = np.mean(sce)
        mean_sceGP = np.mean(sceGP)
        # mean_t_ece = np.mean(t_ece)
        std_accuracy = np.std(accuracy) / math.sqrt(accuracy.shape[0])
        std_accuracyGP = np.std(accuracyGP) / math.sqrt(accuracyGP.shape[0])
        std_ece = np.std(ece) / math.sqrt(ece.shape[0])
        std_eceGP = np.std(eceGP) / math.sqrt(eceGP.shape[0])
        std_ace = np.std(ace) / math.sqrt(ace.shape[0])
        std_aceGP = np.std(aceGP) / math.sqrt(aceGP.shape[0])
        std_tace = np.std(tace) / math.sqrt(tace.shape[0])
        std_taceGP = np.std(taceGP) / math.sqrt(taceGP.shape[0])
        std_sce = np.std(sce) / math.sqrt(sce.shape[0])
        std_sceGP = np.std(sceGP) / math.sqrt(sceGP.shape[0])

        ## IKEA dataset
        m1_auroc = np.array(self.m1_aurocs)
        m1_auprc = np.array(self.m1_auprcs)
        m2_auroc = np.array(self.m2_aurocs)
        m2_auprc = np.array(self.m2_auprcs)
        m1_aurocGP = np.array(self.m1_aurocsGP)
        m1_auprcGP = np.array(self.m1_auprcsGP)
        m2_aurocGP = np.array(self.m2_aurocsGP)
        m2_auprcGP = np.array(self.m2_auprcsGP)

        # F1 score
        f1_10 = np.array(self.f1_10)
        f1_25 = np.array(self.f1_25)
        f1_50 = np.array(self.f1_50)
        f1_10GP = np.array(self.f1_10GP)
        f1_25GP = np.array(self.f1_25GP)
        f1_50GP = np.array(self.f1_50GP)
        f1_macro = np.array(self.f1_macro)
        f1_macroGP = np.array(self.f1_macroGP)

        f1_10_mean = np.mean(f1_10)
        f1_25_mean = np.mean(f1_25)
        f1_50_mean = np.mean(f1_50)
        f1_10_meanGP = np.mean(f1_10GP)
        f1_25_meanGP = np.mean(f1_25GP)
        f1_50_meanGP = np.mean(f1_50GP)
        f1_macro_mean = np.mean(f1_macro)
        f1_macro_meanGP = np.mean(f1_macroGP)

        std_f1_10 = np.std(f1_10) / math.sqrt(f1_10.shape[0])
        std_f1_25 = np.std(f1_25) / math.sqrt(f1_25.shape[0])
        std_f1_50 = np.std(f1_50) / math.sqrt(f1_50.shape[0])
        std_f1_10GP = np.std(f1_10GP) / math.sqrt(f1_10GP.shape[0])
        std_f1_25GP = np.std(f1_25GP) / math.sqrt(f1_25GP.shape[0])
        std_f1_50GP = np.std(f1_50GP) / math.sqrt(f1_50GP.shape[0])
        std_f1_macro = np.std(f1_macro) / math.sqrt(f1_macro.shape[0])
        std_f1_macroGP = np.std(f1_macroGP) / math.sqrt(f1_macroGP.shape[0])

        res_dict = {}
        res_dict["mean"] = {}
        res_dict["mean"]["acc, f1_macro, f1_10_25_50"] = [
            mean_accuracy.item(),
            f1_macro_mean.item(),
            f1_10_mean.item(),
            f1_25_mean.item(),
            f1_50_mean.item()
        ]
        res_dict["mean"]["accGP, f1_macroGP, f1_10_25_50GP"] = [
            mean_accuracyGP.item(),
            f1_macro_meanGP.item(),
            f1_10_meanGP.item(),
            f1_25_meanGP.item(),
            f1_50_meanGP.item()
        ]
        res_dict["mean"]["ece,tace,ace,sce"] = [mean_ece.item(), mean_tace.item(), mean_ace.item(), mean_sce.item()]
        res_dict["mean"]["eceGP,taceGP,aceGP,sceGP"] = [mean_eceGP.item(), mean_taceGP.item(), mean_aceGP.item(),
                                                        mean_sceGP.item()]
        # res_dict["mean"]["t_ece,t_m1_auroc,t_m1_auprc,t_m2_auroc,t_m2_auprc"] = \
        #     [mean_t_ece.item(), mean_t_m1_auroc.item(), mean_t_m1_auprc.item(), mean_t_m2_auroc.item(),
        #      mean_t_m2_auprc.item()]

        res_dict["std"] = {}
        res_dict["std"]["acc, f1_macro, f1_10_25_50"] = [
            std_accuracy.item(),
            std_f1_macro.item(),
            std_f1_10.item(),
            std_f1_25.item(),
            std_f1_50.item()]
        res_dict["std"]["accGP, f1_10_25_50GP"] = [
            std_accuracyGP.item(),
            std_f1_macroGP.item(),
            std_f1_10GP.item(),
            std_f1_25GP.item(),
            std_f1_50GP.item()]

        res_dict["std"]["ece,tace,ace,sce"] = [std_ece.item(), std_tace.item(), std_ace.item(), std_sce.item()]
        res_dict["std"]["eceGP,taceGP,aceGP,sceGP"] = [std_eceGP.item(), std_taceGP.item(), std_aceGP.item(),
                                                       std_sceGP.item()]
        if self.test_data_b:
            mean_m1_auroc = np.mean(m1_auroc)
            mean_m1_auprc = np.mean(m1_auprc)
            mean_m2_auroc = np.mean(m2_auroc)
            mean_m2_auprc = np.mean(m2_auprc)
            mean_m1_aurocGP = np.mean(m1_aurocGP)
            mean_m1_auprcGP = np.mean(m1_auprcGP)
            mean_m2_aurocGP = np.mean(m2_aurocGP)
            mean_m2_auprcGP = np.mean(m2_auprcGP)
            if len(self.m1_aurocs) > 1:
                std_m1_auroc = np.std(m1_auroc) / math.sqrt(m1_auroc.shape[0])
                std_m1_auprc = np.std(m1_auprc) / math.sqrt(m1_auprc.shape[0])
                std_m2_auroc = np.std(m2_auroc) / math.sqrt(m2_auroc.shape[0])
                std_m2_auprc = np.std(m2_auprc) / math.sqrt(m2_auprc.shape[0])
                std_m1_aurocGP = np.std(m1_aurocGP) / math.sqrt(m1_aurocGP.shape[0])
                std_m1_auprcGP = np.std(m1_auprcGP) / math.sqrt(m1_auprcGP.shape[0])
                std_m2_aurocGP = np.std(m2_aurocGP) / math.sqrt(m2_aurocGP.shape[0])
                std_m2_auprcGP = np.std(m2_auprcGP) / math.sqrt(m2_auprcGP.shape[0])
            else:
                mean_m1_auroc = m1_auroc
                mean_m1_auprc = m1_auprc
                mean_m2_auroc = m2_auroc
                mean_m2_auprc = m2_auprc
                mean_m1_aurocGP = m1_aurocGP
                mean_m1_auprcGP = m1_auprcGP
                mean_m2_aurocGP = m2_aurocGP
                mean_m2_auprcGP = m2_auprcGP
                std_m1_auroc = np.array(0.0)
                std_m1_auprc = np.array(0.0)
                std_m2_auroc = np.array(0.0)
                std_m2_auprc = np.array(0.0)
                std_m1_aurocGP = np.array(0.0)
                std_m1_auprcGP = np.array(0.0)
                std_m2_aurocGP = np.array(0.0)
                std_m2_auprcGP = np.array(0.0)
            res_dict["mean"]["m1_auroc,m1_auprc,m2_auroc,m2_auprc"] = \
                [mean_m1_auroc.item(), mean_m1_auprc.item(), mean_m2_auroc.item(),
                 mean_m2_auprc.item()]
            res_dict["mean"]["m1_aurocGP,m1_auprcGP,m2_aurocGP,m2_auprcGP"] = \
                [mean_m1_aurocGP.item(), mean_m1_auprcGP.item(), mean_m2_aurocGP.item(),
                 mean_m2_auprcGP.item()]
            res_dict["std"]["m1_auroc, m1_auprc, m2_auroc, m2_auprc"] = \
                [std_m1_auroc.item(), std_m1_auprc.item(), std_m2_auroc.item(), std_m2_auprc.item()]
            res_dict["std"]["m1_aurocGP, m1_auprcGP, m2_aurocGP, m2_auprcGP"] = \
                [std_m1_aurocGP.item(), std_m1_auprcGP.item(), std_m2_aurocGP.item(), std_m2_auprcGP.item()]

        ## empty bimacs
        if self.is_empty:
            m1_aurocEmpty = np.array(self.m1_aurocsEmpty)
            m1_auprcEmpty = np.array(self.m1_auprcsEmpty)
            m2_aurocEmpty = np.array(self.m2_aurocsEmpty)
            m2_auprcEmpty = np.array(self.m2_auprcsEmpty)
            m1_aurocEmptyGP = np.array(self.m1_aurocsEmptyGP)
            m1_auprcEmptyGP = np.array(self.m1_auprcsEmptyGP)
            m2_aurocEmptyGP = np.array(self.m2_aurocsEmptyGP)
            m2_auprcEmptyGP = np.array(self.m2_auprcsEmptyGP)

            if len(self.m1_aurocsEmpty) > 1:
                mean_m1_aurocEmpty = np.mean(m1_aurocEmpty)
                mean_m1_auprcEmpty = np.mean(m1_auprcEmpty)
                mean_m2_aurocEmpty = np.mean(m2_aurocEmpty)
                mean_m2_auprcEmpty = np.mean(m2_auprcEmpty)
                mean_m1_aurocEmptyGP = np.mean(m1_aurocEmptyGP)
                mean_m1_auprcEmptyGP = np.mean(m1_auprcEmptyGP)
                mean_m2_aurocEmptyGP = np.mean(m2_aurocEmptyGP)
                mean_m2_auprcEmptyGP = np.mean(m2_auprcEmptyGP)

                std_m1_aurocEmpty = np.std(m1_aurocEmpty) / math.sqrt(m1_aurocEmpty.shape[0])
                std_m1_auprcEmpty = np.std(m1_auprcEmpty) / math.sqrt(m1_auprcEmpty.shape[0])
                std_m2_aurocEmpty = np.std(m2_aurocEmpty) / math.sqrt(m2_aurocEmpty.shape[0])
                std_m2_auprcEmpty = np.std(m2_auprcEmpty) / math.sqrt(m2_auprcEmpty.shape[0])
                std_m1_aurocEmptyGP = np.std(m1_aurocEmptyGP) / math.sqrt(m1_aurocEmptyGP.shape[0])
                std_m1_auprcEmptyGP = np.std(m1_auprcEmptyGP) / math.sqrt(m1_auprcEmptyGP.shape[0])
                std_m2_aurocEmptyGP = np.std(m2_aurocEmptyGP) / math.sqrt(m2_aurocEmptyGP.shape[0])
                std_m2_auprcEmptyGP = np.std(m2_auprcEmptyGP) / math.sqrt(m2_auprcEmptyGP.shape[0])
            else:
                mean_m1_aurocEmpty = m1_aurocEmpty
                mean_m1_auprcEmpty = m1_auprcEmpty
                mean_m2_aurocEmpty = m2_aurocEmpty
                mean_m2_auprcEmpty = m2_auprcEmpty
                mean_m1_aurocEmptyGP = m1_aurocEmptyGP
                mean_m1_auprcEmptyGP = m1_auprcEmptyGP
                mean_m2_aurocEmptyGP = m2_aurocEmptyGP
                mean_m2_auprcEmptyGP = m2_auprcEmptyGP

                std_m1_aurocEmpty = np.array(0.0)
                std_m1_auprcEmpty = np.array(0.0)
                std_m2_aurocEmpty = np.array(0.0)
                std_m2_auprcEmpty = np.array(0.0)
                std_m1_aurocEmptyGP = np.array(0.0)
                std_m1_auprcEmptyGP = np.array(0.0)
                std_m2_aurocEmptyGP = np.array(0.0)
                std_m2_auprcEmptyGP = np.array(0.0)
            res_dict["mean"]["m1_aurocEmpty,m1_auprcEmpty,m2_aurocEmpty,m2_auprcEmpty"] = \
                [mean_m1_aurocEmpty.item(), mean_m1_auprcEmpty.item(), mean_m2_aurocEmpty.item(),
                 mean_m2_auprcEmpty.item()]
            res_dict["mean"]["m1_aurocEmptyGP,m1_auprcEmptyGP,m2_aurocEmptyGP,m2_auprcEmptyGP"] = \
                [mean_m1_aurocEmptyGP.item(), mean_m1_auprcEmptyGP.item(), mean_m2_aurocEmptyGP.item(),
                 mean_m2_auprcEmptyGP.item()]
            res_dict["std"]["m1_aurocEmpty, m1_auprcEmpty, m2_aurocEmpty, m2_auprcEmpty"] = \
                [std_m1_aurocEmpty.item(), std_m1_auprcEmpty.item(),
                 std_m2_aurocEmpty.item(), std_m2_auprcEmpty.item()]
            res_dict["std"]["m1_aurocEmptyGP, m1_auprcEmptyGP, m2_aurocEmptyGP, m2_auprcEmptyGP"] = \
                [std_m1_aurocEmptyGP.item(), std_m1_auprcEmptyGP.item(),
                 std_m2_aurocEmptyGP.item(), std_m2_auprcEmptyGP.item()]

        ## Noised Bimacs dataset
        if self.add_noise:
            # accuracy
            accuracyNoised = np.array(self.accuraciesNoised)
            accuracyNoisedGP = np.array(self.accuraciesNoisedGP)
            mean_accuracyNoised = np.mean(accuracyNoised)
            mean_accuracyNoisedGP = np.mean(accuracyNoisedGP)
            std_accuracyNoised = np.std(accuracyNoised) / math.sqrt(accuracyNoised.shape[0])
            std_accuracyNoisedGP = np.std(accuracyNoisedGP) / math.sqrt(accuracyNoisedGP.shape[0])
            # F1 score
            f1_10Noised = np.array(self.f1_10Noised)
            f1_25Noised = np.array(self.f1_25Noised)
            f1_50Noised = np.array(self.f1_50Noised)
            f1_10NoisedGP = np.array(self.f1_10NoisedGP)
            f1_25NoisedGP = np.array(self.f1_25NoisedGP)
            f1_50NoisedGP = np.array(self.f1_50NoisedGP)
            f1_macroNoised = np.array(self.f1_macroNoised)
            f1_macroNoisedGP = np.array(self.f1_macroNoisedGP)

            f1_10Noised_mean = np.mean(f1_10Noised)
            f1_25Noised_mean = np.mean(f1_25Noised)
            f1_50Noised_mean = np.mean(f1_50Noised)
            f1_10Noised_meanGP = np.mean(f1_10NoisedGP)
            f1_25Noised_meanGP = np.mean(f1_25NoisedGP)
            f1_50Noised_meanGP = np.mean(f1_50NoisedGP)
            f1_macroNoised_mean = np.mean(f1_macroNoised)
            f1_macroNoised_meanGP = np.mean(f1_macroNoisedGP)

            std_f1_10Noised = np.std(f1_10Noised) / math.sqrt(f1_10Noised.shape[0])
            std_f1_25Noised = np.std(f1_25Noised) / math.sqrt(f1_25Noised.shape[0])
            std_f1_50Noised = np.std(f1_50Noised) / math.sqrt(f1_50Noised.shape[0])
            std_f1_10NoisedGP = np.std(f1_10NoisedGP) / math.sqrt(f1_10NoisedGP.shape[0])
            std_f1_25NoisedGP = np.std(f1_25NoisedGP) / math.sqrt(f1_25NoisedGP.shape[0])
            std_f1_50NoisedGP = np.std(f1_50NoisedGP) / math.sqrt(f1_50NoisedGP.shape[0])
            std_f1_macroNoised = np.std(f1_macroNoised) / math.sqrt(f1_macroNoised.shape[0])
            std_f1_macroNoisedGP = np.std(f1_macroNoisedGP) / math.sqrt(f1_macroNoisedGP.shape[0])

            m1_aurocNoised = np.array(self.m1_aurocsNoised)
            m1_auprcNoised = np.array(self.m1_auprcsNoised)
            m2_aurocNoised = np.array(self.m2_aurocsNoised)
            m2_auprcNoised = np.array(self.m2_auprcsNoised)
            m1_aurocNoisedGP = np.array(self.m1_aurocsNoisedGP)
            m1_auprcNoisedGP = np.array(self.m1_auprcsNoisedGP)
            m2_aurocNoisedGP = np.array(self.m2_aurocsNoisedGP)
            m2_auprcNoisedGP = np.array(self.m2_auprcsNoisedGP)
            if len(self.m1_aurocsNoised) > 1:
                mean_m1_aurocNoised = np.mean(m1_aurocNoised)
                mean_m1_auprcNoised = np.mean(m1_auprcNoised)
                mean_m2_aurocNoised = np.mean(m2_aurocNoised)
                mean_m2_auprcNoised = np.mean(m2_auprcNoised)
                mean_m1_aurocNoisedGP = np.mean(m1_aurocNoisedGP)
                mean_m1_auprcNoisedGP = np.mean(m1_auprcNoisedGP)
                mean_m2_aurocNoisedGP = np.mean(m2_aurocNoisedGP)
                mean_m2_auprcNoisedGP = np.mean(m2_auprcNoisedGP)

                std_m1_aurocNoised = np.std(m1_aurocNoised) / math.sqrt(m1_aurocNoised.shape[0])
                std_m1_auprcNoised = np.std(m1_auprcNoised) / math.sqrt(m1_auprcNoised.shape[0])
                std_m2_aurocNoised = np.std(m2_aurocNoised) / math.sqrt(m2_aurocNoised.shape[0])
                std_m2_auprcNoised = np.std(m2_auprcNoised) / math.sqrt(m2_auprcNoised.shape[0])
                std_m1_aurocNoisedGP = np.std(m1_aurocNoisedGP) / math.sqrt(m1_aurocNoisedGP.shape[0])
                std_m1_auprcNoisedGP = np.std(m1_auprcNoisedGP) / math.sqrt(m1_auprcNoisedGP.shape[0])
                std_m2_aurocNoisedGP = np.std(m2_aurocNoisedGP) / math.sqrt(m2_aurocNoisedGP.shape[0])
                std_m2_auprcNoisedGP = np.std(m2_auprcNoisedGP) / math.sqrt(m2_auprcNoisedGP.shape[0])
            elif len(self.m1_aurocsNoised) == 1:
                mean_m1_aurocNoised = m1_aurocNoised
                mean_m1_auprcNoised = m1_auprcNoised
                mean_m2_aurocNoised = m2_aurocNoised
                mean_m2_auprcNoised = m2_auprcNoised
                mean_m1_aurocNoisedGP = m1_aurocNoisedGP
                mean_m1_auprcNoisedGP = m1_auprcNoisedGP
                mean_m2_aurocNoisedGP = m2_aurocNoisedGP
                mean_m2_auprcNoisedGP = m2_auprcNoisedGP

                std_m1_aurocNoised = np.array(0.0)
                std_m1_auprcNoised = np.array(0.0)
                std_m2_aurocNoised = np.array(0.0)
                std_m2_auprcNoised = np.array(0.0)
                std_m1_aurocNoisedGP = np.array(0.0)
                std_m1_auprcNoisedGP = np.array(0.0)
                std_m2_aurocNoisedGP = np.array(0.0)
                std_m2_auprcNoisedGP = np.array(0.0)

            res_dict["mean"]["accNoised, f1_macroNoised, f1_10_25_50Noised"] = [
                mean_accuracyNoised.item(),
                f1_macroNoised_mean.item(),
                f1_10Noised_mean.item(),
                f1_25Noised_mean.item(),
                f1_50Noised_mean.item()
            ]
            res_dict["mean"]["accNoisedGP, f1_macroNoisedGP, f1_10_25_50NoisedGP"] = [
                mean_accuracyNoisedGP.item(),
                f1_macroNoised_meanGP.item(),
                f1_10Noised_meanGP.item(),
                f1_25Noised_meanGP.item(),
                f1_50Noised_meanGP.item()
            ]
            res_dict["std"]["acc, f1_macro, f1_10_25_50"] = [
                std_accuracyNoised.item(),
                std_f1_macroNoised.item(),
                std_f1_10Noised.item(),
                std_f1_25Noised.item(),
                std_f1_50Noised.item()
            ]
            res_dict["std"]["accGP, f1_macroGP, f1_10_25_50GP"] = [
                std_accuracyNoisedGP.item(),
                std_f1_macroNoisedGP.item(),
                std_f1_10NoisedGP.item(),
                std_f1_25NoisedGP.item(),
                std_f1_50NoisedGP.item()
            ]
            
            if len(self.m1_aurocsNoised)>=1:
            	res_dict["mean"]["m1_aurocNoised,m1_auprcNoised,m2_aurocNoised,m2_auprcNoised"] = \
                	[mean_m1_aurocNoised, mean_m1_auprcNoised, mean_m2_aurocNoised,
                 	mean_m2_auprcNoised]
            	res_dict["mean"]["m1_aurocNoisedGP,m1_auprcNoisedGP,m2_aurocNoisedGP,m2_auprcNoisedGP"] = \
                	[mean_m1_aurocNoisedGP, mean_m1_auprcNoisedGP, mean_m2_aurocNoisedGP,
                 	mean_m2_auprcNoisedGP]            
            	res_dict["std"]["m1_aurocNoised, m1_auprcNoised, m2_aurocNoised, m2_auprcNoised"] = \
                	[std_m1_aurocNoised.item(), std_m1_auprcNoised.item(), std_m2_aurocNoised.item(),
                 	std_m2_auprcNoised.item()]
            	res_dict["std"]["m1_aurocNoisedGP, m1_auprcNoisedGP, m2_aurocNoisedGP, m2_auprcNoisedGP"] = \
                	[std_m1_aurocNoisedGP.item(), std_m1_auprcNoisedGP.item(),
                 	std_m2_aurocNoisedGP.item(), std_m2_auprcNoisedGP.item()]


        res_dict["info"] = str(self.arg.config)
        result_save_name = os.path.join(self.arg.results_dir, "result_" + str(self.arg.result_save_name) + "_" +
                                        str(self.noise_type) + "_" + str(self.noise_intensity) + "_bimacs_ikea.json")
        with open(result_save_name, "w") as f:
            json.dump(res_dict, f)
        print(res_dict)

    def update_roc_by_features(self):

        # Pre temperature scaling
        # m1 - Uncertainty/Confidence Metric 1
        #      for deterministic model: logsumexp, for ensemble: entropy
        # m2 - Uncertainty/Confidence Metric 2
        #      for deterministic model: entropy, for ensemble: MI
        if self.a_tested:
            if self.b_tested:
                try:
                    (_, _, _), (_, _, _), m1_auroc, m1_auprc = get_roc_auc_logits(self.logits, self.logitsB,
                                                                                  logsumexp,
                                                                                  self.arg.device,
                                                                                  confidence=True)
                    (_, _, _), (_, _, _), m2_auroc, m2_auprc = get_roc_auc_logits(self.logits, self.logitsB,
                                                                                  entropy,
                                                                                  self.arg.device,
                                                                                  confidence=False)
                    # GP results
                    self.m1_aurocsGP.append(m1_auroc)
                    self.m1_auprcsGP.append(m1_auprc)
                    self.m2_aurocsGP.append(m2_auroc)
                    self.m2_auprcsGP.append(m2_auprc)
                except RuntimeError as e:
                    print("Runtime Error caught: " + str(e))
            if self.emptied:
                try:
                    (_, _, _), (_, _, _), m1_auroc, m1_auprc = get_roc_auc_logits(self.logits, self.logitsEmpty,
                                                                                  logsumexp,
                                                                                  self.arg.device,
                                                                                  confidence=True)
                    (_, _, _), (_, _, _), m2_auroc, m2_auprc = get_roc_auc_logits(self.logits, self.logitsEmpty,
                                                                                  entropy,
                                                                                  self.arg.device,
                                                                                  confidence=False)
                    self.m1_aurocsEmptyGP.append(m1_auroc)
                    self.m1_auprcsEmptyGP.append(m1_auprc)
                    self.m2_aurocsEmptyGP.append(m2_auroc)
                    self.m2_auprcsEmptyGP.append(m2_auprc)
                except RuntimeError as e:
                    print("Runtime Error caught: " + str(e))
            if self.noised:
                try:
                    (_, _, _), (_, _, _), m1_auroc, m1_auprc = get_roc_auc_logits(self.logits, self.logitsNoised,
                                                                                  logsumexp,
                                                                                  self.arg.device,
                                                                                  confidence=True)
                    (_, _, _), (_, _, _), m2_auroc, m2_auprc = get_roc_auc_logits(self.logits, self.logitsNoised,
                                                                                  entropy,
                                                                                  self.arg.device,
                                                                                  confidence=False)
                    self.m1_aurocsNoisedGP.append(m1_auroc)
                    self.m1_auprcsNoisedGP.append(m1_auprc)
                    self.m2_aurocsNoisedGP.append(m2_auroc)
                    self.m2_auprcsNoisedGP.append(m2_auprc)
                except RuntimeError as e:
                    print("Runtime Error caught: " + str(e))
        self.initialize_features()

    def update_roc_by_softmax(self):
        if self.a_tested:
            if self.b_tested:
                (_, _, _), (_, _, _), m1_auroc, m1_auprc = get_roc_auc_logits(self.logitsSoft, self.logitsSoftB,
                                                                              logsumexp,
                                                                              self.arg.device,
                                                                              confidence=True)
                (_, _, _), (_, _, _), m2_auroc, m2_auprc = get_roc_auc_logits(self.logitsSoft, self.logitsSoftB,
                                                                              entropy,
                                                                              self.arg.device, confidence=False)

                self.m1_aurocs.append(m1_auroc)
                self.m1_auprcs.append(m1_auprc)
                self.m2_aurocs.append(m2_auroc)
                self.m2_auprcs.append(m2_auprc)

            if self.emptied:
                (_, _, _), (_, _, _), m1_auroc, m1_auprc = get_roc_auc_logits(self.logitsSoft, self.logitsSoftEmpty,
                                                                              logsumexp,
                                                                              self.arg.device,
                                                                              confidence=True)
                (_, _, _), (_, _, _), m2_auroc, m2_auprc = get_roc_auc_logits(self.logitsSoft, self.logitsSoftEmpty,
                                                                              entropy,
                                                                              self.arg.device, confidence=False)

                self.m1_aurocsEmpty.append(m1_auroc)
                self.m1_auprcsEmpty.append(m1_auprc)
                self.m2_aurocsEmpty.append(m2_auroc)
                self.m2_auprcsEmpty.append(m2_auprc)

            if self.noised:
                (_, _, _), (_, _, _), m1_auroc, m1_auprc = get_roc_auc_logits(self.logitsSoft, self.logitsSoftNoised,
                                                                              logsumexp,
                                                                              self.arg.device,
                                                                              confidence=True)
                (_, _, _), (_, _, _), m2_auroc, m2_auprc = get_roc_auc_logits(self.logitsSoft, self.logitsSoftNoised,
                                                                              entropy,
                                                                              self.arg.device, confidence=False)

                self.m1_aurocsNoised.append(m1_auroc)
                self.m1_auprcsNoised.append(m1_auprc)
                self.m2_aurocsNoised.append(m2_auroc)
                self.m2_auprcsNoised.append(m2_auprc)

    def update_gaussian_model(self, weight_i):
        if (arg.model_type == "gauss"):
            # Evaluate a GMM model
            print("Gauss Model")
            if (arg.gauss_mean.endswith('.npy')):
                mean_file = arg.gauss_mean
                cov_file = arg.gauss_cov
            else:
                mean_file = arg.gauss_mean + "_" + str(weight_i) + ".npy"
                cov_file = arg.gauss_cov + "_" + str(weight_i) + ".npy"
            # mean_file = "gauss_mean_SN_res_max1_1.npy"
            # cov_file = "gauss_cov_SN_res_max1_1.npy"
            gauss_mean_file = os.path.join(arg.gauss_model_dir, mean_file)
            gauss_cov_file = os.path.join(arg.gauss_model_dir, cov_file)
            if os.path.exists(gauss_mean_file) and os.path.exists(gauss_cov_file):
                classwise_mean = np.load(gauss_mean_file)
                classwise_cov = np.load(gauss_cov_file)
                self.gaussians_model, self.jitter_eps = get_gmm(classwise_mean, classwise_cov,
                                                                num_classes=self.arg.model_args['num_class'],
                                                                device=self.arg.device)
            else:
                # feature_shape = ()
                # if arg.dataset == 'bimacs':
                #     feature_shape = (512, 120)
                # elif arg.dataset == 'ikea':
                #     feature_shape = (512, 500)
                embeddings, labels = get_embeddings(self.model, self.data_loader['train'],
                                                    dtype=torch.double, device=self.arg.device,
                                                    storage_device=self.arg.device)
                try:
                    classwise_mean, classwise_cov = gmm_fit(embeddings=embeddings, labels=labels,
                                                            num_classes=self.arg.model_args['num_class'],
                                                            device=self.arg.device)
                    np.save(gauss_mean_file, classwise_mean)
                    np.save(gauss_cov_file, classwise_cov)
                    self.gaussians_model, self.jitter_eps = get_gmm(classwise_mean, classwise_cov,
                                                                    num_classes=self.arg.model_args['num_class'],
                                                                    device=self.arg.device)
                except RuntimeError as e:
                    print("Runtime Error caught: " + str(e))

    def eval_model(self):
        self.num_class = self.arg.model_args['num_class']
        if arg.model_type == "ensemble":
            # TODO: ensemble predictions of several models
            return
        # using gaussian model with density
        else:
            # evaluate on the original model
            (conf_matrix, accuracy, labels_list, predictions, confidences) = \
                test_classification_net(self.model, self.data_loader['test'], device=self.arg.device)
            labels = torch.tensor(labels_list).unsqueeze(0)
            # print(labels.shape)

            # evaluate gaussian model
            # logitsGP: log probability from gaussian model: BS x classes
            # logiys_soft: output of network, assume it processed by a softmax layer
            # labels: is label
            if self.arg.gaussian_process:
                logitsGP, labels, logtis_soft, features = gmm_evaluate(self.model, self.gaussians_model,
                                                                       self.data_loader['test'],
                                                                       device=self.arg.device,
                                                                       num_classes=self.arg.model_args['num_class'],
                                                                       storage_device=self.arg.device
                                                                       )
                # print(labels.shape)

                (_, accuracyGP, _, predictionsGP, confidencesGP) = test_classification_net_logits(logitsGP, labels)
                # (_, accuracyGP, _, predictionsGP, confidencesGP) = test_classification_net_softmax(logitsGP, labels)
                # eceLoss: loss(self, output, labels, n_bins=15, logits=True):
                ece = self.eceLoss.loss(logtis_soft.cpu().numpy(), labels.squeeze().cpu().numpy())
                ace = self.aceLoss.loss(logtis_soft.cpu().numpy(), labels.squeeze().cpu().numpy())
                tace = self.taceLoss.loss(logtis_soft.cpu().numpy(), labels.squeeze().cpu().numpy())
                sce = self.sceLoss.loss(logtis_soft.cpu().numpy(), labels.squeeze().cpu().numpy())
                eceGP = self.eceLoss.loss(logitsGP.cpu().numpy(), labels.squeeze().cpu().numpy())
                aceGP = self.aceLoss.loss(logitsGP.cpu().numpy(), labels.squeeze().cpu().numpy())
                taceGP = self.taceLoss.loss(logitsGP.cpu().numpy(), labels.squeeze().cpu().numpy())
                sceGP = self.sceLoss.loss(logitsGP.cpu().numpy(), labels.squeeze().cpu().numpy())
                pr_segGP = framewiselabel2seglabel(predictionsGP)

            gt_seg = framewiselabel2seglabel(labels_list)
            pr_seg = framewiselabel2seglabel(predictions)

            if self.test_data_b:
                self.accuraciesB.append(accuracy)
                self.gt_segsB.append(gt_seg)
                self.pr_segsB.append(pr_seg)
                self.labelsB.append(labels)
                self.predictionsB.append(predictions)

                if self.arg.gaussian_process:
                    self.ecesB.append(ece)
                    self.acesB.append(ace)
                    self.tacesB.append(tace)
                    self.scesB.append(sce)
                    # self.t_ecesB.append(t_ece)
                    self.featuresB.append(features)
                    self.accuraciesBGP.append(accuracyGP)
                    self.pr_segsBGP.append(pr_segGP)
                    self.ecesBGP.append(eceGP)
                    self.acesBGP.append(aceGP)
                    self.tacesBGP.append(taceGP)
                    self.scesBGP.append(sceGP)
                    self.logitsB.append(logitsGP)
                    self.predictionsBGP.append(predictionsGP)
                    self.logitsSoftB.append(logtis_soft)
            elif self.is_empty:
                self.accuraciesEmpty.append(accuracy)
                self.gt_segsEmpty.append(gt_seg)
                self.pr_segsEmpty.append(pr_seg)
                self.labelsEmpty.append(labels)
                self.predictionsEmpty.append(predictions)
                if self.arg.gaussian_process:
                    self.featuresEmpty.append(features)
                    self.accuraciesEmptyGP.append(accuracyGP)
                    self.pr_segsEmptyGP.append(pr_segGP)
                    self.ecesEmpty.append(ece)
                    self.acesEmpty.append(ace)
                    self.tacesEmpty.append(tace)
                    self.scesEmpty.append(sce)
                    # self.t_ecesNoised.append(t_ece)
                    self.ecesNoisedGP.append(eceGP)
                    self.acesNoisedGP.append(aceGP)
                    self.tacesNoisedGP.append(taceGP)
                    self.scesNoisedGP.append(sceGP)
                    self.logitsEmpty.append(logitsGP)
                    self.predictionsEmptyGP.append(predictionsGP)
                    self.logitsSoftEmpty.append(logtis_soft)

            elif self.add_noise:
                self.accuraciesNoised.append(accuracy)
                self.gt_segsNoised.append(gt_seg)
                self.pr_segsNoised.append(pr_seg)
                self.labelsNoised.append(labels)
                self.predictionsNoised.append(predictions)
                if self.arg.gaussian_process:
                    self.featuresNoised.append(features)
                    self.accuraciesNoisedGP.append(accuracyGP)
                    self.pr_segsNoisedGP.append(pr_segGP)
                    self.ecesNoised.append(ece)
                    self.acesNoised.append(ace)
                    self.tacesNoised.append(tace)
                    self.scesNoised.append(sce)
                    # self.t_ecesNoised.append(t_ece)
                    self.ecesNoisedGP.append(eceGP)
                    self.acesNoisedGP.append(aceGP)
                    self.tacesNoisedGP.append(taceGP)
                    self.scesNoisedGP.append(sceGP)
                    self.logitsNoised.append(logitsGP)
                    self.predictionsNoisedGP.append(predictionsGP)
                    self.logitsSoftNoised.append(logtis_soft)
            elif self.test_a and not self.add_noise:
                self.accuracies.append(accuracy)
                self.gt_segs.append(gt_seg)
                self.pr_segs.append(pr_seg)
                self.labels.append(labels)
                self.predictions.append(predictions)
                if self.arg.gaussian_process:
                    self.features.append(features)
                    self.accuraciesGP.append(accuracyGP)
                    self.eces.append(ece)
                    self.aces.append(ace)
                    self.taces.append(tace)
                    self.sces.append(sce)
                    # self.t_eces.append(t_ece)
                    self.ecesGP.append(eceGP)
                    self.acesGP.append(aceGP)
                    self.tacesGP.append(taceGP)
                    self.scesGP.append(sceGP)
                    self.logits.append(logitsGP)
                    self.pr_segsGP.append(pr_segGP)
                    self.predictionsGP.append(predictionsGP)
                    self.logitsSoft.append(logtis_soft)

            else:
                raise Exception("Wrong dataset setup!")

    def start(self):
        if self.arg.phase == 'test':
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.eval_model()
            

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
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    init_seed(0)

    original_path = arg.test_feeder_args['original_path']
    data_path_b = arg.test_feeder_args['data_path_b']
    hands = ['r', 'l']
    res = {}
    num_class = arg.model_args['num_class']

    print("model:", arg.model.split(".")[1])
    print("results name: ", arg.result_save_name)
    print("weight dir: ", arg.weight_dir.split("/")[-2])
    # print("coeff: ", arg.model_args['coeff'])

    # arg.results_dir = os.path.join(str(arg.results_dir), str(arg.model.split(".")[1]))
    arg.results_dir = Path(arg.results_dir) / Path(str(arg.result_save_name))
    if not os.path.exists(arg.results_dir):
        os.mkdir(arg.results_dir)

    # initialize processor and update gaussian model with train data
    processor = Processor(arg)
    # processor.update_gaussian_model(0)

    # evaluate gaussian model on bimacs test dataset

    # for i in range(arg.runs):
    for i in range(1):
        if arg.gaussian_process:
            processor.load_model(i)
            processor.update_gaussian_model(i)
        print(f"Evaluating run: {(i + 1)}")
        if arg.test_a:
            processor.arg.model_args['dataset'] = 'bimacs'
            processor.load_model(i)
            print("Running on Bimacs dataset")
            # Bimacs dataset:
            # ts: task 1:9
            for ts in range(1, 10):
                # take (repeat) time 0:9
                for tk in range(0, 10):
                    for h in hands:
                        # ares = {}
                        name = 's{}ts{}tk{}{}'.format(arg.subject, ts, tk, h)
                        data_path = os.path.join(original_path, name + '_data.npy')
                        label_path = os.path.join(original_path, name + '_label.pkl')
                        processor.load_val_data(data_path, label_path, test_a=True, test_b=False)
                        processor.start()
                        # res['s{}ts{}tk{}{}'.format(arg.subject, ts, tk, h)] = ares
                        processor.update_f1_results()

        if arg.test_noise:
            print("Running on Noised Bimacs dataset")
            # for noised Bimacs dataset:
            processor.arg.model_args['dataset'] = 'bimacs'
            processor.load_model(i)
            for ts in range(1, 10):
                # take (repeat) time 0:9
                for tk in range(0, 10):
                    for h in hands:
                        name = 's{}ts{}tk{}{}'.format(arg.subject, ts, tk, h)
                        data_path = os.path.join(original_path, name + '_data.npy')
                        label_path = os.path.join(original_path, name + '_label.pkl')
                        processor.load_val_data(data_path, label_path, add_noise=True,
                                                noise_intensity=arg.noise_intensity,
                                                noise_type=arg.noise_type)
                        processor.start()
                        processor.update_f1_results()
        if arg.test_empty:
            print("Running on empty Bimacs dataset")
            # for empty Bimacs dataset:
            processor.arg.model_args['dataset'] = 'bimacs'
            processor.load_model(i)
            for ts in range(1, 10):
                # take (repeat) time 0:9
                for tk in range(0, 10):
                    for h in hands:
                        name = 's{}ts{}tk{}{}'.format(arg.subject, ts, tk, h)
                        data_path = os.path.join(original_path, name + '_data.npy')
                        label_path = os.path.join(original_path, name + '_label.pkl')
                        processor.load_val_data(data_path, label_path, is_empty=True)
                        processor.start()
                        # processor.update_f1_results()
        if arg.test_ikea:
            print("Running on IKEA dataset")
            # reload model for IKEA dataset with a new graph
            processor.arg.model_args['dataset'] = 'ikea'
            processor.f1_updated = False
            processor.load_model(i)
            # evaluate gaussian model on IKEA test dataset:
            folders_1 = os.listdir(data_path_b)
            for _ in folders_1:
                folders_2 = os.listdir(os.path.join(data_path_b + _))
                folders_2.sort(key=lambda x: int(x[1:4]))
                for __ in folders_2:
                    if "_3d" in data_path_b:
                        # 3D
                        data_path = os.path.join(data_path_b + _ + '/' + __ + '/3d_array_kalman.npy')
                        label_path = os.path.join(data_path_b + _ + '/' + __ + '/sub_label.pkl')
                    # 2D
                    else:
                        files = os.listdir(os.path.join(data_path_b + _ + '/' + __))
                        data_path = os.path.join(data_path_b + _ + '/' + __ + '/' + files[0])
                        label_path = os.path.join(data_path_b + _ + '/' + __ + '/' + files[1])
                    processor.load_val_data(data_path, label_path, test_b=True)
                    processor.start()
                    # processor.update_f1_results()
        if arg.test_a:
            processor.update_roc_by_features()
            processor.update_roc_by_softmax()
    # processor.plotting_results()
    processor.get_metrics_results()

    print('Done.')
