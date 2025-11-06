#!/usr/bin/env python
from __future__ import print_function

import argparse
import inspect
import os
import pickle
import random
import shutil
import time
import json
from collections import OrderedDict

import numpy as np
# torch
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torchvision import transforms
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm

from utils.framewise_loss import SegmentationLoss


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
        '--log_name',
        default='log.json',
        help='the log save name')
    parser.add_argument('-model_saved_name', default='')
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
        '--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument(
        '--test-feeder-args',
        default=dict(),
        help='the arguments of data loader for test')
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument(
        '--model-args',
        type=dict,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--weights',
        default=None,
        help='the weights for network initialization')
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')
    parser.add_argument(
        '--test-batch-size', type=int, default=1, help='test batch size')
    return parser


class Processor():
    """
        Processor for Skeleton-based Action Recgnition
    """

    def __init__(self, arg, data_path, label_path):
        self.arg = arg
        self.load_model()
        self.load_data(data_path, label_path)

    def load_data(self, data_path, label_path):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(data_path=data_path, label_path=label_path),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
            worker_init_fn=init_seed)

    def load_model(self):
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.output_device = output_device
        Model = import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        self.model = Model(**self.arg.model_args).cuda(output_device)
        # self.loss = SegmentationLoss().cuda(output_device)
        self.loss = nn.CrossEntropyLoss(ignore_index=-1).cuda(output_device)
        if self.arg.weights:
            weights = torch.load(self.arg.weights)
            weights = OrderedDict(
                [[k.split('module.')[-1],
                  v.cuda(output_device)] for k, v in weights.items()])
            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)
        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.arg.device,
                    output_device=output_device)

    def eval(self, loader_name=['test']):
        self.model.eval()
        for ln in loader_name:
            loss_value = []
            score_frag = []
            for _, (data, label, _) in enumerate(self.data_loader[ln]):
                with torch.no_grad():
                    data = Variable(
                        data.float().cuda(self.output_device),
                        requires_grad=False)
                    label = Variable(
                        torch.tensor(label, dtype=int).unsqueeze(0).cuda(self.output_device),
                        requires_grad=False)
                    # reshape label
                    if label.shape[0] != 1:
                        label = label.unsqueeze(0)
                    output = self.model(data)
                    loss = self.loss(output, label)
                    _, predict_label = torch.max(output.data, 1)
                    label = label.data.cpu().numpy().squeeze()
                    predict_label = predict_label.data.cpu().numpy().squeeze()[label >= 0]
                    score_frag.append(predict_label)
                    loss_value.append(loss.data.item())
                    label = label[label >= 0]

                predict = predict_label.tolist()
                true = label.tolist()
                ares['predict'] = predict
                ares['ground_truth'] = true
                # for i, x in enumerate(predict):

            score = np.concatenate(score_frag)
            loss = np.mean(loss_value)
            accuracy = self.data_loader[ln].dataset.acc_seg(score, 1, label)
            ares['accuracy'] = accuracy
            ares['loss'] = loss

    def start(self):
        if self.arg.phase == 'test':
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.eval(loader_name=['test'])


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
    res = {}
    print("weight: ", arg.weights.split('/')[-1])
    print("model: ", arg.model)

    folders_1 = os.listdir(original_path)
    for _ in folders_1:
        folders_2 = os.listdir(os.path.join(original_path + _))
        folders_2.sort(key=lambda x: int(x[1:4]))
        for __ in tqdm(folders_2):
            files = os.listdir(os.path.join(original_path + _ + '/' + __))
            data_path = os.path.join(original_path + _ + '/' + __ + '/' + files[0])
            label_path = os.path.join(original_path + _ + '/' + __ + '/' + files[1])
            ares = {}
            processor = Processor(arg, data_path, label_path)
            processor.start()
            index = os.path.join(_ + '/' + __)
            res[index] = ares

    resfile = arg.log_name
    print('save file:', resfile)
    with open(resfile, 'wt') as f:
        json.dump(res, f)
    print('Done.')