#!/usr/bin/env python
from __future__ import print_function
import sys

import argparse
import inspect
import os
import pickle
import random
import shutil
import time
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
from torch.optim.lr_scheduler import LRScheduler
from tqdm import tqdm

from utils.framewise_loss import (SegmentationLoss, SmoothLoss, TemporalMultiClassDiceLoss, TaskGraphLoss,
                                  TaskGraphDistanceLoss)
from utils.mixup_tools import mixup_data


class GradualWarmupScheduler(LRScheduler):
    def __init__(self, optimizer, total_epoch, after_scheduler=None):
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        self.last_epoch = -1
        super().__init__(optimizer)

    def get_lr(self):
        return [base_lr * (self.last_epoch + 1) / self.total_epoch for base_lr in self.base_lrs]

    def step(self, epoch=None, metric=None):
        if self.last_epoch >= self.total_epoch - 1:
            if metric is None:
                return self.after_scheduler.step(epoch)
            else:
                return self.after_scheduler.step(metric, epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)


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

    parser.add_argument('-model_save_dir', default='')
    parser.add_argument('-model_save_name', default='')
    parser.add_argument(
        '--config',
        default='./config/nturgbd-cross-view/test_bone.yaml',
        help='path to the configuration file')

    # processor
    parser.add_argument(
        '--phase', default='train', help='must be train, val or test')
    parser.add_argument(
        '--save-score',
        type=str2bool,
        default=False,
        help='if ture, the classification score will be stored')

    # visulize and debug
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed for pytorch')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='the interval for printing messages (#iteration)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=1,
        help='the interval for storing models (#iteration)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=5,
        help='the interval for evaluating models (#iteration)')
    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not')
    parser.add_argument(
        '--show-topk',
        type=int,
        default=[1],
        nargs='+',
        help='which Top K accuracy will be shown')

    # feeder
    parser.add_argument(
        '--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=32,
        help='the number of worker for data loader')
    parser.add_argument(
        '--train-feeder-args',
        default=dict(),
        help='the arguments of data loader for training')
    parser.add_argument(
        '--test-feeder-args',
        default=dict(),
        help='the arguments of data loader for test')
    parser.add_argument(
        '--val-feeder-args',
        default=dict(),
        help='the arguments of data loader for validation')

    # model
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
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')

    # optim
    parser.add_argument(
        '--base-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument(
        '--step',
        type=int,
        default=[20, 40, 60],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument(
        '--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument(
        '--batch-size', type=int, default=256, help='training batch size')
    parser.add_argument(
        '--val-batch-size', type=int, default=256, help='validation batch size')
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch')
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=80,
        help='stop training in which epoch')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0005,
        help='weight decay for optimizer')
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.2,
        help='alpha of beta distribution for data mix up')
    parser.add_argument(
        '--only-objects',
        type=bool,
        default=False,
        help='only mix objects')
    parser.add_argument(
        '--only-skeleton',
        type=bool,
        default=False,
        help='only mix objects')
    parser.add_argument(
        '--index-start',
        type=int,
        default=12,
        help='the start index of object nodes, bimacs: 12, behave: 22')
    parser.add_argument('--only_train_part', default=False)
    parser.add_argument('--only_train_epoch', default=0)
    parser.add_argument('--warm_up_epoch', default=0)
    return parser


class Processor():
    """ 
        Processor for Skeleton-based Action Recgnition
    """

    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        if arg.phase == 'train':
            if not arg.train_feeder_args['debug']:
                if os.path.isdir(arg.model_save_dir):
                    print('log_dir: ', arg.model_save_dir, 'already exist')
                    # answer = input('delete it? y/n:')
                    # if answer == 'y':
                    #     shutil.rmtree(arg.model_save_dir)
                    #     print('Dir removed: ', arg.model_save_dir)
                    #     input('Refresh the website of tensorboard by pressing any keys')
                    # else:
                    #     print('Dir not removed: ', arg.model_save_dir)
                self.train_writer = SummaryWriter(os.path.join(arg.model_save_dir, 'train'), 'train')
                self.val_writer = SummaryWriter(os.path.join(arg.model_save_dir, 'val'), 'val')
            else:
                self.train_writer = self.val_writer = SummaryWriter(os.path.join(arg.model_save_dir, 'test'), 'test')
        self.global_step = 0
        self.load_model()
        self.load_optimizer()
        self.load_data()
        self.lr = self.arg.base_lr
        self.best_acc = 0

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                drop_last=True,
                worker_init_fn=init_seed)
        self.data_loader['val'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.val_feeder_args),
            batch_size=self.arg.val_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed)

    def load_model(self):
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.output_device = output_device
        Model = import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), os.path.join(str(self.arg.work_dir), str(self.arg.model_save_name)))
        # print(Model)
        self.model = Model(**self.arg.model_args).cuda(output_device)
        print("Number of parameters:", sum([param.nelement() for param in self.model.parameters()]))
        # print(self.model)
        # self.loss_motion = SegmentationLoss(num_class=self.arg.model_args['num_class']).cuda(output_device)
        self.loss_smooth = SmoothLoss().cuda(output_device)
        self.loss_task = TaskGraphDistanceLoss(num_class=self.arg.model_args['num_class']).cuda(output_device)
        # self.loss_task = TaskGraphLoss(num_class=self.arg.model_args['num_class']).cuda(output_device)
        self.loss_motion = nn.CrossEntropyLoss(ignore_index=-1).cuda(output_device)

        if self.arg.weights:
            #self.global_step = int(arg.weights[:-3].split('-')[-1])
            self.print_log('Load weights from {}.'.format(self.arg.weights))
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)

            weights = OrderedDict(
                [[k.split('module.')[-1],
                  v.cuda(output_device)] for k, v in weights.items()])

            keys = list(weights.keys())
            for w in self.arg.ignore_weights:
                for key in keys:
                    if w in key:
                        if weights.pop(key, None) is not None:
                            self.print_log('Sucessfully Remove Weights: {}.'.format(key))
                        else:
                            self.print_log('Can Not Remove Weights: {}.'.format(key))

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

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'AdamW':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.arg.base_lr*0.1,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

        lr_scheduler_pre = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=self.arg.step, gamma=0.1)

        self.lr_scheduler = GradualWarmupScheduler(self.optimizer, total_epoch=self.arg.warm_up_epoch,
                                                   after_scheduler=lr_scheduler_pre)
        self.print_log('using warm up, epoch: {}'.format(self.arg.warm_up_epoch))

    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            yaml.dump(arg_dict, f)

    def adjust_learning_rate(self, epoch):
        if self.arg.optimizer in ['SGD', 'Adam', 'AdamW']:
            if epoch < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * 0.5 * (1 + np.cos(
                    np.pi * (epoch - self.arg.warm_up_epoch) / (self.arg.num_epoch - self.arg.warm_up_epoch)))
            # else:
            #     lr = self.arg.base_lr * (
            #             0.1 ** np.sum(epoch >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError()

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(os.path.join(self.arg.work_dir, self.arg.model_save_name)), 'a') as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def train(self, epoch, save_model=False):
        self.model.train()
        self.print_log('Training epoch: {}'.format(epoch))
        loader = self.data_loader['train']
        self.adjust_learning_rate(epoch)
        # for name, param in self.model.named_parameters():
        #     self.train_writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
        loss_value = []
        self.train_writer.add_scalar('epoch', epoch, self.global_step)
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        process = tqdm(loader)
        if self.arg.only_train_part:
            if epoch > self.arg.only_train_epoch:
                print('only train part, require grad')
                for key, value in self.model.named_parameters():
                    if 'PA' in key:
                        value.requires_grad = True
                        # print(key + '-require grad')
            else:
                print('only train part, do not require grad')
                for key, value in self.model.named_parameters():
                    if 'PA' in key:
                        value.requires_grad = False
                        # print(key + '-not require grad')
        for batch_idx, data in enumerate(process):
            self.global_step += 1
            # get data
            data["motion"] = Variable(data["motion"].float().cuda(self.output_device), requires_grad=False)
            if "image" in list(data.keys()):
                data["image"] = Variable(data["image"].float().cuda(self.output_device), requires_grad=False)/255.0
                # data["masked_image"] = Variable(data["masked_image"].float().cuda(self.output_device), requires_grad=False)/255.0
            label = data["label"]
            index = data["index"]
            # label = Variable(torch.from_numpy(np.array(label, dtype=np.int_)).cuda(self.output_device), requires_grad=False)
            if isinstance(label, list) or isinstance(label, np.ndarray):
                label = np.array(label)
                label = torch.stack(label).cuda(device=self.output_device)
                label = label.transpose(1, 0)
            elif isinstance(label, torch.Tensor):
                label = label.cuda(self.output_device)
            if label.dim() == 2:
                label = label.to(torch.long)
            elif label.dim() == 3:  # onehot
                label = label.to(torch.float)
            timer['dataloader'] += self.split_time()

            mix_data, label_a, label_b, lam = mixup_data(data, label, only_objects=self.arg.only_objects,
                                                         only_skeleton=self.arg.only_skeleton,
                                                         alpha=self.arg.alpha, index_start=self.arg.index_start)
            # forward
            output = self.model(mix_data)
            # if batch_idx == 0 and epoch == 0:
            #     self.train_writer.add_graph(self.model, output)
            if isinstance(output, tuple):
                output, l1 = output
                l1 = l1.mean()
            else:
                l1 = 0
            loss_smooth = self.loss_smooth(output["pred_motion"])
            if "pred_task" in list(output.keys()):
                loss_task = self.loss_task(output["pred_task"], label)
                self.train_writer.add_scalar('loss_task', loss_task, self.global_step)
            loss_motion_a = self.loss_motion(output["pred_motion"], label_a)
            loss_motion_b = self.loss_motion(output["pred_motion"], label_b)
            loss_motion = loss_motion_a*lam + (1-lam)*loss_motion_b + l1
            # loss = loss_motion + loss_smooth + loss_task
            # loss = loss_motion + loss_smooth
            # loss = loss_motion + loss_task
            loss = loss_motion
            # backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            loss_value.append(loss.data.item())
            timer['model'] += self.split_time()

            value, predict_label = torch.max(output["pred_motion"].data, 1)
            if label.ndim == 3:
                true_classes = torch.argmax(label, dim=1)
            elif label.ndim == 2:
                true_classes = label
            acc = torch.mean((predict_label == true_classes.data).float())
            self.train_writer.add_scalar('acc', acc, self.global_step)
            self.train_writer.add_scalar('loss', loss.data.item(), self.global_step)
            self.train_writer.add_scalar('loss_motion', loss_motion.data.item(), self.global_step)
            self.train_writer.add_scalar('loss_l1', l1, self.global_step)
            self.train_writer.add_scalar('loss_smooth', loss_smooth, self.global_step)
            # self.train_writer.add_scalar('batch_time', process.iterable.last_duration, self.global_step)

            # statistics
            self.lr = self.optimizer.param_groups[0]['lr']
            self.train_writer.add_scalar('lr', self.lr, self.global_step)
            # if self.global_step % self.arg.log_interval == 0:
            #     self.print_log(
            #         '\tBatch({}/{}) done. Loss: {:.4f}  lr:{:.6f}'.format(
            #             batch_idx, len(loader), loss.data[0], lr))
            timer['statistics'] += self.split_time()

        # statistics of time consumption and loss
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        self.print_log(
            '\tMean training loss: {:.4f}.'.format(np.mean(loss_value)))
        self.print_log(
            '\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(
                **proportion))

        if save_model:
            state_dict = self.model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1],
                                    v.cpu()] for k, v in state_dict.items()])

            torch.save(weights, os.path.join(self.arg.model_save_dir, self.arg.model_save_name + '-' + str(epoch) + '-'
                                             + str(int(self.global_step)) + '.pt'))

    def eval(self, epoch, save_score=False, loader_name=['val'], wrong_file=None, result_file=None):
        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')
        if isinstance(self.model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            model = self.model.module
        else:
            model = self.model
        model = model.cuda(self.output_device)
        model.eval()
        self.print_log('Eval epoch: {}'.format(epoch))
        for ln in loader_name:
            loss_value = []
            loss_task_value = []
            loss_smooth_value = []
            loss_motion_value = []
            score_frag = []
            accs = []
            step = 0
            process = tqdm(self.data_loader[ln])
            for batch_idx, data in enumerate(process):
                with torch.no_grad():
                    data["motion"] = Variable(data["motion"].float().cuda(self.output_device), requires_grad=False)
                    if "image" in list(data.keys()):
                        data["image"] = Variable(data["image"].float().cuda(self.output_device), requires_grad=False) / 255.0
                    # data["masked_image"] = Variable(data["masked_image"].float().cuda(self.output_device), requires_grad=False) / 255.0
                    label = data["label"]
                    index = data["index"]
                    if isinstance(label, list):
                        label = torch.stack(label).cuda(device=self.output_device)
                        label = label.transpose(1, 0)
                    elif isinstance(label, torch.Tensor):
                        label = label.cuda(device=self.output_device)
                    if label.dim() == 2:
                        label = label.to(torch.long)
                    elif label.dim() == 3:
                        label = label.to(torch.float)
                    output = model(data)
                    if isinstance(output, tuple):
                        output, l1 = output
                        l1 = l1.mean()
                    else:
                        l1 = 0
                    # label = label.squeeze()
                    loss_motion = self.loss_motion(output["pred_motion"], label) + l1
                    loss_smooth = self.loss_smooth(output["pred_motion"])
                    if "pred_task" in output:
                        loss_task = self.loss_task(output["pred_task"], label, onehot=False)
                        loss_task_value.append(loss_task.data.item())
                    else:
                        loss_task = 0.0
                        loss_task_value.append(loss_task)

                    value, predict_label = torch.max(output["pred_motion"].data, 1)
                    step += 1
                    score_frag.append(predict_label.data.cpu().numpy())
                    loss_motion_value.append(loss_motion.data.item())
                    loss_smooth_value.append(loss_smooth.data.item())
                    # acc = torch.mean((predict_label == label.data).float())
                    # accs.append(acc.data.item())

                if wrong_file is not None or result_file is not None:
                    predict = list(predict_label.cpu().numpy())
                    true = list(label.data.cpu().numpy())
                    for i, x in enumerate(predict):
                        if result_file is not None:
                            f_r.write(str(x.tolist()) + '\n' + str(true[i].tolist()) + '\n')
                        # if x != true[i] and wrong_file is not None:
                            # f_w.write(str(index[i]) + ',' + str(x) + ',' + str(true[i]) + '\n')
            score = np.concatenate(score_frag)
            loss_motion = np.mean(loss_motion_value)
            accuracy = self.data_loader[ln].dataset.acc_seg(score, 1)
            if accuracy > self.best_acc:
                self.best_acc = accuracy
                self.best_loss = loss_motion
                self.best_epoch = epoch
                self.best_model = os.path.join(self.arg.model_save_dir, self.arg.model_save_name + '-' + str(epoch) + '-' + str(int(self.global_step)) + '.pt')
                state_dict = model.state_dict()
                weights = OrderedDict([[k.split('module.')[-1],
                                        v.cpu()] for k, v in state_dict.items()])

                torch.save(weights, os.path.join(self.arg.model_save_dir, self.arg.model_save_name + '_best.pt'))

            self.lr_scheduler.step()
            print('\nAccuracy: ', accuracy, ' model: ', self.arg.model_save_name)

            if self.arg.phase == 'train':
                self.val_writer.add_scalar('loss_motion', loss_motion, self.global_step)
                if "loss_task" in list(output.keys()):
                    self.val_writer.add_scalar('loss_task', np.mean(loss_task_value), self.global_step)
                self.val_writer.add_scalar('loss_smooth', np.mean(loss_smooth_value), self.global_step)
                self.val_writer.add_scalar('loss_l1', l1, self.global_step)
                self.val_writer.add_scalar('acc', accuracy, self.global_step)

            self.print_log('\tMean {} loss of {} batches: {}.'.format(ln, len(self.data_loader[ln]), loss_motion))

            for k in self.arg.show_topk:
                self.print_log('\tTop{}: {:.2f}%'.format(
                    k, 100 * self.data_loader[ln].dataset.acc_seg(score, k)))
                if k > 1:
                    self.print_log('\tTop{}: {:.2f}%'.format(
                        k, 100 * self.data_loader[ln].dataset.top_k(score, k)))

            if save_score:
                score_dict = dict(zip(self.data_loader[ln].dataset.sample_name, score))
                with open('{}/epoch{}_{}_score.pkl'.format(
                        self.arg.work_dir, epoch + 1, ln), 'wb') as f:
                    pickle.dump(score_dict, f)

    def start(self):
        if self.arg.phase == 'train':
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size
            if self.arg.weights:
                self.eval(self.arg.start_epoch - 1, save_score=self.arg.save_score, loader_name=['val'])

            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                if self.lr < 1e-6:
                    break
                save_model = ((epoch + 1) % self.arg.save_interval == 0) or (
                        epoch + 1 == self.arg.num_epoch)

                self.train(epoch, save_model=False)

                self.eval(epoch, save_score=self.arg.save_score, loader_name=['val'])
            print('best accuracy: ', self.best_acc, ' at epoach: ', self.best_epoch, '\nbest model: ', self.best_model)
            self.print_log('\tThe best model is at {} epoch, with accuracy {}, loss {}.'.format(
                self.best_epoch, self.best_acc, self.best_loss))

        elif self.arg.phase == 'test':
            if not self.arg.val_feeder_args['debug']:
                wf = self.arg.model_save_name + '_wrong.txt'
                rf = self.arg.model_save_name + '_right.txt'
            else:
                wf = rf = None
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.arg.print_log = False
            self.print_log('Model:   {}.'.format(self.arg.model))
            self.print_log('Weights: {}.'.format(self.arg.weights))
            self.eval(epoch=0, save_score=self.arg.save_score, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.print_log('Done.\n')


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


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
    arg.model_save_dir = os.path.join(arg.model_save_dir, arg.model_save_name)
    init_seed(0)
    processor = Processor(arg)
    processor.start()
    sys.exit(0)
