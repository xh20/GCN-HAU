import math

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, padding, dilation, groups=inplanes, bias=bias)
        self.bn = norm_layer(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        nn.init.constant_(self.PA, 1e-6)
        self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.num_subset = num_subset

        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        self.conv_c = nn.ModuleList()
        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_c.append(nn.Conv1d(1, 1, kernel_size=1, padding=0, bias=False))
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

    def forward(self, x):
        N, C, T, V = x.size()

        A = self.A.cuda()
        # A = self.A
        A = A + self.PA

        # A = self.PA

        y = None
        for i in range(self.num_subset):
            A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)
            A2 = self.conv_b[i](x).view(N, self.inter_c * T, V)

            # A1 = self.soft(torch.matmul(A1, A2) / A1.size(-1))  # N V V

            A1 = torch.matmul(A1, A2) / A1.size(-1)
            A1 = A1.view(N, -1, V * V)
            A1 = self.conv_c[i](A1)
            A1 = A1.squeeze(1).contiguous().view(N, V, V)
            A1 = self.sigmoid(A1)

            A1 = A1 + A[i]
            A2 = x.view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        return self.relu(y)


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A)
        self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.tcn1(self.gcn1(x)) + self.residual(x)
        return self.relu(x)


class unit_jpu(nn.Module):
    def __init__(self, in_channels, width=64):
        super(unit_jpu, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels[0], width, kernel_size=(3, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels[1], width, kernel_size=(3, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels[2], width, kernel_size=(3, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True)
        )

        self.dilation1 = nn.Sequential(
            SeparableConv2d(3 * width, width, kernel_size=(3, 1), padding=(1, 0), dilation=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True)
        )
        self.dilation2 = nn.Sequential(
            SeparableConv2d(3 * width, width, kernel_size=(3, 1), padding=(1, 0), dilation=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True)
        )
        self.dilation3 = nn.Sequential(
            SeparableConv2d(3 * width, width, kernel_size=(3, 1), padding=(1, 0), dilation=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True)
        )
        self.dilation4 = nn.Sequential(
            SeparableConv2d(3 * width, width, kernel_size=(3, 1), padding=(1, 0), dilation=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True)
        )

    def forward(self, *inputs):
        feats = [self.conv3(inputs[-1]), self.conv2(inputs[-2]), self.conv1(inputs[-3])]
        _, _, t, v = feats[-1].size()
        feats[-2] = F.interpolate(feats[-2], (t, v))
        feats[-3] = F.interpolate(feats[-3], (t, v))
        feat = torch.cat(feats, dim=1)
        feat = torch.cat([self.dilation1(feat), self.dilation2(feat), self.dilation3(feat), self.dilation4(feat)],
                         dim=1)

        return feat


class PyramidPooling(nn.Module):
    def __init__(self, in_channels, num_point=26):
        super(PyramidPooling, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d((1, num_point))
        self.pool2 = nn.AdaptiveAvgPool2d((2, num_point))
        self.pool3 = nn.AdaptiveAvgPool2d((3, num_point))
        self.pool4 = nn.AdaptiveAvgPool2d((6, num_point))

        out_channels = int(in_channels / 4)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, num_point), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, num_point), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, num_point), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, num_point), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        _, _, t, v = x.size()
        feat1 = F.interpolate(self.conv1(self.pool1(x)), (t, v))
        feat2 = F.interpolate(self.conv2(self.pool2(x)), (t, v))
        feat3 = F.interpolate(self.conv3(self.pool3(x)), (t, v))
        feat4 = F.interpolate(self.conv4(self.pool4(x)), (t, v))
        return torch.cat((x, feat1, feat2, feat3, feat4), 1)


class unit_PSPHead(nn.Module):
    def __init__(self, in_channels, out_channels, num_point=26):
        super(unit_PSPHead, self).__init__()
        inter_channels = in_channels // 4
        self.convpsp = nn.Sequential(
            PyramidPooling(in_channels, num_point),
            nn.Conv2d(in_channels * 2, inter_channels, (3, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(True),
            # nn.Dropout2d(0.1, False),
            nn.Conv2d(inter_channels, out_channels, (1, num_point))
        )

    def forward(self, x):
        return self.convpsp(x)


class Model(nn.Module):
    def __init__(self, num_class=14, num_point=26, num_point_b=20,
                 num_person=1, dataset='bimacs', graph=None,
                 graph_b=None, graph_args=dict(), coeff=0.0,
                 in_channels=3, drop_out=0):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            if dataset == 'bimacs':
                Graph = import_class(graph)
                self.graph = Graph(**graph_args)
                A = self.graph.A
            else:
                Graph_b = import_class(graph_b)
                self.graph_b = Graph_b(**graph_args)
                A = np.zeros((3, self.num_point, self.num_point))
                A[:, 0:num_point_b, 0:num_point_b] = self.graph_b.A

        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        self.l1 = TCN_GCN_unit(in_channels, 64, A, residual=False)
        self.l2 = TCN_GCN_unit(64, 64, A)
        self.l3 = TCN_GCN_unit(64, 64, A)
        self.l4 = TCN_GCN_unit(64, 64, A)
        self.l5 = TCN_GCN_unit(64, 128, A, stride=2)
        self.l6 = TCN_GCN_unit(128, 128, A)
        self.l7 = TCN_GCN_unit(128, 128, A)
        self.l8 = TCN_GCN_unit(128, 256, A, stride=2)
        self.l9 = TCN_GCN_unit(256, 256, A)
        self.l10 = TCN_GCN_unit(256, 256, A)

        self.jpu = unit_jpu([64, 128, 256], width=64)
        self.head = unit_PSPHead(256, num_class, num_point)
        self.feature = None

        # self.fc1 = nn.Linear(256, num_class)
        # nn.init.normal_(self.fc1.weight, 0, math.sqrt(2. / num_class))
        # self.fc2 = nn.Linear(256, 2)
        # nn.init.normal_(self.fc2.weight, 0, math.sqrt(2. / 2))
        # bn_init(self.data_bn, 1)

        # if drop_out:
        #     self.drop_out = nn.Dropout(drop_out)
        # else:
        #     self.drop_out = lambda x: x

    def forward(self, data):
        x = data['motion']
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        c1 = self.l4(x)
        c2 = self.l5(c1)
        c2 = self.l6(c2)
        c2 = self.l7(c2)
        c3 = self.l8(c2)
        c3 = self.l9(c3)
        c3 = self.l10(c3)

        # feat: bs*M x 64*4 x T x V
        feat = self.jpu(c1, c2, c3)
        self.feature = feat.clone().detach().mean(3)

        # y: bs*M x class x T x 1
        y = self.head(feat)
        # y = y.mean(3)
        new_c = y.size(1)
        # y: bs*M x class x T
        y = y.view(N, new_c, -1)

        out = {
            "pred_motion": y
        }
        return out
