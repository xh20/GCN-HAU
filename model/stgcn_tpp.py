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


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)


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
            SeparableConv2d(3*width, width, kernel_size=(3, 1), padding=(1, 0), dilation=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True)
            )
        self.dilation2 = nn.Sequential(
            SeparableConv2d(3*width, width, kernel_size=(3, 1), padding=(1, 0), dilation=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True)
            )
        self.dilation3 = nn.Sequential(
            SeparableConv2d(3*width, width, kernel_size=(3, 1), padding=(1, 0), dilation=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True)
            )
        self.dilation4 = nn.Sequential(
            SeparableConv2d(3*width, width, kernel_size=(3, 1), padding=(1, 0), dilation=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True)
            )

    def forward(self, *inputs):
        feats = [self.conv3(inputs[-1]), self.conv2(inputs[-2]), self.conv1(inputs[-3])]
        _, _, t, v = feats[-1].size()
        feats[-2] = F.interpolate(feats[-2], (t, v))
        feats[-3] = F.interpolate(feats[-3], (t, v))
        feat = torch.cat(feats, dim=1)
        feat = torch.cat([self.dilation1(feat), self.dilation2(feat), self.dilation3(feat), self.dilation4(feat)], dim=1)

        return feat


class unit_fcnhead(nn.Module):
    def __init__(self, in_channels, out_channels, num_point):
        super(unit_fcnhead, self).__init__()
        inter_channels = in_channels // 4
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=(3, num_point), padding=(1, 0), bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Conv2d(inter_channels, out_channels, kernel_size=(1, 1))
        )

    def forward(self, x):
        return self.conv(x)


class ConvTemporalGraphical(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

    def forward(self, x, A):
        assert A.shape[0] == self.kernel_size

        x = self.conv(x)

        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))

        return x.contiguous(), A


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


class st_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dropout=0, residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res

        return self.relu(x)


class Model(nn.Module):
    def __init__(self, num_class=14, num_point=26, num_point_b=20, num_person=1, dataset='bimacs', graph=None, graph_b=None,
                 graph_args=dict(), in_channels=3, drop_out=0):
        super(Model, self).__init__()

        self.num_point = num_point
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
        self.A = torch.tensor(A, dtype=torch.float32).cuda(0)
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        spatial_kernel_size = A.shape[0]
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)

        self.l1 = st_gcn(in_channels, 64, kernel_size, residual=False)
        self.l2 = st_gcn(64, 64, kernel_size)
        self.l3 = st_gcn(64, 64, kernel_size)
        self.l4 = st_gcn(64, 64, kernel_size)
        self.l5 = st_gcn(64, 128, kernel_size, stride=2)
        self.l6 = st_gcn(128, 128, kernel_size)
        self.l7 = st_gcn(128, 128, kernel_size)
        self.l8 = st_gcn(128, 256, kernel_size, stride=2)
        self.l9 = st_gcn(256, 256, kernel_size)
        self.l10 = st_gcn(256, 256, kernel_size)

        self.jpu = unit_jpu([64, 128, 256], width=64)
        self.head = unit_PSPHead(256, num_class, num_point)
        self.feature = None
        self.edge_importance = nn.ParameterList([
            nn.Parameter(torch.ones(A.shape))
            for i in range(10)
        ])

        # self.fc1 = nn.Linear(256, num_class)
        # nn.init.normal_(self.fc1.weight, 0, math.sqrt(2. / num_class))
        # self.fc2 = nn.Linear(256, 2)
        # nn.init.normal_(self.fc2.weight, 0, math.sqrt(2. / 2))
        # bn_init(self.data_bn, 1)

        # if drop_out:
        #     self.drop_out = nn.Dropout(drop_out)
        # else:
        #     self.drop_out = lambda x: x

    def forward(self, x):
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        x = self.l1(x, self.A * self.edge_importance[0])
        x = self.l2(x, self.A * self.edge_importance[1])
        x = self.l3(x, self.A * self.edge_importance[2])
        c1 = self.l4(x, self.A * self.edge_importance[3])
        c2 = self.l5(c1, self.A * self.edge_importance[4])
        c2 = self.l6(c2, self.A * self.edge_importance[5])
        c2 = self.l7(c2, self.A * self.edge_importance[6])
        c3 = self.l8(c2, self.A * self.edge_importance[7])
        c3 = self.l9(c3, self.A * self.edge_importance[8])
        c3 = self.l10(c3, self.A * self.edge_importance[9])

        feat = self.jpu(c1, c2, c3)
        # feat = feat.mean(3)

        y = self.head(feat)
        new_c = y.size(1)
        y = y.view(N, new_c, -1)
        self.feature = y
        # N*M,C,T,V
        # c_new = x.size(1)
        # x = x.view(N, M, c_new, -1)
        # x = x.mean(3).mean(1)
        # x = self.drop_out(x)
        # return self.fc1(x), self.fc2(x)

        return y
