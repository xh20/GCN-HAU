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


def wrapped_conv(in_c, out_c, kernel_size, stride=(1, 1), padding=(0, 0), dilation=(1, 1), bias=True, groups=1):
    if type(padding) is not tuple:
        padding = (padding, 0)
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, 1)
    if type(stride) is not tuple:
        stride = (stride, 1)
    if type(dilation) is not tuple:
        dilation = (dilation, 1)

    conv = nn.Conv2d(in_c, out_c, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups,
                     bias=bias)
    return conv


class conv_unit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), dilation=(1, 1),
                 groups=1, bias=False, **kwargs):
        super(conv_unit, self).__init__()
        if type(kernel_size) is not tuple:
            kernel_size = (kernel_size, 1)
        if type(stride) is not tuple:
            stride = (stride, 1)
        if type(dilation) is not tuple:
            dilation = (dilation, 1)
        pad = (kernel_size[0] + (kernel_size[0] - 1) * (dilation[0] - 1) - 1) // 2
        self.conv = nn.Sequential(
            wrapped_conv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=(pad, 0),
                         groups=groups, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


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


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class MultiScale_TemporalConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilations=[1, 2, 3, 5],
                 residual=True,
                 residual_kernel_size=1):

        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'

        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size] * len(dilations)
        # Temporal Convolution branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    padding=0),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                TemporalConv(
                    branch_channels,
                    branch_channels,
                    kernel_size=ks,
                    stride=stride,
                    dilation=dilation),
            )
            for ks, dilation in zip(kernel_size, dilations)
        ])

        # Additional Max & 1x1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.BatchNorm2d(branch_channels)  # 为什么还要加bn
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride, 1)),
            nn.BatchNorm2d(branch_channels)
        ))

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)

        # initialize
        self.apply(weights_init)

    def forward(self, x):
        # Input dim: (N,C,T,V)
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        out += res
        return out


class Split_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=8, mid_reduction=1):
        super(Split_gcn, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels == 3 or in_channels == 9:
            self.rel_channels = 8
            self.mid_channels = 16
        else:
            self.rel_channels = in_channels // rel_reduction
            self.mid_channels = in_channels // mid_reduction
        self.mid_channels = 2*out_channels
        self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.ln1 = nn.GroupNorm(1, self.rel_channels)
        self.ln2 = nn.GroupNorm(1, self.rel_channels)
        # self.bn1 = nn.BatchNorm2d(self.rel_channels)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(self.out_channels)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x, A=None, alpha=1.0):
        x1, x3 = self.conv1(x), self.conv3(x)
        N, C, T, V = x1.shape
        # relative distance
        x1 = self.ln1(x1)
        x1 = x1.mean(-2)
        y1 = self.tanh(x1.unsqueeze(-1) - x1.unsqueeze(-2))

        # relative angles
        x2 = self.conv2(x)
        x2 = self.ln2(x2)
        x2 = x2.mean(-2)
        a1 = x2.unsqueeze(-1)
        a2 = x2.unsqueeze(-2)

        y2 = self.tanh(torch.matmul(a1, a2))

        y_sum = self.conv4(y1 + alpha*y2 + A.unsqueeze(0).unsqueeze(0)) if A is not None else 0

        y = torch.einsum('ncuv,nctv->nctu', y_sum, x3).contiguous()
        return y


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, alpha, coff_embedding=4, adaptive=True, residual=True):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.adaptive = adaptive
        self.num_subset = A.shape[0]
        self.convs = nn.ModuleList()
        for i in range(self.num_subset):
            self.convs.append(Split_gcn(in_channels, out_channels))

        if residual:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.down = lambda x: x
        else:
            self.down = lambda x: 0
        if self.adaptive:
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)).contiguous())
        else:
            self.A = Variable(torch.from_numpy(A.astype(np.float32)).contiguous(), requires_grad=False)
        self.alpha = nn.Parameter(alpha*torch.ones(self.num_subset))
        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1e-5)
        bn_init(self.bn, 1e-6)

    def forward(self, x):
        y = None
        if self.adaptive:
            A = self.PA
        else:
            A = self.A.cuda(x.get_device())
        for i in range(self.num_subset):
            z = self.convs[i](x, A[i], self.alpha[i])
            y = z + y if y is not None else z
        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)
        return y


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, alpha=0.1, stride=1, residual=True, adaptive=True, kernel_size=5,
                 dilations=[1, 2, 3, 5, 7, 9]):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A, alpha, adaptive=adaptive)
        self.tcn1 = MultiScale_TemporalConv(out_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                            dilations=dilations,
                                            residual=False)
        # self.cat1 = unit_cat(k_size=3)

        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        y = self.tcn1(self.gcn1(x))
        y = self.relu(y + self.residual(x))
        return y


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


class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, norm_layer=nn.BatchNorm2d):
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

    def forward(self, x):
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
