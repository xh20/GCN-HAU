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


class _DSConv(nn.Module):
    """Depthwise Separable Convolutions"""

    def __init__(self, dw_channels, out_channels, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), **kwargs):
        super(_DSConv, self).__init__()
        self.conv = nn.Sequential(
            conv_unit(dw_channels, dw_channels, kernel_size=kernel_size, stride=stride,
                      groups=dw_channels),
            conv_unit(dw_channels, out_channels, kernel_size=(1, 1))
        )

    def forward(self, x):
        out = self.conv(x)
        return out


class LinearBottleneck(nn.Module):
    """LinearBottleneck used in MobileNetV2"""

    def __init__(self, in_channels, out_channels, t=2, stride=2, **kwargs):
        super(LinearBottleneck, self).__init__()
        self.use_shortcut = stride == 1 and in_channels == out_channels
        self.block = nn.Sequential(
            # pw
            conv_unit(in_channels, in_channels * t, kernel_size=(1, 1), bias=False),
            # dw
            conv_unit(in_channels * t, in_channels * t, stride=stride, bias=False),
            # pw-linear
            wrapped_conv(in_channels * t, out_channels, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.block(x)
        if self.use_shortcut:
            out = x + out
        return out


class PyramidPooling(nn.Module):
    """Pyramid pooling module"""

    def __init__(self, in_channels, out_channels, num_point=26, **kwargs):
        super(PyramidPooling, self).__init__()
        inter_channels = in_channels // 4
        self.conv1 = conv_unit(in_channels, inter_channels, (1, num_point), **kwargs)
        self.conv2 = conv_unit(in_channels, inter_channels, (1, num_point), **kwargs)
        self.conv3 = conv_unit(in_channels, inter_channels, (1, num_point), **kwargs)
        self.conv4 = conv_unit(in_channels, inter_channels, (1, num_point), **kwargs)
        self.out = conv_unit(in_channels * 2, out_channels, (1, 1))

        self.pool1 = nn.AdaptiveAvgPool2d((1, num_point))
        self.pool2 = nn.AdaptiveAvgPool2d((2, num_point))
        self.pool3 = nn.AdaptiveAvgPool2d((3, num_point))
        self.pool4 = nn.AdaptiveAvgPool2d((6, num_point))

    def upsample(self, x, size):
        return F.interpolate(x, size, mode='bilinear', align_corners=True)

    def forward(self, x):
        # size = x.size()[2:]
        _, _, t, v = x.size()
        feat1 = self.upsample(self.conv1(self.pool1(x)), (t, v))
        feat2 = self.upsample(self.conv2(self.pool2(x)), (t, v))
        feat3 = self.upsample(self.conv3(self.pool3(x)), (t, v))
        feat4 = self.upsample(self.conv4(self.pool4(x)), (t, v))
        x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
        x = self.out(x)
        return x


class GlobalFeatureExtractor(nn.Module):
    """Global feature extractor module"""

    def __init__(self, in_channels=256, block_channels=(256, 384, 512),
                 out_channels=512, t=6, num_blocks=(3, 3, 3), num_point=26, residual=True, **kwargs):
        super(GlobalFeatureExtractor, self).__init__()
        self.residual = residual
        self.bottleneck1 = self._make_layer(LinearBottleneck, in_channels, block_channels[0], num_blocks[0], t, 1)
        self.bottleneck2 = self._make_layer(LinearBottleneck, block_channels[0], block_channels[1], num_blocks[1], t, 1)
        self.bottleneck3 = self._make_layer(LinearBottleneck, block_channels[1], block_channels[2], num_blocks[2], t, 1)
        self.ppm = PyramidPooling(block_channels[2], out_channels, num_point=num_point)
        if self.residual:
            self.res = wrapped_conv(in_channels, block_channels[2], (1, 1))

    def _make_layer(self, block, inplanes, planes, blocks, t=6, stride=1):
        layers = []
        layers.append(block(inplanes, planes, t, stride))
        for i in range(1, blocks):
            layers.append(block(planes, planes, t, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.bottleneck1(x)
        x1 = self.bottleneck2(x1)
        x1 = self.bottleneck3(x1)
        y = self.ppm(x1)
        if self.residual:
            y = y + self.res(x)
        return y


class FeatureFusionModule(nn.Module):
    """Feature fusion module"""

    def __init__(self, num_point, higher_in_channels=256, lower_in_channels=512, out_channels=512, scale_factor=4,
                 **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.num_point = num_point
        self.scale_factor = scale_factor
        self.dwconv = conv_unit(lower_in_channels, out_channels, ks=1)
        self.conv_lower_res = nn.Sequential(
            wrapped_conv(out_channels, out_channels, kernel_size=(1, 1)),
            nn.BatchNorm2d(out_channels)
        )
        self.conv_higher_res = nn.Sequential(
            wrapped_conv(higher_in_channels, out_channels, kernel_size=(1, 1)),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(True)

    def forward(self, higher_res_feature, lower_res_feature, size=(1000, 25)):
        lower_res_feature = F.interpolate(lower_res_feature, size=size, mode='bilinear', align_corners=True)
        lower_res_feature = self.dwconv(lower_res_feature)
        lower_res_feature = self.conv_lower_res(lower_res_feature)

        higher_res_feature = self.conv_higher_res(higher_res_feature)
        # print(lower_res_feature.shape[2])
        if higher_res_feature.shape[2] < lower_res_feature.shape[2]:
            higher_res_feature = F.interpolate(higher_res_feature, (lower_res_feature.shape[2], self.num_point),
                                               mode='bilinear', align_corners=True)
        else:
            lower_res_feature = F.interpolate(lower_res_feature, (higher_res_feature.shape[2], self.num_point),
                                              mode='bilinear', align_corners=True)
        out = higher_res_feature + lower_res_feature
        return self.relu(out)


class Classifer(nn.Module):
    """Classifer"""

    def __init__(self, in_channels, num_classes, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), **kwargs):
        super(Classifer, self).__init__()
        self.dsconv1 = _DSConv(in_channels, in_channels, kernel_size, stride, padding)
        self.dsconv2 = _DSConv(in_channels, in_channels)
        self.conv = nn.Sequential(
            # nn.Dropout(0.1),
            wrapped_conv(in_channels, num_classes, kernel_size=(1, 1))
        )
        self.feature = None

    def forward(self, x):
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        self.feature = x.clone().detach()
        x = self.conv(x)
        return x


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

        self.data_bn = nn.BatchNorm1d(num_person * in_channels * self.num_point)

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

        self.global_feature_extractor = GlobalFeatureExtractor(256, [256, 384, 512], 512, 2, [3, 3, 3],
                                                               num_point=num_point)
        self.feature_fusion = FeatureFusionModule(num_point, 256, 512, 512)
        self.classifier = Classifer(512, num_class, (3, num_point), padding=(1, 0), stride=(1, 1))
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

        # x: BS x 512 x 30 x 26
        x = self.global_feature_extractor(c3)

        # x: BS x 512 x 30 x 26
        x = self.feature_fusion(c3, x, size=(T, V))

        # x: BS x 512 x 60 x 26
        # x = F.interpolate(x, (T // 2, V), mode='bilinear', align_corners=True)

        # x: BS x 512 x 120 x 26
        # x = F.interpolate(x, (T, V), mode='bilinear', align_corners=True)

        # y: BS x 14 x 120 x 1
        y = self.classifier(x)

        # feature: BS x 512 x 60 x 26
        self.feature = self.classifier.feature.clone().detach()
        # x = F.interpolate(x, (int(T/2), 1), mode='bilinear', align_corners=True)
        # y = F.interpolate(x, (T, 1), mode='bilinear', align_corners=True)

        new_c = y.size(1)
        y = y.view(N, new_c, -1)

        return y
