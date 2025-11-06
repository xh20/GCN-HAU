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
        self.A = torch.tensor(A, dtype=torch.float32).cuda(0)

        self.edge_importance = nn.ParameterList([
            nn.Parameter(torch.ones(A.shape))
            for i in range(10)
        ])

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
        new_c = y.size(1)
        y = y.view(N, new_c, -1)

        return y
