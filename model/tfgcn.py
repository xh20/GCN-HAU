import math

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

# from layers.spectral_normalization.spectral_norm_conv_inplace import spectral_norm_conv
COEFF = 0.0


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


def conv_negative_init(conv):
    conv.weight.data.fill_(-1.0)
    nn.init.constant_(conv.bias, 0)


def conv_uniform_init(conv):
    n = conv.in_features
    y = 1.0 / np.sqrt(n)
    conv.weight.data.uniform_(-y, y)
    conv.bias.data.fill_(0)


def conv_trunc_norm_init(conv):
    nn.init.trunc_normal_(conv.weight, mean=-0.5)
    nn.init.constant_(conv.bias, 0)


def conv_norm_init(conv):
    nn.init.normal_(conv.weight, mean=-0.5)
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


def wrapped_conv(in_c, out_c, kernel_size, stride=(1, 1), padding=(0, 0), dilation=(1, 1), coeff=None,
                 bias=True, groups=1):
    conv = nn.Conv2d(in_c, out_c, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups,
                     bias=bias)
    return conv


class conv_unit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), dilation=(1, 1), coeff=None,
                 groups=1, bias=True, bn=True, relu=True, pad=None, **kwargs):
        super(conv_unit, self).__init__()
        if type(kernel_size) is not tuple:
            kernel_size = (kernel_size, 1)
        if type(stride) is not tuple:
            stride = (stride, 1)
        if type(dilation) is not tuple:
            dilation = (dilation, 1)
        if pad is None:
            pad = ((kernel_size[0] + (kernel_size[0] - 1) * (dilation[0] - 1) - 1) // 2, 0)

        self.conv = nn.Sequential(
            wrapped_conv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=pad,
                         groups=groups, dilation=dilation, bias=bias, coeff=coeff)
        )
        if bn:
            self.conv.append(nn.BatchNorm2d(out_channels))
        if relu:
            self.conv.append(nn.ReLU(True))

    def forward(self, x):
        return self.conv(x)


class CTRGC(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=8, mid_reduction=1):
        super(CTRGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels <= 8:
            self.rel_channels = 8
            self.mid_channels = 16
        else:
            self.rel_channels = in_channels // rel_reduction
            self.mid_channels = in_channels // mid_reduction
        # conv1 - conv4: Conv2d, bias, no BN, ReLU
        self.conv1 = wrapped_conv(self.in_channels, self.rel_channels, kernel_size=(1, 1))
        self.conv2 = wrapped_conv(self.in_channels, self.rel_channels, kernel_size=(1, 1))
        self.conv3 = wrapped_conv(self.in_channels, self.out_channels, kernel_size=(1, 1))
        self.conv4 = wrapped_conv(self.rel_channels, self.out_channels, kernel_size=(1, 1))
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if in_channels == 3:
                    conv_norm_init(m)
                else:
                    conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)


    def forward(self, x, A=None, alpha=1):
        x1, x2, x3 = self.conv1(x).mean(-2), self.conv2(x).mean(-2), self.conv3(x)
        x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
        x1 = self.conv4(x1) * alpha + (A.unsqueeze(0).unsqueeze(0) if A is not None else 0)  # N,C,V,V
        x1 = torch.einsum('ncuv,nctv->nctu', x1, x3)
        return x1


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, adaptive=True, residual=True):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.adaptive = adaptive
        self.num_subset = A.shape[0]
        self.convs = nn.ModuleList()
        for i in range(self.num_subset):
            self.convs.append(CTRGC(in_channels, out_channels))

        if residual:
            # res: bias, Conv2d, BN, no ReLU
            if in_channels != out_channels:
                self.residual = conv_unit(in_channels, out_channels, kernel_size=(1, 1),
                                          bias=True, bn=True, relu=False)
            else:
                self.residual = lambda x: x
        else:
            self.residual = lambda x: 0
        if self.adaptive:
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)).contiguous())
        else:
            self.A = Variable(torch.from_numpy(A.astype(np.float32)).contiguous(), requires_grad=False)
        self.alpha = nn.Parameter(torch.ones(self.num_subset))
        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if in_channels == 3:
                    conv_norm_init(m)
                else:
                    conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1e-5)

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
        y += self.residual(x)
        y = self.relu(y)
        return y


class MultiScale_TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 dilations=[1, 2, 3, 5], residual=True, residual_kernel_size=1):

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
            # 1. unit: Conv2d, bias, BN and ReLU;
            # 2. unit: Conv2d, bias, BN; ReLU
            nn.Sequential(
                conv_unit(in_channels, branch_channels, kernel_size=(1, 1),
                          bias=False, bn=True, relu=True),
                conv_unit(branch_channels, branch_channels,
                          kernel_size=ks, stride=stride, dilation=dilation,
                          bias=False, bn=True, relu=True),
            )
            for ks, dilation in zip(kernel_size, dilations)
        ])

        # Additional Max & 1x1 branch
        self.branches.append(nn.Sequential(
            # unit: Conv2d, BN, ReLU, bias;
            conv_unit(in_channels, branch_channels, kernel_size=1,
                      bias=False, bn=True, relu=True),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.BatchNorm2d(branch_channels)
        ))

        # Residual Conv2d, bias, BN; ReLU
        self.branches.append(
            conv_unit(in_channels, branch_channels, kernel_size=(1, 1), padding=0, stride=(stride, 1),
                      bias=True, bn=True, relu=False),
        )

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            # res: Conv2d, bias; No BN, ReLU
            self.residual = wrapped_conv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)

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


class PGCN_Encoder_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True, kernel_size=5,
                 dilations=[1, 2, 3, 5, 7, 9]):
        super(PGCN_Encoder_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A, adaptive=adaptive)
        self.tcn1 = MultiScale_TemporalConv(out_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                            dilations=dilations, residual=False)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            # res: bias, Conv2d, BN and ReLU
            self.residual = wrapped_conv(in_channels, out_channels, kernel_size=(1, 1), stride=(stride, 1))

    def forward(self, x):
        x = self.tcn1(self.gcn1(x)) + self.residual(x)
        return self.relu(x)


class LinearBottleneck(nn.Module):
    """LinearBottleneck used in MobileNetV2"""

    def __init__(self, in_channels, out_channels, t=2, stride=2, **kwargs):
        super(LinearBottleneck, self).__init__()
        self.use_shortcut = stride == 1 and in_channels == out_channels
        inter_channel = int(in_channels * t)
        self.block = nn.Sequential(
            # pw: Conv2d, BN and ReLU, bias
            conv_unit(in_channels, inter_channel, kernel_size=(1, 1),
                      bias=False, bn=True, relu=True),
            # dw: Conv2d, BN and ReLU, bias,
            conv_unit(inter_channel, inter_channel, kernel_size=(1, 1), stride=(1, 1),
                      bias=False, bn=True, relu=True),
            # pw-linear: Conv2d, BN, bias, ReLU
            conv_unit(inter_channel, out_channels, kernel_size=(1, 1),
                      bias=True, bn=True, relu=False)
        )

    def forward(self, x):
        out = self.block(x)
        if self.use_shortcut:
            out = x + out
        return out


class PyramidPooling(nn.Module):
    """Pyramid pooling module"""

    def __init__(self, in_channels, out_channels, context_channle=None, num_point=26, t_blocks=[1, 2, 3, 5], **kwargs):
        super(PyramidPooling, self).__init__()
        inter_channels = in_channels // 4
        # conv1 - out: Conv2d, BN, ReLU, bias
        self.conv1 = conv_unit(in_channels, inter_channels, (1, 1),
                               bias=False, bn=True, relu=True, **kwargs)
        self.conv2 = conv_unit(in_channels, inter_channels, (1, 1),
                               bias=False, bn=True, relu=True, **kwargs)
        self.conv3 = conv_unit(in_channels, inter_channels, (1, 1),
                               bias=False, bn=True, relu=True, **kwargs)
        self.conv4 = conv_unit(in_channels, inter_channels, (1, 1),
                               bias=False, bn=True, relu=True, **kwargs)
        if context_channle is not None:
            self.conv_c1 = conv_unit(context_channle, inter_channels, (1, 1),
                                     bias=False, bn=True, relu=True, **kwargs)
            self.conv_c2 = conv_unit(context_channle, inter_channels, (1, 1),
                                     bias=False, bn=True, relu=True, **kwargs)
            self.conv_c3 = conv_unit(context_channle, inter_channels, (1, 1),
                                     bias=False, bn=True, relu=True, **kwargs)
            self.conv_c4 = conv_unit(context_channle, inter_channels, (1, 1),
                                     bias=False, bn=True, relu=True, **kwargs)
            self.out2 = conv_unit(in_channels + inter_channels * 8 + context_channle, out_channels, (1, 1),
                                  bias=False, bn=True, relu=True)
        self.out1 = conv_unit(in_channels * 2, out_channels, (1, 1),
                              bias=False, bn=True, relu=True)

        self.pool1 = nn.AdaptiveAvgPool2d((t_blocks[0], num_point))
        self.pool2 = nn.AdaptiveAvgPool2d((t_blocks[1], num_point))
        self.pool3 = nn.AdaptiveAvgPool2d((t_blocks[2], num_point))
        self.pool4 = nn.AdaptiveAvgPool2d((t_blocks[3], num_point))

        self.sigmoid = nn.Sigmoid()
        # self.att = nn.Conv2d(inter_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def upsample(self, x, size):
        return F.interpolate(x, size, mode='bilinear', align_corners=True)

    # def attention(self, x):
    #     B, C, T, V = x.shape
    #     x1 = x.mean(-1)
    #     score = self.sigmoid(x1.unsqueeze(-1) - x1.unsqueeze(-2))
    #     att = self.att(score)
    #     y = torch.einsum('ncij,nciv->ncjv', att, x)
    #     return y

    def forward(self, x, context=None, size=None):
        # size = x.size()[2:]
        _, _, t, v = x.size()
        if size is None:
            size = (t, v)

        feat1 = self.upsample(self.conv1(self.pool1(x)), size)
        feat2 = self.upsample(self.conv2(self.pool2(x)), size)
        feat3 = self.upsample(self.conv3(self.pool3(x)), size)
        feat4 = self.upsample(self.conv4(self.pool4(x)), size)
        if size[0] != t:
            x = self.upsample(x, size)

        if context is not None:
            feat_c1 = self.upsample(self.conv_c1(self.pool1(context)), (t, v))
            feat_c2 = self.upsample(self.conv_c2(self.pool1(context)), (t, v))
            feat_c3 = self.upsample(self.conv_c3(self.pool1(context)), (t, v))
            feat_c4 = self.upsample(self.conv_c4(self.pool1(context)), (t, v))
            context = self.upsample(context, (t, v))
            x = torch.cat([x, feat1, feat2, feat3, feat4, feat_c1, feat_c2, feat_c3, feat_c4, context], dim=1)
            x = self.out2(x)
        else:
            x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
            x = self.out1(x)
        return x


class TemporalAlignment(nn.Module):
    """align temporal feature module"""

    def __init__(self, in_channels=256, block_channels=(256, 384, 512), context_channel=None,
                 out_channels=512, t=6, num_blocks=(3, 3, 3), num_point=26, residual=True, **kwargs):
        super(TemporalAlignment, self).__init__()
        self.residual = residual
        self.context_channel = context_channel
        self.bottleneck1 = self._make_layer(LinearBottleneck, in_channels, block_channels[0], num_blocks[0], t, 1)
        self.bottleneck2 = self._make_layer(LinearBottleneck, block_channels[0], block_channels[1], num_blocks[1], t, 1)
        self.bottleneck3 = self._make_layer(LinearBottleneck, block_channels[1], block_channels[2], num_blocks[2], t, 1)
        if context_channel is not None:
            self.bottleneck_c1 = self._make_layer(LinearBottleneck, context_channel, block_channels[0], num_blocks[0], t, 1)
            self.bottleneck_c2 = self._make_layer(LinearBottleneck, block_channels[0], block_channels[1], num_blocks[1], t, 1)
            self.bottleneck_c3 = self._make_layer(LinearBottleneck, block_channels[1], block_channels[2], num_blocks[2], t, 1)
        self.ppm = PyramidPooling(block_channels[2], out_channels, num_point=num_point)
        if self.residual:
            # res: Conv2d, bias; No BN, ReLU
            self.res = wrapped_conv(in_channels, out_channels, (1, 1))

    def _make_layer(self, block, inplanes, planes, blocks, t=6, stride=1):
        layers = []
        layers.append(block(inplanes, planes, t, stride))
        for i in range(1, blocks):
            layers.append(block(planes, planes, t, 1))
        return nn.Sequential(*layers)

    def forward(self, x, context=None, size=None):
        x1 = self.bottleneck1(x)
        x1 = self.bottleneck2(x1)
        x1 = self.bottleneck3(x1)
        if context is not None and self.context_channel is not None:
            c1 = self.bottleneck_c1(context)
            c1 = self.bottleneck_c2(c1)
            c1 = self.bottleneck_c3(c1)
            y = self.ppm(x1, c1, size=size)
        else:
            y = self.ppm(x1, size=size)

        if self.residual:
            res = self.res(x)
            if res.shape[-2] < y.shape[-2]:
                res = F.interpolate(res, (y.shape[-2], y.shape[-1]), mode='bilinear', align_corners=True)
            y = y + res
        return y


class GlobalFeatureExtractor(nn.Module):
    """Global feature extractor module"""

    def __init__(self, in_channels=256, block_channels=(256, 384, 512), context_channel=512,
                 out_channels=512, t=6, num_blocks=(3, 3, 3), num_point=26, residual=True, **kwargs):
        super(GlobalFeatureExtractor, self).__init__()
        self.residual = residual
        self.context_channel = context_channel
        self.bottleneck1 = self._make_layer(LinearBottleneck, in_channels, block_channels[0], num_blocks[0], t, 1)
        self.bottleneck2 = self._make_layer(LinearBottleneck, block_channels[0], block_channels[1], num_blocks[1], t, 1)
        self.bottleneck3 = self._make_layer(LinearBottleneck, block_channels[1], block_channels[2], num_blocks[2], t, 1)
        if context_channel is not None:
            self.bottleneck_c1 = self._make_layer(LinearBottleneck, context_channel, block_channels[0], num_blocks[0], t, 1)
            self.bottleneck_c2 = self._make_layer(LinearBottleneck, block_channels[0], block_channels[1], num_blocks[1], t, 1)
            self.bottleneck_c3 = self._make_layer(LinearBottleneck, block_channels[1], block_channels[2], num_blocks[2], t, 1)
        self.ppm = PyramidPooling(block_channels[2], out_channels, num_point=num_point)
        if self.residual:
            # res: Conv2d, bias; No BN, ReLU
            self.res = wrapped_conv(in_channels, out_channels, (1, 1))

    def _make_layer(self, block, inplanes, planes, blocks, t=6, stride=1):
        layers = []
        layers.append(block(inplanes, planes, t, stride))
        for i in range(1, blocks):
            layers.append(block(planes, planes, t, 1))
        return nn.Sequential(*layers)

    def forward(self, x, context=None, size=None):
        x1 = self.bottleneck1(x)
        x1 = self.bottleneck2(x1)
        x1 = self.bottleneck3(x1)
        if context is not None and self.context_channel is not None:
            c1 = self.bottleneck_c1(context)
            c1 = self.bottleneck_c2(c1)
            c1 = self.bottleneck_c3(c1)
            y = self.ppm(x1, c1, size=size)
        else:
            y = self.ppm(x1, size=size)

        if self.residual:
            res = self.res(x)
            if res.shape[-2] < y.shape[-2]:
                res = F.interpolate(res, (y.shape[-2], y.shape[-1]), mode='bilinear', align_corners=True)
            y = y + res
        return y

    def forward2(self, x, context=None):
        if context is not None:
            x1 = self.ppm(x, context)
        else:
            x1 = x
        y = self.bottleneck1(x1)
        y = self.bottleneck2(y)
        y = self.bottleneck3(y)

        if self.residual:
            y = y + self.res(x)
        return y


class FeatureFusionModule(nn.Module):
    """Feature fusion module"""

    def __init__(self, higher_in_channels=256, lower_in_channels=512, out_channels=512, scale_factor=4,
                 **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.scale_factor = scale_factor
        # dwconv: Conv2D BN, relu;
        self.dwconv = conv_unit(lower_in_channels, out_channels, ks=1,
                                bias=False, bn=True, relu=True)

        # lower: Conv2d, bias, BN;
        self.conv_lower_res = conv_unit(out_channels, out_channels,
                                        kernel_size=(1, 1),
                                        bias=True, bn=True, relu=False)
        # residual: Conv2d, bias, BN;
        self.conv_higher_res = conv_unit(higher_in_channels, out_channels,
                                         kernel_size=(1, 1),
                                         bias=True, bn=True, relu=False)
        self.relu = nn.ReLU(True)

    def forward(self, higher_res_feature, lower_res_feature, size=(1000, 25)):
        # c3: higher, x1: lower
        lower_res_feature = F.interpolate(lower_res_feature, size=size, mode='bilinear', align_corners=True)
        lower_res_feature = self.dwconv(lower_res_feature)
        lower_res_feature = self.conv_lower_res(lower_res_feature)

        higher_res_feature = self.conv_higher_res(higher_res_feature)
        # print(lower_res_feature.shape[2])
        higher_res_feature = F.interpolate(higher_res_feature, size=size, mode='bilinear', align_corners=True)
        # if higher_res_feature.shape[2] < lower_res_feature.shape[2]:
        #     higher_res_feature = F.interpolate(higher_res_feature, (lower_res_feature.shape[2], self.num_point),
        #                                        mode='bilinear', align_corners=True)
        # else:
        #     lower_res_feature = F.interpolate(lower_res_feature, (higher_res_feature.shape[2], self.num_point),
        #                                       mode='bilinear', align_corners=True)
        out = higher_res_feature + lower_res_feature
        return self.relu(out)


class _DSConv(nn.Module):
    """Depthwise Separable Convolutions"""

    def __init__(self, dw_channels, out_channels, kernel_size=(3, 1), stride=(1, 1), relu=True, coeff=None,
                 residual=True, padding=(1, 0), **kwargs):
        super(_DSConv, self).__init__()
        # conv: 2 unit: Conv2d, BN and ReLU, bias
        self.conv = nn.Sequential(
            conv_unit(dw_channels, dw_channels, kernel_size=kernel_size, stride=stride, groups=dw_channels,
                      pad=padding, bias=False, bn=True, relu=True, coeff=coeff),
            conv_unit(dw_channels, out_channels, kernel_size=(1, 1),
                      bias=False, bn=True, relu=relu)
        )

    def forward(self, x):
        out = self.conv(x)
        return out


class Classifer(nn.Module):
    """Classifer"""

    def __init__(self, in_channels, num_classes, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), **kwargs):
        super(Classifer, self).__init__()
        self.dsconv1 = _DSConv(in_channels, in_channels, kernel_size, stride, padding, coeff=3.0)
        self.dsconv2 = _DSConv(in_channels, in_channels, relu=True)
        # self.residual = wrapped_conv(in_channels, in_channels, kernel_size=(1, 1))
        self.out = nn.Sequential(
            wrapped_conv(in_channels, num_classes, kernel_size=(1, 1), bias=False)
        )
        # self.relu = nn.ReLU(True)
        self.feature = None

    def forward(self, x):
        x1 = self.dsconv1(x)
        x2 = self.dsconv2(x1)
        # res = self.residual(x.mean(-1, keepdim=True))
        # x3 = x2 + res
        self.feature = x2.clone().detach()
        y = self.out(x2)
        return y

class TaskClassifer(nn.Module):
    """Classifer"""
    def __init__(self, in_channels, num_classes, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), **kwargs):
        super(TaskClassifer, self).__init__()
        inter_channels = in_channels // 8
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(in_channels, inter_channels, (1, 1), stride=(1, 1), padding=(0, 0))
        self.conv2 = nn.Conv2d(in_channels, inter_channels, (1, 1), stride=(1, 1), padding=(0, 0))
        self.dsconv1 = _DSConv(inter_channels, inter_channels*2, (3, 1), stride=(3, 1), padding=(1, 0))
        self.max_pool1 = nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.dsconv2 = nn.Sequential(
            _DSConv(inter_channels*2, inter_channels*4, (3, 1), stride=(2, 1), padding=(1, 0)),
            _DSConv(inter_channels*4, inter_channels*8, (3, 1), stride=(1, 1), padding=(1, 0))
        )
        self.max_pool2 = nn.AdaptiveMaxPool2d((num_classes, 1))
        # self.residual = wrapped_conv(in_channels, in_channels, kernel_size=(1, 1))
        self.out = nn.Sequential(
            wrapped_conv(inter_channels*8, num_classes, kernel_size=(1, 1), bias=False)
        )
        # self.task_map = nn.Sequential(
        #     conv_unit(in_channels, num_classes*num_classes, kernel_size=(1, 1), bias=False),
        #     conv_unit(num_classes*num_classes, num_classes*num_classes, kernel_size=(1, 1), bias=False),
        # )

        self.relu = nn.ReLU(True)
        self.tanh = nn.Tanh()
        self.max_frames = 240

    def forward(self, x):
        N, C, T, V = x.shape
        x1 = self.conv1(x[:, :, 1:, :])
        x2 = self.conv2(x[:, :, :-1, :])
        if T > self.max_frames:
            x1 = F.adaptive_max_pool2d(x1, (self.max_frames, V))
            x2 = F.adaptive_max_pool2d(x2, (self.max_frames, V))
        changes = self.relu(x1 - x2)

        x1 = self.dsconv1(changes)
        x1 = self.max_pool1(x1)
        x2 = self.dsconv2(x1)
        x2 = self.max_pool2(x2)
        y = self.relu(self.out(x2)).squeeze(-1)
        # distance = y.unsqueeze(2) - y.unsqueeze(1)
        # pred_graph = distance.sum(-1)
        return y


class Encoder(nn.Module):
    def __init__(self, num_point, num_person=1, graph=None, graph_args=dict(), in_channels=3, out_channels=256,
                 multi_out=False, drop_out=0):
        super(Encoder, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)
            A = self.graph.A
        self.multi_out = multi_out
        self.num_point = num_point
        # self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        self.l1 = PGCN_Encoder_unit(in_channels, 64, A, residual=False)
        self.l2 = PGCN_Encoder_unit(64, 64, A)
        self.l3 = PGCN_Encoder_unit(64, 64, A)
        self.l4 = PGCN_Encoder_unit(64, 64, A)
        self.l5 = PGCN_Encoder_unit(64, 128, A, stride=2)
        self.l6 = PGCN_Encoder_unit(128, 128, A)
        self.l7 = PGCN_Encoder_unit(128, 128, A)
        self.l8 = PGCN_Encoder_unit(128, 256, A, stride=2)
        self.l9 = PGCN_Encoder_unit(256, 256, A)
        self.l10 = PGCN_Encoder_unit(256, out_channels, A)

    def forward(self, x):

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
        if self.multi_out:
            return c1, c2, c3
        else:
            return c3


class Decoder(nn.Module):
    def __init__(self, num_point, in_channels, context_channels, blocks: list, out_channels, num_blocks, drop_out=0):
        super(Decoder, self).__init__()
        self.global_feature_extractor = GlobalFeatureExtractor(in_channels, blocks, context_channels, out_channels,
                                                               2, num_blocks,
                                                               num_point=num_point, residual=True)
        self.feature_fusion = FeatureFusionModule(in_channels, out_channels, 512)
        self.feature = None

    def forward(self, c3, size: tuple, context=None):
        # c3: Bs x 256 x t x V
        # x1: BS x 512 x t x V
        x1 = self.global_feature_extractor(c3, context)
        # x: BS x 512 x 30 x 26
        x2 = self.feature_fusion(c3, x1, size=size)
        return x2


class CrossGroupAttention(nn.Module):
    def __init__(self, in_channels, num_groups, reduction_ratio=4):
        """
        Args:
            in_channels (int): Total number of input channels (C).
            num_groups (int): Number of groups (g).
            reduction_ratio (int): Reduction ratio for attention computation.
        """
        super(CrossGroupAttention, self).__init__()
        assert in_channels % num_groups == 0, "in_channels must be divisible by num_groups"

        self.num_groups = num_groups
        self.group_channels = in_channels // num_groups
        reduced_channels = max(1, self.group_channels // reduction_ratio)

        # Query, Key, Value projections for cross-group attention
        self.q_proj = nn.Linear(self.group_channels, reduced_channels, bias=False)
        self.k_proj = nn.Linear(self.group_channels, reduced_channels, bias=False)
        self.v_proj = nn.Linear(self.group_channels, self.group_channels, bias=False)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Args:
            x: Input tensor with shape (B, C, T, V)
        Returns:
            Output tensor with the same shape as input.
        """
        B, C, T, V = x.shape
        x = x.view(B, self.num_groups, self.group_channels, T, V)  # Split into groups

        # Global pooling to get compact representations per group
        x_pooled = x.mean(dim=(3, 4))  # Shape: (B, g, group_channels)

        # Compute query, key, and value for attention
        Q = self.q_proj(x_pooled)  # Shape: (B, g, reduced_channels)
        K = self.k_proj(x_pooled)  # Shape: (B, g, reduced_channels)
        V = self.v_proj(x_pooled)  # Shape: (B, g, group_channels)

        # Compute attention scores (similarity between groups)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1))  # Shape: (B, g, g)
        attention_weights = self.softmax(attention_scores)  # Normalize across groups

        # Compute weighted sum of value vectors
        attention_output = torch.matmul(attention_weights, V)  # Shape: (B, g, group_channels)

        # Expand the attention weights back to (B, g, group_channels, T, V)
        attention_output = attention_output.unsqueeze(-1).unsqueeze(-1)  # (B, g, group_channels, 1, 1)
        x = x * attention_output  # Apply cross-group attention

        # Merge groups back
        x = x.view(B, C, T, V)

        return x

class Decoder_Att(nn.Module):
    def __init__(self, num_point, in_channels, context_channels, blocks: list, out_channels, num_blocks, drop_out=0):
        super(Decoder_Att, self).__init__()
        self.att = CrossGroupAttention(in_channels, 8, reduction_ratio=4)
        self.global_feature_extractor = GlobalFeatureExtractor(in_channels, blocks, context_channels, out_channels,
                                                               2, num_blocks,
                                                               num_point=num_point, residual=True)
        self.feature_fusion = FeatureFusionModule(in_channels, out_channels, 512)
        self.feature = None

    def forward(self, c3, size: tuple, context=None):
        # c3: Bs x 256 x t x V
        # x1: BS x 512 x t x V
        x1 = self.global_feature_extractor(c3, context)
        # x: BS x 512 x 30 x 26
        x2 = self.feature_fusion(c3, x1, size=size)
        return x2


class Model(nn.Module):
    def __init__(self, num_class, num_point, num_person=1, graph=None, graph_args=dict(),
                 in_channels=3, en_out_channels=256, drop_out=0, timesteps=1500, sampling_timesteps=25):
        super(Model, self).__init__()

        self.encoder = Encoder(in_channels=in_channels, out_channels=en_out_channels, num_point=num_point,
                               num_person=num_person, graph=graph, graph_args=graph_args)
        self.decoder = Decoder(num_point=num_point, in_channels=en_out_channels, context_channels=None,
                               blocks=[256, 384, 512],
                               out_channels=512, num_blocks=[3, 3, 3])
        self.classifier = Classifer(512, num_class, (3, num_point), padding=(1, 0), stride=(1, 1))
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        self.feature = None

    def forward(self, data):
        x = data['motion']
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x.to(dtype=self.data_bn.weight.dtype))
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        # c3: BS x 256 x 30 x 26
        c3 = self.encoder(x)

        # y: BS x 512 x T x V
        y = self.decoder(c3, (T, V))

        # y: BS x 14 x 120 x 1
        y = self.classifier(y)

        # feature: BS x 512 x 120
        # self.feature = self.classifier.feature.clone().detach()

        new_c = y.size(1)
        y = y.view(N, new_c, -1)
        out = {
            "pred_motion": y
        }
        return out