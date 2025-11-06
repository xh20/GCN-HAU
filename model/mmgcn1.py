import math

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from model.tfgcn import Encoder, Decoder, Classifer, conv_unit, TemporalAlignment, GlobalFeatureExtractor
from model.I3D import I3D_small


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


def gaussian_kernel(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    gauss /= gauss.sum()  # Normalize to ensure the kernel sums to 1
    return gauss.view(1, 1, window_size)  # Reshape to [1, 1, window_size] for conv1d


def apply_gaussian_filter(predictions, window_size, sigma=1.0):
    B, C, T = predictions.shape
    # Create Gaussian kernel
    kernel = gaussian_kernel(window_size, sigma).to(predictions.device)  # Move to the same device as predictions
    # Reshape predictions to [B * C, 1, T] for applying conv1d
    predictions_reshaped = predictions.view(B * C, 1, T)  # Combine B and C dimensions for 1D convolution
    # Apply 1D convolution with the Gaussian kernel along the temporal dimension
    smoothed = F.conv1d(predictions_reshaped, kernel, padding=window_size // 2, groups=1)  # Keep original size
    smoothed_predictions = smoothed.view(B, C, T)
    return smoothed_predictions


def box_filter_1d(input, kernel_size=5):
    B, C, T = input.shape
    # Create a 1D box filter kernel with equal weights
    weight = torch.ones((C, 1, kernel_size), dtype=input.dtype, device=input.device)
    weight = weight / kernel_size  # Normalize the kernel so the sum is 1

    # Apply the box filter using 1D convolution along the temporal axis
    # Here, we use `F.conv1d` for 1D convolution, and we apply it channel-wise (depthwise convolution)
    smoothed = F.conv1d(input, weight, padding=kernel_size // 2, groups=C)
    return smoothed


# PosE for Raw-point Embedding
class PosE_Initial(nn.Module):
    def __init__(self, in_dim, out_dim, alpha, beta):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha, self.beta = alpha, beta

    def forward(self, x):
        _device = x.device

        B, C, T, V = x.shape
        feat_dim = self.out_dim // (self.in_dim * 2)

        feat_range = torch.arange(feat_dim, device=_device).float()
        dim_embed = torch.pow(self.alpha, feat_range / feat_dim)
        div_embed = torch.div(self.beta * x.unsqueeze(-1), dim_embed)

        sin_embed = torch.sin(div_embed)
        cos_embed = torch.cos(div_embed)
        position_embed = torch.stack([sin_embed, cos_embed], dim=-1).flatten(-2)
        position_embed = position_embed.permute(0, 1, 4, 2, 3).reshape(B, self.out_dim, T, V)

        return position_embed

class Model(nn.Module):
    def __init__(self, num_class, num_point, num_person=1, graph=None, i3d_weights=None,
                 graph_args=dict(), in_channels=3, en_out_channels=256, img_out_channels=512, pose_out_channels=384,
                 drop_out=0, timesteps=1500, sampling_timesteps=25):
        super(Model, self).__init__()
        # self.image_encoder = InceptionI3d(num_classes=157, final_endpoint="Mixed_5c")  # num_class: rgb_charades.pt is 157, rgb_imagenet.pt is 400
        # self.image_encoder.load_state_dict(torch.load(i3d_weights))
        self.num_point = num_point
        self.image_encoder = I3D_small(in_channels=3, out_channels=img_out_channels)
        # self.image_encoder = I3D_small(in_channels=3, out_channels=in_channels)
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        # self.connector = nn.Conv2d(1024, en_out_channels, (1, 192))
        self.relu = nn.ReLU(inplace=True)
        self.pose_encoder = PosE_Initial(in_channels, out_dim=pose_out_channels, alpha=1000, beta=100)
        self.motion_encoder = Encoder(in_channels=in_channels, out_channels=en_out_channels, num_point=num_point,
                                      num_person=num_person, graph=graph, graph_args=graph_args)
        self.image_extractor = GlobalFeatureExtractor(in_channels=img_out_channels+pose_out_channels, out_channels=512, t=0.5,
                                                      context_channel=None, block_channels=[256, 384, 512],
                                                      num_point=1, residual=True)
        self.decoder1 = Decoder(num_point=num_point, in_channels=en_out_channels, context_channels=None,
                                blocks=[256, 384, 512], out_channels=512, num_blocks=[3, 3, 3])
        # self.decoder2 = Decoder(num_point=1, in_channels=512, blocks=[256, 384, 512],
        #                        out_channels=512, num_blocks=[3, 3, 3])
        # self.defussion = Diffusion(256, 256, 64, timesteps, sampling_timesteps)
        self.classifier = Classifer(512*2, num_class, (3, 1), padding=(1, 0), stride=(1, 1))
        # self.task_head = conv_unit(1, 1, kernel_size=1)
        self.feature = None
        self.tanh = nn.Tanh()

    def pred_task(self, x):
        # x: B x C x T
        distance = x.unsqueeze(2) - x.unsqueeze(1)
        distance_matrix = torch.norm(distance, dim=-1)  # Shape: (B, C, C)
        out = self.task_head(distance_matrix.unsqueeze(1))

        return out

    def forward(self, data, noise_loss=None):
        # motion stream
        x = data["motion"]
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x.to(dtype=self.data_bn.weight.dtype))
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        # image/flow stream
        image = data["image"]
        y1 = self.image_encoder(image)
        N, Ci, Ti, H, W = y1.shape
        y1 = y1.view(N, Ci, Ti, H*W)
        y_p = self.pose_encoder(x)
        y_p = F.adaptive_avg_pool2d(y_p, (Ti, H*W))
        y1 = torch.cat((y1, y_p), dim=1)
        # y1 = F.adaptive_avg_pool2d(y1, (T, 1))
        # y1 = self.image_extractor(y1, context=x, size=(T, 1))
        y1 = self.image_extractor(y1, size=(T, 1))
        N, C1, T1, V1 = y1.shape
        self.layer_norm = nn.LayerNorm([C1, T1, V1]).to(y1.device)
        y1 = self.layer_norm(y1)

        y2 = self.motion_encoder(x)
        y2 = self.decoder1(y2, (T, 1))

        y3 = torch.cat((y2, y1), dim=1)

        # y3 = torch.concat((y1, y2), dim=-1)
        # y: BS x 14 x T x 1
        y = self.classifier(y3)
        # pre_task = self.task_head(y3)
        # feature: BS x 512 x 120
        # self.feature = self.classifier.feature.clone().detach()

        new_c = y.size(1)
        y = y.view(N, new_c, -1)
        # y = self.tanh(y)
        y = apply_gaussian_filter(y, 3)
        out = {
            "pred_motion": y,
            # "pred_task": pre_task,
        }
        if noise_loss is None:
            return out
        else:
            return out, noise_loss



if __name__ == "__main__":
    num_node = 26
    graph_args = {"labeling_mode": 'spatial', "num_node": num_node}

    model = Model(num_class=15, num_point=num_node, graph="graph.bimacs_partbdy.Graph", graph_args=graph_args)

    data = {"motion": torch.randn(4, 3, 240, num_node, 1),
            "image": torch.randn(4, 3, 8, 480, 480),
            "masked_image": torch.randn(4, 3, 8, 480, 480)
            }
    output = model.forward(data)
