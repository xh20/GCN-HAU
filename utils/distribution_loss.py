import torch
import torch.nn as nn
from torch.nn.modules.loss import BCELoss


class DistributionLoss():
    def __init__(self, num_class=14, length_feature=256):
        super(DistributionLoss, self).__init__()
        self.num_class = num_class
        self.length_feature = length_feature

    def forward(self, res, tar):
        # print(res.size(), tar.size())

        return super(DistributionLoss, self).forward(res, tar)