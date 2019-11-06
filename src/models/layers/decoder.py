"""
https://github.com/bestfitting/kaggle/blob/master/siim_acr/src/layers/layer_util.py
"""

import torch.nn as nn
import torch.nn.functional as F

from .attention import CBAM_Module
from .util_layers import ConvBn2d


class Decoder(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels,
                 up_sample=True,
                 attention_type=None,
                 attention_kernel_size=3,
                 reduction=16,
                 reslink=False):
        super(Decoder, self).__init__()
        self.conv1 = ConvBn2d(in_channels,  middle_channels, kernel_size=3, padding=1)
        self.conv2 = ConvBn2d(middle_channels, out_channels, kernel_size=3, padding=1)
        self.up_sample = up_sample
        self.attention = attention_type is not None
        self.attention_type = attention_type
        self.reslink = reslink
        if attention_type is None:
            pass
        elif attention_type.find('cbam') >= 0:
            self.channel_gate = CBAM_Module(out_channels, reduction, attention_kernel_size)
        if self.reslink:
            self.shortcut = ConvBn2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))

    def forward(self, x, size=None):
        if self.up_sample:
            if size is None:
                x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)  # False
            else:
                x = F.interpolate(x, size=size, mode='bilinear', align_corners=True)
        if self.reslink:
            shortcut = self.shortcut(x)
        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        if self.attention:
            x = self.channel_gate(x)
        if self.reslink:
            x = F.relu(x+shortcut)
        return x
