"""
https://github.com/earhian/SIIM-ACR-Pneumothorax-Segmentation-5th/blob/master/models/unet.py
https://github.com/bestfitting/kaggle/blob/master/siim_acr/src/networks/unet_resnet.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import Decoder, ConvBnRelu2d, MaskScoringHead, ASPP
from ..backbones import resnet34, resnet50, se_resnext50_32x4d, EfficientNetEncoder, se_resnext101_32x4d


def select_basemodel(model_name, pretrained=True):
    assumed_model_name = {
        'resnet34', 'resnet50', 'seresnext50',
        'efficientnet-b3', 'efficientnet-b4', 'efficientnet-b5', 'seresnext101'
    }
    assert model_name in assumed_model_name, f'Invalid model name: {model_name}, \n  assumed: {assumed_model_name}'

    if model_name == 'resnet34':
        basemodel = resnet34(pretrained=pretrained)
        planes = [256 // 4, 512 // 4, 1024 // 4, 2048 // 4]
    elif model_name == 'resnet50':
        basemodel = resnet50(pretrained=pretrained)
        planes = [256, 512, 1024, 2048]
    elif model_name == 'seresnext50':
        basemodel = se_resnext50_32x4d(pretrained='imagenet' if pretrained else None)  # TODO modify
        planes = [256, 512, 1024, 2048]

        # TODO down channel if memory is very big
        inplanes = 64
        basemodel.layer0 = nn.Sequential(
            ConvBnRelu2d(3, inplanes, stride=2, padding=1),
            ConvBnRelu2d(inplanes, inplanes, stride=2, padding=1),
        )
    elif model_name == 'seresnext101':
        basemodel = se_resnext101_32x4d(pretrained='imagenet' if pretrained else None)
        # planes = [256 // 4, 512 // 4, 1024 // 4, 2048 // 4]
        planes = [256, 512, 1024, 2048]

        # TODO down channel if memory is very big
        inplanes = 64
        basemodel.layer0 = nn.Sequential(
            ConvBnRelu2d(3, inplanes, stride=2, padding=1),
            ConvBnRelu2d(inplanes, inplanes, stride=2, padding=1),
        )
    elif model_name in {'efficientnet-b3', 'efficientnet-b4', 'efficientnet-b5'}:
        basemodel = EfficientNetEncoder(model_name)
        planes = basemodel.planes
    else:
        raise NotImplementedError
    return basemodel, planes


class UNet(nn.Module):
    """Basic UNet with hyper columns"""
    def __init__(self, model_name, out_channel, base_ch=32,
                 att_type='cbam', reduction=16, pretrained=True, use_aspp=False):
        super(UNet, self).__init__()
        assert att_type is None or att_type in {'cbam'}
        self.basemodel, self.planes = select_basemodel(model_name, pretrained=pretrained)
        self.use_aspp = use_aspp

        self.center = nn.Sequential(
            ConvBnRelu2d(self.planes[3], self.planes[3], kernel_size=3, padding=1),
            ConvBnRelu2d(self.planes[3], self.planes[2], kernel_size=3, padding=1),
        )

        self.base_ch = base_ch
        kwargs_decoder = {
            'attention_type': att_type, 'attention_kernel_size': 1,
            'reduction': reduction, 'out_channels': base_ch
        }
        self.decoder5 = Decoder(self.planes[3] + self.planes[2], 512, **kwargs_decoder)
        self.decoder4 = Decoder(self.planes[2] + base_ch, 256, **kwargs_decoder)
        self.decoder3 = Decoder(self.planes[1] + base_ch, 128, **kwargs_decoder)
        self.decoder2 = Decoder(self.planes[0] + base_ch, 64, **kwargs_decoder)
        self.decoder1 = Decoder(base_ch, 32, **kwargs_decoder)

        if use_aspp:
            self.aspp = ASPP(inplanes=base_ch*5, mid_c=base_ch*2, dilations=[1, 6, 12, 18])
            self.final = nn.Sequential(
                nn.Conv2d(base_ch*2, out_channel, kernel_size=1, padding=0),
            )
        else:
            self.final = nn.Sequential(
                ConvBnRelu2d(base_ch*5, base_ch*2, kernel_size=3, padding=1),
                ConvBnRelu2d(base_ch*2, base_ch, kernel_size=3, padding=1),
                nn.Conv2d(base_ch, out_channel, kernel_size=1, padding=0),
            )

    def forward(self, x):
        e2, e3, e4, e5 = self.basemodel(x)  # 1/4, 1/8, 1/16, 1/32

        c = self.center(e5)

        d5 = self.decoder5(torch.cat([c,  e5], 1))
        d4 = self.decoder4(torch.cat([d5, e4], 1))
        d3 = self.decoder3(torch.cat([d4, e3], 1))
        d2 = self.decoder2(torch.cat([d3, e2], 1))
        d1 = self.decoder1(d2)

        f = torch.cat((
            d1,
            F.interpolate(d2, scale_factor=2,  mode='bilinear', align_corners=False),
            F.interpolate(d3, scale_factor=4,  mode='bilinear', align_corners=False),
            F.interpolate(d4, scale_factor=8,  mode='bilinear', align_corners=False),
            F.interpolate(d5, scale_factor=16, mode='bilinear', align_corners=False),
        ), 1)

        if self.use_aspp:
            f = self.aspp(f)

        logit = self.final(f)
        return logit


class MSUNet(UNet):
    """UNet with Mask Scoring Head"""
    def __init__(self, model_name, out_channel, base_ch=32,
                 att_type='cbam', reduction=16, pretrained=True, use_aspp=False):
        super(MSUNet, self).__init__(
            model_name=model_name, out_channel=out_channel, base_ch=base_ch,
            att_type=att_type, reduction=reduction, pretrained=pretrained, use_aspp=use_aspp
        )

        self.ms_head = MaskScoringHead(
            in_feature_ch=self.base_ch * 2 if use_aspp else self.base_ch*5,
            num_classes=out_channel,
            mask_feature_size=(16, 32),
        )

    def forward(self, x):
        e2, e3, e4, e5 = self.basemodel(x)  # 1/4, 1/8, 1/16, 1/32

        c = self.center(e5)

        d5 = self.decoder5(torch.cat([c,  e5], 1))
        d4 = self.decoder4(torch.cat([d5, e4], 1))
        d3 = self.decoder3(torch.cat([d4, e3], 1))
        d2 = self.decoder2(torch.cat([d3, e2], 1))
        d1 = self.decoder1(d2)

        f = torch.cat((
            d1,
            F.interpolate(d2, scale_factor=2,  mode='bilinear', align_corners=False),
            F.interpolate(d3, scale_factor=4,  mode='bilinear', align_corners=False),
            F.interpolate(d4, scale_factor=8,  mode='bilinear', align_corners=False),
            F.interpolate(d5, scale_factor=16, mode='bilinear', align_corners=False),
        ), 1)

        if self.use_aspp:
            f = self.aspp(f)
        logit = self.final(f)
        mask_score_logit = self.ms_head(f, logit)

        return logit, mask_score_logit


class ClsHead(nn.Module):
    def __init__(self, in_ch=512, num_classes=4):
        super(ClsHead, self).__init__()
        self.in_ch = in_ch

        # TODO compare other pooling method
        # I think pyramid pool is more useful...
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(in_ch, num_classes)

    def forward(self, x):
        # BATCH x CH x H x W => BATCH x CH x 1 x 1
        x = self.pool(x)

        # BATCH x CH x 1 x 1 => BATCH x CH
        x = x.squeeze(3).squeeze(2)

        # BATCH x CH => BATCH x CLASS
        x = self.fc(x)
        return x


class ClsUNet(UNet):
    """UNet with Classification Head"""
    def __init__(self, model_name, out_channel,
                 att_type='cbam', reduction=16, pretrained=True, num_class_cls=4):
        super(ClsUNet, self).__init__(
            model_name=model_name, out_channel=out_channel,
            att_type=att_type, reduction=reduction, pretrained=pretrained
        )

        # Sometimes, num_class != out_channel
        self.num_class_cls = num_class_cls
        self.cls_head = ClsHead(in_ch=self.planes[3], num_classes=num_class_cls)

    def forward(self, x):
        e2, e3, e4, e5 = self.basemodel(x)

        c = self.center(e5)

        d5 = self.decoder5(torch.cat([c,  e5], 1))
        d4 = self.decoder4(torch.cat([d5, e4], 1))
        d3 = self.decoder3(torch.cat([d4, e3], 1))
        d2 = self.decoder2(torch.cat([d3, e2], 1))
        d1 = self.decoder1(d2)

        f = torch.cat((
            d1,
            F.interpolate(d2, scale_factor=2,  mode='bilinear', align_corners=False),
            F.interpolate(d3, scale_factor=4,  mode='bilinear', align_corners=False),
            F.interpolate(d4, scale_factor=8,  mode='bilinear', align_corners=False),
            F.interpolate(d5, scale_factor=16, mode='bilinear', align_corners=False),
        ), 1)
        logit = self.final(f)
        logit_cls = self.cls_head(e5)
        return logit, logit_cls


class MSClsUNet(UNet):
    """
    UNet with Classification Head and Mask-Scoring Head
    Not Work !!!
    """
    def __init__(self, model_name, out_channel,
                 att_type='cbam', reduction=16, pretrained=True, num_class_cls=4):
        super(MSClsUNet, self).__init__(
            model_name=model_name, out_channel=out_channel,
            att_type=att_type, reduction=reduction, pretrained=pretrained
        )

        # Sometimes, num_class != out_channel
        self.num_class_cls = num_class_cls
        self.cls_head = ClsHead(in_ch=self.planes[3], num_classes=num_class_cls)

        self.ms_head = MaskScoringHead(
            in_feature_ch=self.base_ch*5,
            num_classes=out_channel,
            mask_feature_size=(16, 32),
        )

    def forward(self, x):
        e2, e3, e4, e5 = self.basemodel(x)

        c = self.center(e5)

        d5 = self.decoder5(torch.cat([c,  e5], 1))
        d4 = self.decoder4(torch.cat([d5, e4], 1))
        d3 = self.decoder3(torch.cat([d4, e3], 1))
        d2 = self.decoder2(torch.cat([d3, e2], 1))
        d1 = self.decoder1(d2)

        f = torch.cat((
            d1,
            F.interpolate(d2, scale_factor=2,  mode='bilinear', align_corners=False),
            F.interpolate(d3, scale_factor=4,  mode='bilinear', align_corners=False),
            F.interpolate(d4, scale_factor=8,  mode='bilinear', align_corners=False),
            F.interpolate(d5, scale_factor=16, mode='bilinear', align_corners=False),
        ), 1)
        logit = self.final(f)
        logit_cls = self.cls_head(e5)
        logit_ms = self.ms_head(f, logit)
        return logit, logit_ms, logit_cls
