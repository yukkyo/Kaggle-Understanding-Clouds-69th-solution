import torch
import torch.nn as nn
import torch.nn.functional as F


"""
Mask Scoring Head

CAUTION:
    This code is not fully reproduce the paper !
    This module can't predict mask score

Feature of Mask Scoring R-CNN (MS R-CNN)
- https://arxiv.org/pdf/1903.00241.pdf
- https://github.com/zjhuang22/maskscoring_rcnn
  - https://github.com/zjhuang22/maskscoring_rcnn/tree/master/maskrcnn_benchmark/modeling/roi_heads/maskiou_head
"""


class MaskMetricsPredictor(nn.Module):
    def __init__(self, num_classes, embedding_size=1024):
        super(MaskMetricsPredictor, self).__init__()
        self.embedding_size = embedding_size
        self.maskiou = nn.Linear(embedding_size, num_classes)
        nn.init.normal_(self.maskiou.weight, mean=0, std=0.01)
        nn.init.constant_(self.maskiou.bias, 0)

    def forward(self, x):
        maskiou = self.maskiou(x)
        return maskiou


class MaskScoreFeatureExtractor(nn.Module):
    def __init__(self, in_feature_ch,
                 mask_feature_size=(16, 48), embedding_size=1024):
        super(MaskScoreFeatureExtractor, self).__init__()
        self.in_feature_ch = in_feature_ch
        self.mask_feature_size = mask_feature_size
        self.embedding_size = embedding_size

        # (C, H, W) -> (C, mask_feature_height, mask_feature_width)
        self.pool = nn.AdaptiveMaxPool2d(mask_feature_size)

        base_ch = 128
        self.mask_score_fcn1 = nn.Conv2d(in_feature_ch, base_ch*4, 3, 1, 1)
        self.mask_score_fcn2 = nn.Conv2d(base_ch*4, base_ch*2, 3, 1, 1)
        self.mask_score_fcn3 = nn.Conv2d(base_ch*2, base_ch, 3, 1, 1)
        self.mask_score_fcn4 = nn.Conv2d(base_ch, base_ch, 3, 2, 1)

        if len(mask_feature_size) == 1:
            conved_size = (mask_feature_size//2)**2
        else:
            conved_size = (mask_feature_size[0] // 2) * (mask_feature_size[1] // 2)
        self.mask_score_fc1 = nn.Linear(base_ch * conved_size, embedding_size)
        self.mask_score_fc2 = nn.Linear(embedding_size, embedding_size)

        for l in [self.mask_score_fcn1, self.mask_score_fcn2,
                  self.mask_score_fcn3, self.mask_score_fcn4]:
            nn.init.kaiming_normal_(l.weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(l.bias, 0)

        for l in [self.mask_score_fc1, self.mask_score_fc2]:
            nn.init.kaiming_uniform_(l.weight, a=1)
            nn.init.constant_(l.bias, 0)

    def forward(self, x, mask):
        x = self.pool(x)
        mask_pool = self.pool(mask)
        x = torch.cat((x, mask_pool), 1)

        x = F.relu(self.mask_score_fcn1(x))
        x = F.relu(self.mask_score_fcn2(x))
        x = F.relu(self.mask_score_fcn3(x))
        x = F.relu(self.mask_score_fcn4(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.mask_score_fc1(x))
        x = F.relu(self.mask_score_fc2(x))

        return x


class MaskScoringHead(torch.nn.Module):
    def __init__(self, in_feature_ch, num_classes=4,
                 mask_feature_size=(16, 48), embedding_size=1024):
        super(MaskScoringHead, self).__init__()
        self.embedding_size = embedding_size
        self.feature_extractor = MaskScoreFeatureExtractor(
            in_feature_ch=in_feature_ch+num_classes,  # feature_ch + mask_ch
            mask_feature_size=mask_feature_size,
            embedding_size=embedding_size
        )
        self.predictor = MaskMetricsPredictor(
            num_classes=num_classes,
            embedding_size=embedding_size
        )

    def forward(self, features, masks):
        x = self.feature_extractor(features, masks)
        pred_mask_score = self.predictor(x)
        return pred_mask_score
