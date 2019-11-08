import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    """
    Reference:
        https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65938
    """
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        loss_bce = self.bce_loss(inputs, targets)
        pt = torch.exp(-loss_bce)
        loss_f = self.alpha * (torch.tensor(1.) - pt)**self.gamma * loss_bce
        return loss_f.mean()


def dice(preds, targets):
    smooth = 1e-7
    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)

    intersection = (preds_flat * targets_flat).sum()  # .float()
    union = (preds_flat.sum() + targets_flat.sum())  # .float()

    dice_score = (2.0 * intersection + smooth) / (union + smooth)
    return dice_score


class SoftmaxBCEWithLogitsLoss(nn.Module):
    def __init__(self):
        super(SoftmaxBCEWithLogitsLoss, self).__init__()
        self.bce_loss = nn.BCELoss()

    def forward(self, logits, targets):
        preds = torch.softmax(logits, dim=1)
        return self.bce_loss(preds, targets)


class MyBCEWithLogitsLoss(nn.Module):
    def __init__(self, softmax=False):
        super(MyBCEWithLogitsLoss, self).__init__()
        self.softmax = softmax
        if softmax:
            self.bce_loss = SoftmaxBCEWithLogitsLoss()
        else:
            self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        return self.bce_loss(logits, targets)


class DiceLoss(nn.Module):
    def __init__(self, softmax=False):
        super(DiceLoss, self).__init__()
        self.softmax = softmax

    def forward(self, logits, targets):
        if self.softmax:
            # softmax channel-wise
            preds = torch.softmax(logits, dim=1)
        else:
            preds = torch.sigmoid(logits)
        return 1. - dice(preds, targets)


class BCEDiceLoss(nn.Module):
    """Loss defined as alpha * BCELoss - (1 - alpha) * DiceLoss"""
    def __init__(self, alpha=0.5, softmax=False):
        super(BCEDiceLoss, self).__init__()
        self.bce_loss = MyBCEWithLogitsLoss(softmax=softmax)
        self.dice_loss = DiceLoss()
        self.alpha = alpha
        self.softmax = softmax

    def forward(self, logits, targets):
        bce_loss = self.bce_loss(logits, targets)
        dice_loss = self.dice_loss(logits, targets)
        loss = self.alpha * bce_loss + (1. - self.alpha) * dice_loss
        return loss


class BCEFocalLoss(nn.Module):
    def __init__(self, alpha=0.5, softmax=False):
        super(BCEFocalLoss, self).__init__()
        self.bce_loss = MyBCEWithLogitsLoss(softmax=softmax)
        self.focal_loss = FocalLoss()
        self.alpha = alpha
        self.softmax = softmax

    def forward(self, logits, targets):
        bce_loss = self.bce_loss(logits, targets)
        focal_loss = self.focal_loss(logits, targets)
        loss = self.alpha * bce_loss + (1. - self.alpha) * focal_loss
        return loss


class BCEDiceFocalLoss(nn.Module):
    def __init__(self, r_d=2., r_b=1., r_f=2., softmax=False):
        super(BCEDiceFocalLoss, self).__init__()
        self.r_d = r_d
        self.r_b = r_b
        self.r_f = r_f
        self.softmax = softmax
        self.bce_loss = MyBCEWithLogitsLoss(softmax=softmax)
        self.focal_loss = FocalLoss()
        self.dice_loss = DiceLoss()

    def forward(self, logits, targets):
        loss_d = self.r_d * self.dice_loss(logits, targets)
        loss_b = self.r_b * self.bce_loss(logits, targets)
        loss_f = self.r_f * self.focal_loss(logits, targets)
        return loss_d + loss_b + loss_f


class DiceScoreL2Loss(nn.Module):
    """Loss for mask scoring UNet"""
    def __init__(self):
        super(DiceScoreL2Loss, self).__init__()

    def forward(self, logits, dice_logits, masks_gt):
        """
        CH: mask channel

        logits: BATCH x CH x H x W
        dice_logits: BATCH x CH
        masks_gt: BATCH x CH x H x W
        """
        _, _, h, w = masks_gt.size()

        # BATCH x CH x H x W => -1 x H x W
        # TODO is it need sigmoid ???
        masks_preds = torch.sigmoid(logits).view(-1, h, w)

        masks_gt = masks_gt.view(-1, h, w)

        # -1 x H x W => -1
        masks_gt_sum = masks_gt.sum((1, 2))
        pos_index = masks_gt_sum > 0
        if torch.sum(pos_index) == 0:
            return dice_logits.sum() * 0

        # POSNUM x H x W => POSNUM
        pos_intersect = (masks_preds[pos_index] * masks_gt[pos_index]).sum()
        pos_union = masks_preds[pos_index].sum() + masks_gt_sum[pos_index].sum()
        dice_gt_pos = 2. * pos_intersect / pos_union

        # Calc L2 Loss between dice_gt and dice_preds
        # origin code not use sigmoid...
        # dice_preds_pos = dice_logits.view(-1)[pos_index]
        dice_preds_pos = torch.sigmoid(dice_logits.view(-1)[pos_index])

        # Calc L2 Loss
        cond = torch.abs(dice_gt_pos - dice_preds_pos)
        loss = 0.5 * cond**2
        return loss.mean()


class DiceScoreL2LossWrapper(nn.Module):
    """
    Wrapper for Normal Loss
    """
    def __init__(self, base_loss, alpha=1.0):
        super(DiceScoreL2LossWrapper, self).__init__()
        self.base_loss = base_loss
        self.alpha = alpha
        self.ms_loss = DiceScoreL2Loss()

    def forward(self, outputs, masks_gt):
        *logit, mask_score_logit = outputs
        if len(logit) == 1:
            logit = logit[0]

        loss = self.base_loss(logit, masks_gt)
        loss += self.alpha * self.ms_loss(logit, mask_score_logit, masks_gt)
        return loss


class ActiveContourLoss(nn.Module):
    """
    Implementation of active contour loss function for medical image segmentation based on
    "Learning Active Contour Models for Medical Image Segmentation" by Chen, Xu, et al.
    - CVPR 2019 Paper
    - http://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Learning_Active_Contour_Models_for_Medical_Image_Segmentation_CVPR_2019_paper.pdf
    - https://github.com/xuuuuuuchen/Active-Contour-Loss/blob/master/Active-Contour-Loss.py
      - Original code is Keras funciton
    """
    def __init__(self, region_weight=0.5):
        super(ActiveContourLoss, self).__init__()
        self.epsilon = 1e-8
        self.region_weight = region_weight
        self.c1 = 1.  # paper definition
        self.c2 = 0.  # paper definition

    def forward(self, logits, masks):
        """
        logits, masks: BATCH x C x H x W
        """
        preds = torch.sigmoid(logits)

        # Length term
        x = preds[..., 1:, :] - preds[..., :-1, :]  # width direction
        y = preds[..., 1:] - preds[..., :-1]  # height direction
        delta_x = x[..., 1:, :-2] ** 2
        delta_y = y[..., :-2, 1:] ** 2
        loss_length = (delta_x + delta_y + self.epsilon).sqrt().sum()

        # Region term
        # use preds[:, c] and masks[: c] if multi channel
        region_in = preds * ((masks - self.c1) ** 2)
        region_in = region_in.sum().abs()  # equ.(12) in the paper
        region_out = (1. - preds) * ((masks - self.c2) ** 2)
        region_out = region_out.sum().abs()  # equ.(12) in the paper
        loss_region = region_in + region_out

        loss = (1. - self.region_weight) * loss_length + self.region_weight * loss_region

        b, c, h, w = logits.size()
        loss /= (b * c * h * w)  # normalize, this is no line in original code !!!
        return loss


class ClsLossWrapper(nn.Module):
    """Class loss Wrapper"""
    def __init__(self, base_loss, alpha=0.1, use_focal=True, use_class0_cls=True):
        super(ClsLossWrapper, self).__init__()
        self.base_loss = base_loss
        self.alpha = alpha
        self.use_focal = use_focal

        # Sometimes class 0 is background class
        self.use_class0_cls = use_class0_cls

        if use_focal:
            self.cls_loss = BCEFocalLoss()
        else:
            self.cls_loss = nn.BCEWithLogitsLoss()

        # TODO check avg pool
        self.pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, outputs, masks_gt):
        *logit, logit_cls = outputs
        if len(logit) == 1:
            logit = logit[0]
        loss = self.base_loss(logit, masks_gt)

        if not self.use_class0_cls:
            # BATCH x CH x H x W -> BATCH x (CH - 1) x H x W
            masks_gt = masks_gt[:, 1:]

        # BATCH x CH x H x W -> BATCH x CH
        cls_gt = self.pool(masks_gt).squeeze()
        loss += self.alpha * self.cls_loss(logit_cls, cls_gt)
        return loss
