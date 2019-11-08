import torch
from typing import List


def dice_pos_neg_1ch(preds, labels, top_score_threshold=0.5,
                     min_contour_area=0, bottom_score_threshold=None):
    """
    Dice score separately pos and neg for 1ch

    preds: after torch.sigmoid(), BATCH x H x W
    labels: 0 or 1, BATCH x H x W
    """
    assert preds.size() == labels.size()

    p = (preds > top_score_threshold).float()
    t = labels.float()

    sum_dim = (1, 2)  # H x W
    t_sum = t.sum(sum_dim)
    p_sum = p.sum(sum_dim)
    is_pos_true = t_sum > 0
    is_neg_true = t_sum == 0

    # Predict mask or no mask by min_contour_area(with top_score_threshold)
    is_neg_pred = p_sum <= min_contour_area

    # Calc Dice neg_index
    # The Dice coefficient is defined to be 1 when both X and Y are empty.
    dice_neg = (is_neg_pred.float() * is_neg_true.float())[is_neg_true]

    # Triplet threshold
    if bottom_score_threshold is not None:
        p = (preds > bottom_score_threshold).float()

    # Predict 0
    p[is_neg_pred] = 0.
    p_sum = p.sum(sum_dim)

    # Calc Dice pos_index, not need eps in Dice because true label non zero
    intersect = (p * t).sum(sum_dim)[is_pos_true]
    union = (t_sum + p_sum)[is_pos_true]
    dice_pos = 2 * intersect / union

    ret_dice = torch.cat([dice_pos, dice_neg], dim=0)
    return ret_dice, dice_pos, dice_neg


def dice_pos_neg(preds, labels, num_class=4,
                 top_score_thresholds: List[float] = None,
                 min_contour_areas: List[int] = None,
                 bottom_score_thresholds: List[float] = None,
                 post_process: bool = False,
                 skip_first_channel: bool = False):
    """return dice score for each <ImageId, ClassId>"""

    if len(top_score_thresholds) == 1:
        top_score_thresholds = top_score_thresholds * num_class
    if len(min_contour_areas) == 1:
        min_contour_areas = min_contour_areas * num_class
    if len(bottom_score_thresholds) == 1:
        bottom_score_thresholds = bottom_score_thresholds * num_class

    if skip_first_channel:
        # BATCH x CH x H x W -> BATCH x (CH - 1) x H x W
        preds = preds[:, 1:]

    dices = list()
    dices_pos = list()
    dices_neg = list()

    if post_process:
        max_tmp = preds.max(dim=1, keepdim=True)[0]
        preds[preds < max_tmp] = 0.

    for i in range(num_class):
        dice_tmp, dice_pos, dice_neg = dice_pos_neg_1ch(
            preds[:, i], labels[:, i],
            top_score_threshold=top_score_thresholds[i],
            min_contour_area=min_contour_areas[i],
            bottom_score_threshold=bottom_score_thresholds[i]
        )
        dices.append(dice_tmp)
        dices_pos.append(dice_pos)
        dices_neg.append(dice_neg)
    dices = torch.cat(dices, dim=0)
    dices_pos = torch.cat(dices_pos, dim=0)
    dices_neg = torch.cat(dices_neg, dim=0)
    return dices, dices_pos, dices_neg


def dice_pytorch(preds, targets, noise_thr: int = 0):
    """
    https://www.kaggle.com/iafoss/hypercolumns-pneumothorax-fastai-0-831-lb
    logits: Batchsize x 1 x H x W
    targets: Batchsize x 1 x H x W
    """
    eps = 1e-8
    sum_dim = (2, 3)

    # remove noisable preds
    preds_sum = preds.sum(sum_dim)  # Batchsize x 1
    preds[preds_sum < noise_thr] = 0

    # intersect
    intersect = (preds * targets).sum(sum_dim).float()  # Batchsize x 1

    # union = |A| + |B|
    union = (preds + targets).sum(sum_dim).float()  # Batchsize x 1

    # dice
    dice = (2.0 * intersect + eps) / (union + eps)  # Batchsize x 1
    return dice


def dice_nomask(preds, labels, num_class=5, threshold=0.5,
                skip_first_class=False, min_contour_area=1024):
    """Normal dice"""
    dices = list()
    dices_pos = list()
    dices_neg = list()

    for i in range(num_class):
        if i == 0 and skip_first_class:
            # Skip nonmask class
            continue
        dice_tmp, dice_pos, dice_neg = dice_pos_neg_1ch(
            preds[:, i], labels[:, i],
            top_score_threshold=threshold,
            min_contour_area=min_contour_area
        )
        dices.append(dice_tmp)
        dices_pos.append(dice_pos)
        dices_neg.append(dice_neg)
    dices = torch.cat(dices, dim=0)
    dices_pos = torch.cat(dices_pos, dim=0)
    dices_neg = torch.cat(dices_neg, dim=0)
    return dices, dices_pos, dices_neg
