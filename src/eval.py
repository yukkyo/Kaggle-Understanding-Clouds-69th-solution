import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

from factory import read_yaml, get_model, get_dataloader
from utils import src_backup, setup_logger, dice_pos_neg, dice_nomask


"""
Eval trained weight
Search best thresholds for each class if args.find_thresholds == True
"""


class Verifier:
    def __init__(self, model, dataloader, metrics_func, out_channel,
                 mask_score=False, fp16=False, tta_flip=False, softmax=False, remove_black_area=False):
        self.model = model
        self.dataloader = dataloader
        self.metrics_func = metrics_func
        self.mask_score = mask_score
        self.fp16 = fp16
        self.tta_flip = tta_flip
        self.softmax = softmax
        self.out_channel = out_channel
        self.remove_black_area = remove_black_area

        # Model & GPU setting
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.fp16:
            self.model = self.model.half()
        self.model = self.model.to(self.device)
        self.model.eval()  # Caution: this code change bn output ?

        self.all_preds, self.all_masks = self._make_preds()

    def _make_preds(self):
        all_preds = list()
        all_masks = list()

        with torch.no_grad():
            for batch in tqdm(self.dataloader, total=len(self.dataloader)):
                img, gt_mask = batch
                if self.fp16:
                    img = img.half()
                    gt_mask = gt_mask.half()

                img = img.to(self.device)
                gt_mask = gt_mask.to(self.device)

                # Predict
                logits = self.model(img)
                if self.mask_score:
                    logits = logits[0]  # mask score unet return logits,mask_score

                if self.softmax:
                    # softmax channel-wise
                    preds = torch.softmax(logits, dim=1).float()
                else:
                    preds = torch.sigmoid(logits)

                # Test Time Augmentation
                if self.tta_flip:
                    logits_flip = self.model(img.flip(3))
                    if self.mask_score:
                        logits_flip = logits_flip[0]  # mask score unet return logits,mask_score
                    if self.softmax:
                        # softmax channel-wise
                        preds_flip = torch.softmax(logits_flip, dim=1).float()
                    else:
                        preds_flip = torch.sigmoid(logits_flip)
                    preds_flip = preds_flip.flip(3)
                    preds = (preds + preds_flip) / 2.

                if self.softmax:
                    preds = F.one_hot(preds.argmax(dim=1),
                                      self.out_channel).transpose(2, 3).transpose(1, 2).float()

                if self.remove_black_area:
                    # BATCH x CH x H x W -> BATCH x H x W -> BATCH x 1 x H x W
                    not_black = (img.sum(dim=1) > 0.).unsqueeze(1).float()
                    preds = not_black * preds
                all_preds.append(preds.cpu())
                all_masks.append(gt_mask.cpu())
        return all_preds, all_masks

    def valid_thres(self, func_args):
        dice, dice_pos, dice_neg = list(), list(), list()

        with torch.no_grad():
            for preds, gt_mask in tqdm(zip(self.all_preds, self.all_masks), total=len(self.dataloader)):
                preds = preds.to(self.device)
                gt_mask = gt_mask.to(self.device)

                # Calc metrics
                d, d_pos, d_neg = self.metrics_func(preds, gt_mask, **func_args)
                dice.append(d)
                dice_pos.append(d_pos)
                dice_neg.append(d_neg)

        # Mean
        dice = torch.cat(dice, dim=0)
        dice_pos = torch.cat(dice_pos, dim=0)
        dice_neg = torch.cat(dice_neg, dim=0)
        return dice.mean(), dice_pos.mean(), dice_neg.mean()


def make_parse():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg('--debug', action='store_true', help='debug')
    arg('--config', type=str, default=None, required=True)
    arg('--model-path', type=str, default=None, required=True)
    arg('--kfold', type=int, default=None, required=True, help='target dataset kfold')
    return parser.parse_args()


def main():
    args = make_parse()
    cfg = read_yaml(fpath=args.config)
    cfg.Data.dataset.kfold = args.kfold
    cfg.Data.dataloader.valid.use_sampler = False
    cfg.General.debug = args.debug

    for key, value in cfg.items():
        print(f"    {key.ljust(30)}: {value}")

    n_class = len(cfg.General.labels)
    output_path = Path(cfg.General.workdir)

    # Main logger
    logger_main = setup_logger(f'eval', output_path / f'eval_{args.kfold}_kfold.log')

    # Get model and dataloader
    model = get_model(cfg)
    val_dataloader = get_dataloader(cfg, phase='valid')

    # Load trained weight
    logger_main.info('Load trained weight')
    loaded_dict = torch.load(args.model_path)
    new_dict = dict()
    logger_main.info('Convert dict')
    for k, v in loaded_dict['state_dict'].items():
        new_dict[k.replace('net.', '')] = v
    logger_main.info('Load state dict')
    model.load_state_dict(new_dict)

    # Make tester
    logger_main.info('Make valider')
    valider = Verifier(
        model=model,
        dataloader=val_dataloader,
        # metrics_func=dice_pos_neg,
        metrics_func=dice_nomask,
        mask_score=(cfg.Model.model_arch in {'msunet', 'clsunet'}),
        fp16=cfg.General.fp16,
        # TODO enable to input some ttas as List
        # tta_flip=('flip' in cfg.Augmentation.tta),
        tta_flip=True,
        out_channel=cfg.Model.out_channel,
        softmax=(cfg.Model.output == 'softmax'),
        # TODO add config
        remove_black_area=True
    )

    logger_main.info('Start eval !')
    # top_thresholds_tmp = [0.5] * 4
    # min_contours_tmp = [256] * 4
    # bottom_thresholds_tmp = [0.4] * 4
    #
    # func_args_tmp = {
    #     'num_class': n_class,
    #     'top_score_thresholds': top_thresholds_tmp,
    #     'min_contour_areas': min_contours_tmp,
    #     'bottom_score_thresholds': bottom_thresholds_tmp,
    #     'skip_first_channel': n_class < cfg.Model.out_channel
    # }
    func_args_tmp = {
        'num_class': n_class,
        'threshold': 0.5,
        'skip_first_class': n_class < cfg.Model.out_channel
    }

    for k, v in func_args_tmp.items():
        logger_main.info(f'{k}: {v}')

    # Eval each thresholds
    dice, dice_pos, dice_neg = valider.valid_thres(func_args_tmp)
    logger_main.info(f'dice: {dice}, dice_pos: {dice_pos}, dice_neg: {dice_neg}')


if __name__ == '__main__':
    main()
