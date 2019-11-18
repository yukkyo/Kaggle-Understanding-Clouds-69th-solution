import argparse
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from itertools import product
from collections import defaultdict
import cv2
from functools import partial

from factory import read_yaml, get_dataloader
from utils import setup_logger, dice_pos_neg
from trainer import LightningModuleSeg


"""
Eval trained weight
Search best thresholds for each class
"""


class Verifier:
    def __init__(self, pl_model, dataloader, metrics_func, fp16=False):
        self.pl_model = pl_model
        self.dataloader = dataloader
        self.metrics_func = metrics_func
        self.fp16 = fp16

        # Model & GPU setting
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.fp16:
            self.pl_model.net = self.pl_model.net.half()
        self.pl_model.net = self.pl_model.net.to(self.device)
        self.pl_model.net.eval()  # Caution: this code change bn output ?

        self.test_height = 350
        self.test_width = 525
        self.resize = partial(cv2.resize, dsize=(self.test_width, self.test_height))

        self._make_preds()

    def _resize(self, x: torch.Tensor):
        bs, ch, _, _ = x.size()
        x_new = np.zeros((bs, self.test_height, self.test_width, ch))

        # (BS, CH, H, W) -> (BS, H, W, CH)
        x_tmp = x.permute(0, 2, 3, 1).numpy()
        for i in range(bs):
            x_new[i] = self.resize(x_tmp[i])

        # (BS, H, W, CH) -> (BS, CH, H, W)
        x_new2 = torch.from_numpy(x_new).permute(0, 3, 1, 2).float()
        return x_new2

    def _make_preds(self):
        self.all_preds = list()
        self.all_masks = list()
        total_bs = 0

        with torch.no_grad():
            for batch in tqdm(self.dataloader, total=len(self.dataloader)):
                img, gt_mask = batch
                total_bs += gt_mask.size(0)
                if self.fp16:
                    img = img.half()
                    gt_mask = gt_mask.half()
                img = img.to(self.device)

                # Predict and Test Time Augmentation
                _, preds, preds_cls = self.pl_model.pred_imgs(img)
                preds, preds_cls = self.pl_model.apply_tta(img, preds, preds_cls)

                preds = self._resize(preds.cpu().float())
                gt_mask = self._resize(gt_mask.cpu().float())
                self.all_preds.append(preds)
                self.all_masks.append(gt_mask)
        print(f'Total size: {total_bs}')

    def valid_thres(self, func_args):
        dice, dice_pos, dice_neg = list(), list(), list()

        with torch.no_grad():
            for preds, gt_mask in tqdm(zip(self.all_preds, self.all_masks), total=len(self.dataloader)):
                preds = preds.to(self.device)
                gt_mask = gt_mask.to(self.device)
                # TODO return metrics name
                d, d_pos, d_neg = self.metrics_func(preds, gt_mask, **func_args)
                dice.append(d)
                dice_pos.append(d_pos)
                dice_neg.append(d_neg)

        # Calc mean
        dice = torch.cat(dice, dim=0).mean().cpu().item()
        dice_pos = torch.cat(dice_pos, dim=0).mean().cpu().item()
        dice_neg = torch.cat(dice_neg, dim=0).mean().cpu().item()
        return dice, dice_pos, dice_neg


def make_parse():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg('--debug', action='store_true', help='debug')
    arg('--config', type=str, default=None, required=True)
    arg('--model-path', type=str, default=None, required=True)
    arg('--kfold', type=int, default=None, required=True, help='target dataset kfold')
    arg('--tta-rotate180', action='store_true', help='debug')
    arg('--debug-thres', action='store_true', help='debug')
    return parser.parse_args()


def main():
    args = make_parse()
    cfg = read_yaml(fpath=args.config)
    cfg.Data.dataset.kfold = args.kfold
    cfg.Data.dataloader.valid.use_sampler = False
    cfg.General.debug = args.debug

    if args.tta_rotate180:
        print(f'Enable tta rotate180 !')
        cfg.Augmentation.tta = ['hflip', 'vflip', 'rotate180']
    else:
        cfg.Augmentation.tta = ['hflip', 'vflip']

    for key, value in cfg.items():
        print(f"    {key.ljust(30)}: {value}")

    n_class = len(cfg.General.labels)
    output_path = Path('../output/model') / Path(args.config).stem

    # Main logger
    logger_name = f'eval_{Path(args.config).stem}_{Path(args.model_path).stem}.log'
    logger_main = setup_logger(f'eval', output_path / logger_name)

    # PyTorch Lightning module
    pl_model = LightningModuleSeg(cfg)

    # Get dataloader
    bs = cfg.Data.dataloader.batch_size
    cfg.Data.dataloader.batch_size = bs * len(cfg.General.gpus)
    val_dataloader = get_dataloader(cfg, phase='valid')

    # Load trained weight
    logger_main.info('Load trained weight')
    loaded_dict = torch.load(args.model_path)
    new_dict = dict()
    logger_main.info('Convert dict')
    for k, v in loaded_dict['state_dict'].items():
        new_dict[k.replace('net.', '')] = v
    logger_main.info('Load state dict')
    pl_model.net.load_state_dict(new_dict)

    # Make tester
    logger_main.info('Make valider')
    valider = Verifier(
        pl_model=pl_model,
        dataloader=val_dataloader,
        metrics_func=dice_pos_neg,
        fp16=cfg.General.fp16,
    )

    logger_main.info('Start eval !')

    top_thresholds = [0.6, 0.65]
    bottom_thresholds = [0.4, 0.45]

    img_area = cfg.Data.dataset.img_height * cfg.Data.dataset.img_width
    min_sizes = (img_area / np.arange(2.5, 6.5, 0.5) ** 2).astype(np.int).tolist()
    min_sizes = sorted(min_sizes)

    min_sizes_eachclass = {
        0: [10000, 12500, 15000, 17500, 20000, 22500, 25000],
        1: [7500, 10000, 12500, 15000, 17500],
        2: [7500, 10000, 12500, 15000, 17500],
        3: [5000, 7500, 10000, 12500],
    }

    if args.debug_thres:
        top_thresholds = [0.6]
        bottom_thresholds = [0.4]
        min_sizes_eachclass = {i: [10000] for i in range(4)}

    df_dict = defaultdict(list)
    logger_main.info(f'top_thres,min_contours,bottom_thres,dice,dice_pos,dice_neg')
    combs = product(
        top_thresholds,
        bottom_thresholds,
        min_sizes_eachclass[0],
        min_sizes_eachclass[1],
        min_sizes_eachclass[2],
        min_sizes_eachclass[3]
    )
    for top_thres, bottom_thres, *min_contours in combs:
        func_args_tmp = {
            'num_class': n_class,
            'top_score_thresholds': [top_thres],
            'min_contour_areas': min_contours,
            'bottom_score_thresholds': [bottom_thres],
            'post_process': False,
            'skip_first_channel': valider.pl_model.skip_first_class
        }

        # Eval each thresholds
        dice, dice_pos, dice_neg = valider.valid_thres(func_args_tmp)
        logger_main.info(f'{top_thres},{min_contours},{bottom_thres},{dice},{dice_pos},{dice_neg}')

        # Add values to df_dict
        for k, v in func_args_tmp.items():
            df_dict[k].append(v)
        df_dict['dice'].append(dice)
        df_dict['dice_pos'].append(dice_pos)
        df_dict['dice_neg'].append(dice_neg)

    if args.debug or args.debug_thres:
        return

    df = pd.DataFrame(df_dict)
    mode = 'best' if 'best' in args.model_path else 'latest'
    fname = f'search_thres_{Path(args.config).stem}_{Path(args.model_path).stem}_kfold{args.kfold}_mode_{mode}.csv'
    df.to_csv(output_path / fname, index=False)


if __name__ == '__main__':
    main()
