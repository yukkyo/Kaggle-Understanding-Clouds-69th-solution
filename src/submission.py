import argparse
import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from pathlib import Path
from functools import partial

from factory import read_yaml, get_dataloader
from trainer import LightningModuleSeg
from utils import mask2rle, setup_logger


"""
Make submission csv
"""


def make_parse():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg('--debug', action='store_true', help='debug')
    arg('--config', type=str, default=None, required=True)

    arg('--kfolds', type=str, default=None, required=True,
        help='target dataset kfold: str, if use some folds, ex. 123')

    # Thresholds
    arg('--thres-top', type=float, default=0.6)
    arg('--thres-bottom', type=float, default=0.4)
    arg('--min-contour', type=int, default=10000)
    arg(',,min,contours', type=str, default='12800,8192,10113,10113')

    # Other option
    arg('--thres-before-mean', action='store_true', help='Apply triplet threshold before mean')
    arg('--use-best', action='store_true', help='use best loss weight')
    arg('--use-postprocess', action='store_true', help='use best loss weight')
    return parser.parse_args()


def triplet_thresholds(prob: np.ndarray, top: float,
                       min_area: int, bottom: float):
    area = (prob > top).sum()
    if area < min_area:
        prob[:] = 0
        return prob
    return (prob > bottom).astype(np.float)


def post_process(probability, threshold=0.6, min_size=10000):
    """
    Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored
    """
    # don't remember where I saw it
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((350, 525), np.float32)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num


class Predictor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.pl_model = LightningModuleSeg(cfg)
        self.test_loader = get_dataloader(cfg, phase='test')

        self.fp16 = cfg.General.fp16
        self.test_df_path = cfg.Data.dataset.test_df
        self.labels = cfg.General.labels
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_weight(self, weight_path: str):
        self.pl_model.net = self.pl_model.net.cpu()

        # Convert PyTorch-Lightning checkpoint to normal weight
        loaded_dict = torch.load(weight_path)
        new_dict = dict()
        for k, v in loaded_dict['state_dict'].items():
            new_dict[k.replace('net.', '')] = v

        # Load weights dict
        self.pl_model.net.load_state_dict(new_dict)

        # Apply device info
        if self.fp16:
            self.pl_model.net = self.pl_model.net.half()
        self.pl_model.net = self.pl_model.net.to(self.device)
        self.pl_model.net.eval()

    def predict(self):
        all_img_ids = list()
        all_preds = list()
        total_size = 0

        with torch.no_grad():
            for batch in tqdm(self.test_loader, total=len(self.test_loader)):
                img, img_id = batch
                bs = len(img_id)
                total_size += bs

                if self.fp16:
                    img = img.half()
                img = img.to(self.device)

                # Predict and Test Time Augmentation
                _, preds, preds_cls = self.pl_model.pred_imgs(img)
                preds, preds_cls = self.pl_model.apply_tta(img, preds, preds_cls)
                preds = preds.cpu().numpy().astype(np.float)

                # TODO post process by cls preds
                for i in range(bs):
                    all_img_ids.append(img_id[i])
                    pred_tmp = preds[i]
                    if pred_tmp.shape != (len(self.labels), 350, 525):
                        pred_tmp2 = np.zeros((len(self.labels), 350, 525))
                        for ch in range(len(self.labels)):
                            pred_tmp2[ch] = cv2.resize(pred_tmp[ch], dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
                        pred_tmp = pred_tmp2
                    all_preds.append(pred_tmp)
        return all_img_ids, all_preds

    def make_submission(self, imgid_to_pred) -> pd.DataFrame:
        df_sub = pd.read_csv(self.test_df_path)
        df_sub = df_sub.drop(columns=['EncodedPixels'])

        ret_imgids = list()
        ret_rles = list()

        # Encode pred to rle for each channel
        for img_id, pred in tqdm(imgid_to_pred.items()):
            for i, label in enumerate(self.labels):
                img_id_tmp = f'{img_id}_{label}'
                ret_imgids.append(img_id_tmp)

                # Convert pred to RLE
                pred_tmp = pred[i]
                if pred_tmp.sum() < 1:
                    ret_rles.append('')
                else:
                    ret_rles.append(mask2rle(pred_tmp))

        # Image_Label, EncodedPixels
        df_tmp = pd.DataFrame({'Image_Label': ret_imgids, 'EncodedPixels': ret_rles})
        df = df_sub.merge(df_tmp, on=['Image_Label'], how='left')
        return df


def main():
    args = make_parse()
    cfg = read_yaml(fpath=args.config)
    output_path = Path(cfg.General.workdir)
    n_class = len(cfg.General.labels)

    # Thresholds
    top = args.thres_top
    bottom = args.thres_bottom

    # TODO enable to change min_contour for each class(ch)
    min_areas = list(map(int, args.min_contours.split(',')))
    # min_areas = [12800, 8192, 10113, 10113]
    thres_func = partial(triplet_thresholds, top=top, bottom=bottom)

    # Modify config
    cfg.General.debug = args.debug
    bs = cfg.Data.dataloader.batch_size
    cfg.Data.dataloader.batch_size = bs * len(cfg.General.gpus)

    logger_main = setup_logger(f'test', output_path / f'test.log')

    # Make predictor
    logger_main.info('')
    predictor = Predictor(cfg=cfg)

    # Preds for each kfolds weights
    imgid_to_preds = dict()
    all_kfolds = [int(c) for c in args.kfolds]
    for kfold in all_kfolds:
        if args.use_best:
            weight_path = str(output_path / f'kfold_{kfold}_bestloss.ckpt')
        else:
            weight_path = str(output_path / f'kfold_{kfold}_latest.ckpt')
        print(f'weight load: {weight_path}')
        predictor.load_weight(weight_path=weight_path)

        print(f'predict start')
        img_ids, preds = predictor.predict()

        for imgid, pred in zip(img_ids, preds):
            if args.thres_before_mean:
                for ch in range(n_class):
                    if args.use_postprocess:
                        pred_tmp, num = post_process(pred[ch])
                        pred[ch] = pred_tmp
                    else:
                        pred[ch] = thres_func(pred[ch], min_area=min_areas[ch])

            if imgid in imgid_to_preds:
                imgid_to_preds[imgid] += pred
            else:
                imgid_to_preds[imgid] = pred

    # Make mean preds
    num_kfold = float(len(all_kfolds))
    print(f'num_kfold: {num_kfold}')
    print(f'Make mean pred and postprocess')
    for imgid, pred in tqdm(imgid_to_preds.items()):
        pred /= num_kfold

        if not args.thres_before_mean:
            for ch in range(n_class):
                if args.use_postprocess:
                    pred_tmp, num = post_process(pred[ch])
                    pred[ch] = pred_tmp
                else:
                    pred[ch] = thres_func(pred[ch], min_area=min_areas[ch])

        imgid_to_preds[imgid] = pred

    # Make RLE
    print(f'Make RLE')
    df_final = predictor.make_submission(imgid_to_pred=imgid_to_preds)

    # Make Submission df
    name_df = f'sub_{Path(args.config).stem}_top{str(int(top*10)).zfill(2)}_minarea{"-".join(min_areas)}_' \
        f'bottom{str(int(bottom*10)).zfill(2)}_usebest_{args.use_best}.csv'
    df_final.to_csv(output_path / name_df, index=False)


if __name__ == '__main__':
    main()
