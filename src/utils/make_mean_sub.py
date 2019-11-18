import pandas as pd
import pickle
from tqdm import tqdm
from pathlib import Path
from functools import partial
import gc

from utils import mask2rle, setup_logger, triplet_thresholds, post_process


def make_submission(imgid_to_pred) -> pd.DataFrame:
    print(f'make submission')
    test_df_path='../input/sample_submission.csv'
    labels = ['Fish', 'Flower', 'Gravel', 'Sugar']
    df_sub = pd.read_csv(test_df_path)
    df_sub = df_sub.drop(columns=['EncodedPixels'])

    ret_imgids = list()
    ret_rles = list()

    # Encode pred to rle for each channel
    for img_id, pred in tqdm(imgid_to_pred.items()):
        for i, label in enumerate(labels):
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
    imgid_to_preds_paths = [
        # '../output/model/model056/preds_model056_kfolds12345_usebest_True.pkl',
        # '../output/model/model052/preds_model052_kfolds12345_usebest_True.pkl',
        # '../output/model/model059/preds_model059_kfolds12345_usebest_True.pkl',
        '../output/model/model063/preds_model063_kfolds12345_usebest_True.pkl',
        '../output/model/model064/preds_model064_kfolds12345_usebest_True.pkl'
    ]
    print(imgid_to_preds_paths)

    # thresholds
    # min_areas = [12500, 12500, 10000, 6000]  # Model 052 best, after add, 0.647366
    # min_areas = [14000., 10500., 9000., 8100.]  # Model 052 best, 0.6494966864585876

    min_areas = [17500, 10000, 10000, 5000]  # Model 056 best, after add, 	0.647339
    # min_areas = [16000.0, 10500.0, 10000.0, 7000.0]  # Model 056 best, 0.64889

    min_areas = [15000, 11250, 10000, 5500]  # mean of after add 052, 056

    # min_areas = [7500, 12500, 7500, 12500]  # Model 022 best
    top_score_thresholds = 0.6
    bottom_score_thresholds = 0.4

    # Best 059 model after add, 0.648148
    top_score_thresholds = 0.6
    bottom_score_thresholds = 0.45
    min_areas = [17500, 10000, 10000, 10000]  # mean of after add 052, 056

    # average of 052 and 059
    top_score_thresholds = 0.6
    bottom_score_thresholds = 0.45
    min_areas = [17500, 10000, 10000, 7500]  # mean of after add 052, 059
    min_areas = [25000, 12500, 12500, 7500]  # mean of after add 052, 059

    # # best 063
    # top_score_thresholds = 0.6
    # bottom_score_thresholds = 0.4
    # min_areas = [17500, 15000, 12500, 5000]

    # # best 064
    # top_score_thresholds = 0.65
    # bottom_score_thresholds = 0.45
    # min_areas = [15000, 12500, 7500, 5000]

    # # best average of 063, 064
    # top_score_thresholds = 0.625
    # bottom_score_thresholds = 0.425
    # min_areas = [16250, 13750, 10000, 5000]

    thres_func = partial(triplet_thresholds, top=top_score_thresholds, bottom=bottom_score_thresholds)

    # Read pickle objects
    imgid_to_pred = dict()
    for p in imgid_to_preds_paths:
        print(f'load: {p}')
        with open(p, mode='rb') as f:
            imgid_to_pred_tmp = pickle.load(f)

        for imgid, pred in imgid_to_pred_tmp.items():
            if imgid in imgid_to_pred:
                imgid_to_pred[imgid] += pred
            else:
                imgid_to_pred[imgid] = pred

    # Make Average
    print('Make average')
    num_tmp = float(len(imgid_to_preds_paths))
    n_class = 4
    for imgid, pred in tqdm(imgid_to_pred.items()):
        pred /= num_tmp

        for ch in range(n_class):
            pred[ch] = thres_func(pred[ch], min_area=min_areas[ch])

        imgid_to_pred[imgid] = pred

    # Convert RLE
    df_name = f'sub_model063064_kfolds12345_top{str(int(top_score_thresholds*100)).zfill(3)}' \
        f'_minarea{"-".join(str(int(m)) for m in min_areas)}' \
        f'_bottom{str(int(bottom_score_thresholds*100)).zfill(3)}_usebest_052059thres_modified.csv'
    df_path = Path('../output/final') / df_name
    df = make_submission(imgid_to_pred)
    df.to_csv(df_path, index=False)


if __name__ == '__main__':
    main()
