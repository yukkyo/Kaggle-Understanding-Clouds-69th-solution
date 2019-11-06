import argparse
import pandas as pd
from pathlib import Path
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


"""Decription
Make kfold csv from train.csv

Usage:
    $ python data_process/s01_make_kfold_csv.py
"""


def make_parse():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--kfold', type=int, default=5)
    arg('--train-df', type=str, default='../input/train.csv')
    return parser.parse_args()


def main():
    args = make_parse()
    df = pd.read_csv(args.train_df)

    # Add columns
    df['Label'] = df.Image_Label.map(lambda x: x.split('_')[1])
    df['ImageId'] = df.Image_Label.map(lambda x: x.split('_')[0])

    # Extract unique image ids
    labels = ['Fish', 'Flower', 'Gravel', 'Sugar']
    df_new = pd.DataFrame({'ImageId': df.ImageId.unique()})
    df_new['kfold'] = -1

    for l in labels:
        df_tmp = (df[df.Label == l]
                  .drop(columns=['Image_Label', 'Label'])
                  .rename(columns={'EncodedPixels': l})
                  .reset_index(drop=True))
        df_new = df_new.merge(df_tmp, on='ImageId')

    # Make ont-hot vector
    df_new2 = df_new.copy()
    for l in labels:
        df_new2[l] = (~pd.isna(df_new2[l])).astype('int')
    y = df_new2.iloc[:, 2:].values

    # Make kfolds
    indxs = list(range(len(df_new2)))
    mskf = MultilabelStratifiedKFold(n_splits=args.kfold, random_state=42)
    for i, (train_index, test_index) in enumerate(mskf.split(indxs, y)):
        df_new.loc[test_index, 'kfold'] = i + 1

    new_path = Path(args.train_df).parent / f'train_{args.kfold}kfold.csv'
    df_new.to_csv(new_path, index=False)


if __name__ == '__main__':
    main()
