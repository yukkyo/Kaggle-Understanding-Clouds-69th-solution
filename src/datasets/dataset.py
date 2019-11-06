import os
import cv2
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset

from utils import rles_to_mask


BAD_IMAGE_IDS = {
    # https://www.kaggle.com/c/understanding_cloud_organization/discussion/105359#latest-629844
    '046586a.jpg',
    '1588d4c.jpg',
    '1e40a05.jpg',
    '41f92e5.jpg',
    '449b792.jpg',
    '563fc48.jpg',
    '8bd81ce.jpg',
    'b092cc1.jpg',
    'c0306e5.jpg',
    'c26c635.jpg',
    'e04fea3.jpg',
    'e5f2f24.jpg',
    'eda52f2.jpg',
    'fa645da.jpg',
}


class CloudDataset(Dataset):
    def __init__(self, df_path, img_dir, alb_transforms, img_height=1400, img_width=2100,
                 kfold=1, phase='train', background_class=False, debug=False, remove_bad_img=False):
        assert phase in {'train', 'valid', 'test'}, f"phase must be train or valid or test, but got {phase}"

        self.df_path = df_path
        self.img_dir = img_dir
        self.transforms = alb_transforms
        self.kfold = kfold
        self.phase = phase
        self.img_height = img_height
        self.img_width = img_width
        self.remove_bad_img = remove_bad_img

        self.background_class = background_class
        self.label_def = ['Fish', 'Flower', 'Gravel', 'Sugar']
        self.num_class = 4

        # ImageId, kfold, EncodedPixels_1, EncodedPixels_2, EncodedPixels_3, EncodedPixels_4
        self.df = pd.read_csv(self.df_path)

        if remove_bad_img:
            self.df = self.df[~self.df.ImageId.isin(BAD_IMAGE_IDS)]

        if debug:
            self.df = self.df[:200]

        if phase == 'train':
            self.df = self.df[self.df.kfold != kfold].reset_index(drop=True)
        elif phase == 'valid':
            self.df = self.df[self.df.kfold == kfold].reset_index(drop=True)

        self.labels = list()
        if phase in {'train', 'valid'}:
            nas = [self.df[f'EncodedPixels_{i + 1}'].isna() for i in range(self.num_class)]
            is_any_exists = sum(nas)
            self.labels = [1 if x > 0 else 0 for x in is_any_exists]

        if phase == 'test':
            self.img_ids = self.df.Image_Label.map(lambda x: x.split('_')[0]).unique().tolist()
        else:
            self.img_ids = self.df.ImageId.tolist()
        self.img_paths = [os.path.join(self.img_dir, img_id) for img_id in self.img_ids]

        if phase != 'test':
            self.rles = list()
            row_names = [f'EncodedPixels_{i+1}' for i in range(self.num_class)]
            for _, row in tqdm(self.df.iterrows()):
                rle_tmp = list()
                for r_name in row_names:
                    rle_tmp.append(row[r_name])
                self.rles.append(rle_tmp)

        print(f"{self.phase} len(self.img_paths): {len(self.img_paths)}")

    def __getitem__(self, idx):
        img = self.get_img(idx)

        if self.phase == 'test':
            img = self.transforms(image=img)["image"]
            return img, self.img_ids[idx]

        mask = self.get_mask(idx)

        # Augmentation
        augmented = self.transforms(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']  # height x width x mask_ch
        mask = mask.permute(2, 0, 1)  # mask_ch x height x width
        return img.float(), mask.float()

    def __len__(self):
        return len(self.img_paths)

    def get_img(self, idx):
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path).astype(float)
        img = cv2.resize(img, (self.img_width, self.img_height))
        return img

    def get_mask(self, idx):
        mask = rles_to_mask(self.rles[idx], non_mask_class=self.background_class)
        mask = cv2.resize(mask, (self.img_width, self.img_height))
        return mask
