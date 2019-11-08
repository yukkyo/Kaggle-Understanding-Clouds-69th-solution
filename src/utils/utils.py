import os
import random
import numpy as np
import torch
import logging
from pathlib import Path
import shutil
from typing import List


def rle_decode(mask_rle: str = '', shape: tuple = (1400, 2100)):
    """
    Decode rle encoded mask.

    :param mask_rle: run-length as string formatted (start length)
    :param shape: (height, width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')


def make_mask_from_rles(rles: List[str], shape: tuple = (1400, 2100)):
    """
    Create mask based on df, image name and shape.
    TODO check making empty mask
    """
    masks = np.zeros((shape[0], shape[1], 4), dtype=np.float32)

    for idx, label in enumerate(rles):
        if label is not np.nan:
            mask = rle_decode(label)
            masks[:, :, idx] = mask

    return masks


def mask2rle(img):
    """
    Convert mask to rle.
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    """
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def seed_everything(seed=73):
    """
      https://www.kaggle.com/soulmachine/siim-deeplabv3
      Make PyTorch deterministic.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def setup_logger(logger_name, log_file, level=logging.INFO, null_format=False):
    logger_target = logging.getLogger(logger_name)
    if null_format:
        formatter = logging.Formatter('')
    else:
        formatter = logging.Formatter('%(asctime)s : %(message)s')
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger_target.setLevel(level)
    logger_target.addHandler(file_handler)
    logger_target.addHandler(stream_handler)

    return logging.getLogger(logger_name)


def src_backup(input_dir: Path, output_dir: Path):
    for src_path in input_dir.glob("**/*.py"):
        new_path = output_dir / src_path.name
        shutil.copy2(str(src_path), str(new_path))
