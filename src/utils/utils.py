import os
import random
import numpy as np
import torch
import logging
from pathlib import Path
import shutil


def rles_to_mask(rles, non_mask_class=False, num_class=4, img_size=(1400, 2100)):
    n_class = num_class + int(non_mask_class)
    mask_size = (*img_size, n_class)
    masks = np.zeros(mask_size, dtype=np.float32)  # float32 is V.Imp
    # 4:class 1～4 (ch:0 - 3)

    for idx, label in enumerate(rles):
        if non_mask_class:
            idx += 1
        if isinstance(label, str):
            # if label is not np.nan:
            label = label.split(" ")
            positions = map(int, label[0::2])
            length = map(int, label[1::2])
            mask = np.zeros(img_size[0] * img_size[1], dtype=np.uint8)
            for pos, le in zip(positions, length):
                mask[pos:(pos + le)] = 1
            masks[:, :, idx] = mask.reshape(img_size, order='F')

    if non_mask_class:
        masks[:, :, 0] = 1. - masks[:, :, 1:].sum(axis=2)
    return masks


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
    print("* src Backup start !")
    for src_path in input_dir.glob("**/*.py"):
        new_path = output_dir / src_path.name
        shutil.copy2(str(src_path), str(new_path))
    print("* src Backup end !")
