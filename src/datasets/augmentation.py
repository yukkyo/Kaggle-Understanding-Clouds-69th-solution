import numpy as np

from albumentations.core.transforms_interface import DualTransform

"""
Custom augmentation
"""


class RandomInputBlack(DualTransform):
    def __init__(self, p=0.5, sl=0.1, sh=0.3, r1=0.3, r2=1/0.3, img_h=None, img_w=None, always_apply=False):
        super(RandomInputBlack, self).__init__(always_apply, p)
        self.EPSILON = p
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.r2 = r2
        self.img_h = img_h
        self.img_w = img_w

    def apply(self, img, xmin=0, ymin=0, xmax=0, ymax=0, **params):
        img[ymin:ymax, xmin:xmax, :] = 0
        return img

    def apply_to_mask(self, img, xmin=0, ymin=0, xmax=0, ymax=0, **params):
        img[ymin:ymax, xmin:xmax, :] = 0
        return img

    def get_params(self):
        while True:
            s = np.random.uniform(self.sl, self.sh) * self.img_h * self.img_w
            r = np.random.uniform(self.r1, self.r2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, self.img_w)
            top = np.random.randint(0, self.img_h)

            if left + w <= self.img_w and top + h <= self.img_h:
                break
        return {"xmin": left, "ymin": top, "xmax": left + w, "ymax": top + h}

    def get_transform_init_args_names(self):
        return 'sl', 'sh', 'r1', 'r2'
