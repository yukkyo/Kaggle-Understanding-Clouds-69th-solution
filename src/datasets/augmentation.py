import numpy as np
import cv2

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


class RandomInputBlackEllipse(DualTransform):
    def __init__(self, p=0.5, sl=0.02, sh=0.04, r1=0.3, r2=1/0.3, max_angle=180,
                 img_h=None, img_w=None, always_apply=False):
        super(RandomInputBlackEllipse, self).__init__(always_apply, p)
        self.EPSILON = p
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.r2 = r2
        self.max_angle = max_angle
        self.img_h = img_h
        self.img_w = img_w

    def apply(self, img, x_center=0, y_center=0, w=0, h=0, angle=0, **params):
        img = cv2.ellipse(
            img, (x_center, y_center), (w, h),
            angle=angle, startAngle=0, endAngle=360, color=(0, 0, 0), thickness=-1,
        )
        return img

    def apply_to_mask(self, img, x_center=0, y_center=0, w=0, h=0, angle=0, **params):
        img = cv2.ellipse(
            img, (x_center, y_center), (w, h),
            angle=angle, startAngle=0, endAngle=360, color=(0, 0, 0), thickness=-1,
        )
        return img

    def get_params(self):
        angle = np.random.randint(self.max_angle)
        while True:
            s = np.random.uniform(self.sl, self.sh) * self.img_h * self.img_w
            r = np.random.uniform(self.r1, self.r2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, self.img_w)
            top = np.random.randint(0, self.img_h)

            if left + w <= self.img_w and top + h <= self.img_h:
                break
        x_center = left + w//2
        y_center = top + h//2
        return {"x_center": x_center, "y_center": y_center, "w": w, "h": h, "angle": angle}

    def get_transform_init_args_names(self):
        return 'sl', 'sh', 'r1', 'r2', 'max_angle'
