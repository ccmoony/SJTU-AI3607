from typing import List

import cv2
import numpy as np
import torch
from torch import Tensor


class ScaleToLimitRange:
    def __init__(self, w_lo: int, w_hi: int, h_lo: int, h_hi: int) -> None:
        assert w_lo <= w_hi and h_lo <= h_hi
        self.w_lo = w_lo
        self.w_hi = w_hi
        self.h_lo = h_lo
        self.h_hi = h_hi

    def __call__(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        r = h / w
        lo_r = self.h_lo / self.w_hi
        hi_r = self.h_hi / self.w_lo
        assert lo_r <= h / w <= hi_r, f"img ratio h:w {r} not in range [{lo_r}, {hi_r}]"

        scale_r = min(self.h_hi / h, self.w_hi / w)
        if scale_r < 1.0:
            # one of h or w highr that hi, so scale down
            img = cv2.resize(
                img, None, fx=scale_r, fy=scale_r, interpolation=cv2.INTER_LINEAR
            )
            return img

        scale_r = max(self.h_lo / h, self.w_lo / w)
        if scale_r > 1.0:
            # one of h or w lower that lo, so scale up
            img = cv2.resize(
                img, None, fx=scale_r, fy=scale_r, interpolation=cv2.INTER_LINEAR
            )
            return img

        # in the rectangle, do not scale
        assert self.h_lo <= h <= self.h_hi and self.w_lo <= w <= self.w_hi
        return img


class ScaleAugmentation:
    def __init__(self, lo: float, hi: float) -> None:
        assert lo <= hi
        self.lo = lo
        self.hi = hi

    def __call__(self, img: np.ndarray) -> np.ndarray:
        k = np.random.uniform(self.lo, self.hi)
        img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_LINEAR)
        return img
'''
Add random gausian noise to the image
'''
class GausianBlur:
    def __init__(self, scale: float=1.0) -> None:
        self.scale =scale
       
    def __call__(self, img: np.ndarray) -> np.ndarray:
        noise = 0.001*self.scale*np.random.randn(*img.shape)
        img = img + noise
        return img

'''
Randomly rotate the image(less than max_angle)
'''
class RandomRotate:
    def __init__(self, max_angle: float=5.0) -> None:
        self.max_angle = max_angle
       
    def __call__(self, img: np.ndarray) -> np.ndarray:
            rows, cols = img.shape[:2]
            angle = np.random.uniform(-self.max_angle, self.max_angle)
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
            rotated_image = cv2.warpAffine(img, M, (cols, rows))
            return rotated_image
