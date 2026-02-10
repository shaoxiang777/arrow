import os
import sys
import cv2
import numpy as np

def xyz2lonlat(xyz):
    atan2 = np.arctan2
    asin = np.arcsin

    norm = np.linalg.norm(xyz, axis=-1, keepdims=True)
    xyz_norm = xyz / norm
    x = xyz_norm[..., 0:1]
    y = xyz_norm[..., 1:2]
    z = xyz_norm[..., 2:]

    lon = atan2(x, z)
    lat = asin(y)
    lst = [lon, lat]

    out = np.concatenate(lst, axis=-1)
    return out

def lonlat2XY(lonlat, shape):
    X = (lonlat[..., 0:1] / (2 * np.pi) + 0.5) * (shape[1] - 1)
    Y = (lonlat[..., 1:] / (np.pi) + 0.5) * (shape[0] - 1)
    lst = [X, Y]
    out = np.concatenate(lst, axis=-1)

    return out 

class Equirectangular:

    _maps_cache = {}

    def __init__(self, img_input):
        # 1. 检查输入是否为字符串（路径）
        if isinstance(img_input, str):
            self._img = cv2.imread(img_input, cv2.IMREAD_COLOR)
            if self._img is None:
                raise FileNotFoundError(f"无法读取路径下的图像: {img_input}")
        # 2. 如果输入已经是图像矩阵（如来自视频流的 NumPy 数组）
        else:
            self._img = img_input
            
        # 获取图像尺寸
        [self._height, self._width, _] = self._img.shape
        
    

    def GetPerspective(self, FOV, THETA, PHI, height, width):
        #
        # THETA is left/right angle, PHI is up/down angle, both in degree
        #
        # 创建一个唯一的 Key 来标识当前的投影参数
        cache_key = (FOV, THETA, PHI, height, width, self._height, self._width)

        if cache_key not in Equirectangular._maps_cache:
            f = 0.5 * width * 1 / np.tan(0.5 * FOV / 180.0 * np.pi)
            cx = (width - 1) / 2.0
            cy = (height - 1) / 2.0
            K = np.array([
                    [f, 0, cx],
                    [0, f, cy],
                    [0, 0,  1],
                ], np.float32)
            K_inv = np.linalg.inv(K)
            
            x = np.arange(width)
            y = np.arange(height)
            x, y = np.meshgrid(x, y)
            z = np.ones_like(x)
            xyz = np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1)
            xyz = xyz @ K_inv.T

            y_axis = np.array([0.0, 1.0, 0.0], np.float32)
            x_axis = np.array([1.0, 0.0, 0.0], np.float32)
            R1, _ = cv2.Rodrigues(y_axis * np.radians(THETA))
            R2, _ = cv2.Rodrigues(np.dot(R1, x_axis) * np.radians(PHI))
            R = R2 @ R1
            xyz = xyz @ R.T
            lonlat = xyz2lonlat(xyz) 
            XY = lonlat2XY(lonlat, shape=self._img.shape).astype(np.float32)
            Equirectangular._maps_cache[cache_key] = (XY[..., 0], XY[..., 1])

        map_x, map_y = Equirectangular._maps_cache[cache_key]
        persp = cv2.remap(self._img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

        return persp
