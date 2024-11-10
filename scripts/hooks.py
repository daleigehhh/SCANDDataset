import cv2
import numpy as np
from typing import Callable, Optional, Any, Tuple
from functools import wraps
import os
import pandas as pd


class SaveBackViewHook(object):
    def __init__(self,
                 base_path: str,
                 fps: int = 30,
                 ):
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        self.base_path = base_path
        self.fps = fps
        self.four_cc = cv2.VideoWriter_fourcc(*'XVID')

    def __call__(self, func: Callable[..., np.array]) -> Callable[..., np.array]:
        @wraps(func)
        def wrapper(*args, **kwargs):
            binary_map, bag_name = func(*args, **kwargs)
            frame_size = binary_map.shape[1:]
            save_path = os.path.join(self.base_path, bag_name.split('.')[0] + '_back.avi')
            writer = cv2.VideoWriter(save_path,
                                     self.four_cc,
                                     self.fps,
                                     frame_size,
                                     isColor=False)
            for i in range(len(binary_map)):
                frame = binary_map[i].astype(np.uint8) * 255
                writer.write(frame)
            writer.release()
            return binary_map, bag_name
        return wrapper

class SaveFrontViewHook(object):
    def __init__(self,
                 base_path: str,
                 fps: int = 30):
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        self.base_path = base_path
        self.fps = fps
        self.four_cc = cv2.VideoWriter_fourcc(*'XVID')

    def __call__(self, func: Callable[..., np.array]) -> Callable[..., np.array]:
        @wraps(func)
        def wrapper(*args, **kwargs):
            binary_map, bag_name = func(*args, **kwargs)
            binary_map_flipped = np.flip(binary_map, 0)
            frame_size = binary_map.shape[1:]
            save_path = os.path.join(self.base_path,
                                     bag_name.split('.')[0] + '_front.avi')
            writer = cv2.VideoWriter(save_path,
                                     self.four_cc,
                                     self.fps,
                                     frame_size,
                                     isColor=False)
            for i in range(len(binary_map_flipped)):
                frame = binary_map[i].astype(np.uint8) * 255
                writer.write(frame)
            writer.release()
            return binary_map, bag_name
        return wrapper

class SavePoseControlHook(object):
    def __init__(self,
                 base_path):
        self.base_path = base_path
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)

    def __call__(self,
                 func: Callable[..., Tuple[np.array, ...]]) -> \
            Callable[..., Tuple[np.array, ...]]:
        @wraps(func)
        def wrapper(*args, **kwargs):
            data, cnt = func(*args, **kwargs)
            bag_name = data['bag_name']
            vel = data['velocity']
            vel_reverse = np.flip(-vel, 0)
            pd.DataFrame(vel).to_csv(os.path.join(self.base_path, bag_name.split('.')[0]
                                                  + '_forward_vel.csv'), index=False, header=False)
            pd.DataFrame(vel_reverse).to_csv(os.path.join(self.base_path, bag_name.split('.')[0]
                                                  + '_backward_vel.csv'), index=False, header=False)
            return data, cnt
        return wrapper
