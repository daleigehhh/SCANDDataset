import os
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from torchvision.transforms.functional import vflip, hflip
import pickle
from typing import Callable, Any, Dict, List, Tuple, Iterable, Optional
import numpy as np
from omegaconf import OmegaConf


class RandomVFlipAction(object):
    def __init__(self, prob: float=.5) -> None:
        self.prob = prob

    def __call__(self, video: torch.tensor, control: torch.tensor) \
            -> Tuple[torch.tensor, torch.tensor]:
        if torch.rand(1) < self.prob:
            video = vflip(video)
            control[..., 0].mul_(-1)
        return video, control


class RandomHFlipAction(object):
    def __init__(self, prob: float=.5) -> None:
        self.prob = prob

    def __call__(self, video: torch.tensor, control: torch.tensor) \
            -> Tuple[torch.tensor, torch.tensor]:
        if torch.rand(1) < self.prob:
            video = hflip(video)
            control[..., -1].mul_(-1)
        return video, control

class RandomRotate(object):
    def __init__(self, degrees: Iterable[float], prob: float=.5):
        self.prob = prob
        self.operation = transforms.RandomRotation(degrees)

    def __call__(self, video: torch.tensor, control: torch.tensor) \
            -> Tuple[torch.tensor, torch.tensor]:
        if torch.rand(1) < self.prob:
            video = self.operation(video)
        return video, control

class DataAugmentation(object):
    def __init__(self,
                 config_path: str) -> None:
        self.aug_list = []
        config = OmegaConf.load(config_path)
        try:
            rotate_config= config['rotation']
        except KeyError:
            rotate_config = None
        try:
            vflip_config = config['vflip']
        except KeyError:
            vflip_config = None
        try:
            hflip_config = config['hflip']
        except KeyError:
            hflip_config = None
        if rotate_config is not None:
            rotate_aug = RandomRotate(**rotate_config)
            self.aug_list.append(rotate_aug)
        if vflip_config is not None:
            vflip_aug = RandomVFlipAction(**vflip_config)
            self.aug_list.append(vflip_aug)
        if hflip_config is not None:
            hflip_aug = RandomHFlipAction(**hflip_config)
            self.aug_list.append(hflip_aug)

    def __len__(self):
        return len(self.aug_list)

    def __getitem__(self, idx):
        return self.aug_list[idx]

    def __setitem__(self, idx, value):
        self.aug_list[idx] = value

    def __call__(self, maps, velocity):
        try:
            for aug in self.aug_list:
                maps, velocity = aug(maps, velocity)
        except Exception:
            raise RuntimeError("Augmentation inconsistent")
        return maps, velocity


class SCADOGMSubDataset(Dataset):
    def __init__(self,
                 hist_seq_len: int,
                 pred_seq_len: int,
                 data: Dict[str, np.array],
                 backward_aug = False,
                 augmentation: Optional[Callable[..., Tuple[torch.tensor, ...]]] = None
                 ) -> None:
        self.hist_seq_len = hist_seq_len
        self.pred_seq_len = pred_seq_len
        self.seq_len = hist_seq_len + pred_seq_len
        self.data = data
        self.bag_name = data['bag_name']
        if backward_aug:
            maps_numpy = np.concatenate([data['maps'], data['reversed_maps']], 0)
            velocity_numpy = np.concatenate([data['velocity'], data['reversed_velocity']], 0)
            self.maps = torch.from_numpy(maps_numpy)
            self.velocity = torch.from_numpy(velocity_numpy)
            self.length = len(self.maps) - self.seq_len + 1
        else:
            self.maps = torch.from_numpy(data['maps'])
            self.velocity = torch.from_numpy(data['velocity'])
            self.length = len(data['maps']) - self.seq_len + 1
        if not self._check_consistency():
            raise RuntimeError(f"Inconsistent dataset when checking data from {self.bag_name}")
        self.augmentation = augmentation

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx) -> Dict[str, torch.tensor]:
        # For now, [b, t, c, h, w] or [N, L, C, H, W]
        maps = self.maps[idx: idx+self.seq_len].unsqueeze(1)
        velocity = self.velocity[idx: idx+self.seq_len]
        maps, velocity = self.augmentation(maps, velocity)
        out = {
            'maps': maps,
            'velocity': velocity
        }
        return out

    def _check_consistency(self) -> bool:
        return len(self.data['maps']) == len(self.data['velocity']) and \
               len(self.data['maps']) == len(self.data['pose']) and \
               self.length > 0


class SCANDOGMDataset(ConcatDataset):
    def __init__(self,
                 hist_seq_len: int,
                 pred_seq_len: int,
                 blob_path_list: List[str],
                 backward_aug = False,
                 augmentation: Optional[Callable[..., Tuple[torch.tensor, ...]]] = None) -> None:
        data_list = []
        for blob_path in blob_path_list:
            with open(blob_path, 'rb') as f:
                try:
                    data_list.extend(pickle.load(f))
                except Exception as e:
                    print(f"load blob file {blob_path.split('/')[-1]} failed. "
                          f"as {e}")
        sub_datasets = [SCADOGMSubDataset(hist_seq_len, pred_seq_len, data_bag, backward_aug, augmentation)
                   for data_bag in data_list]
        super().__init__(sub_datasets)

if __name__ == "__main__":
    d_aug = DataAugmentation('../configs/data_aug.yaml')

    blob_path_list = os.listdir('../blobs')
    blob_path_list = [os.path.join('../blobs', i) for i in blob_path_list]

    import time

    start_time = time.time()
    scand = SCANDOGMDataset(hist_seq_len=10,
                            pred_seq_len=30,
                            blob_path_list=blob_path_list,
                            backward_aug=True,
                            augmentation=d_aug)
    dloader = DataLoader(scand, batch_size=32, shuffle=True, num_workers=8)

    for i in dloader:
        # print(i.keys())
        # print(i['maps'].shape)
        pass
    end_time = time.time()

    print(f'length of the SCAND datasets {len(scand)}, {end_time - start_time} sec elapsed')
