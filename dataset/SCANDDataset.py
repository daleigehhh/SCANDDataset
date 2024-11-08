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
                 augmentation: Optional[Callable[..., Tuple[torch.tensor, ...]]] = None
                 ) -> None:
        self.hist_seq_len = hist_seq_len
        self.pred_seq_len = pred_seq_len
        self.seq_len = hist_seq_len + pred_seq_len
        self.data = data
        self.bag_name = data['bag_name']
        self.length = len(data['maps']) - (self.hist_seq_len + self.pred_seq_len) + 1
        if not self._check_consistency():
            raise RuntimeError(f"Inconsistent dataset when checking {self.bag_name}")
        self.augmentation = augmentation

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx) -> Dict[str, torch.tensor]:
        maps = self.data['maps'][idx:idx+self.seq_len]
        velocity = self.data['velocity'][idx:idx+self.seq_len]
        # For now, [b, t, c, h, w] or [N, L, C, H, W]
        maps = torch.from_numpy(maps).unsqueeze(1)
        velocity = torch.from_numpy(velocity)
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
                 blob_path: str,
                 augmentation: Optional[Callable[..., Tuple[torch.tensor, ...]]] = None) -> None:
        with open(blob_path, 'rb') as f:
            data = pickle.load(f)
        sub_datasets = [SCADOGMSubDataset(hist_seq_len, pred_seq_len, data_bag, augmentation)
                   for data_bag in data]
        super().__init__(sub_datasets)


if __name__ == "__main__":
    d_aug = DataAugmentation('../configs/data_aug.yaml')

    scand = SCANDOGMDataset(hist_seq_len=10,
                            pred_seq_len=30,
                            blob_path='../blobs/data.pkl',
                            augmentation=d_aug)
    print(f'length of the SCAND datasets')
