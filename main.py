from scripts.extractor import ROSBagExtractor
from dataset.SCANDDataset import SCANDOGMDataset, DataAugmentation
from torch.utils.data import DataLoader


if __name__ == "__main__":
    extractor = ROSBagExtractor(base_path='bags',
                                target='blobs',
                                config='configs/Jackal.yaml',
                                )
    extractor.extract_all()

    data_aug = DataAugmentation(config_path='configs/data_aug.yaml')

    dset = SCANDOGMDataset(hist_seq_len=10,
                           pred_seq_len=10,
                           blob_path='blobs/data.pkl',
                           augmentation=data_aug)

    dloader = DataLoader(dset, batch_size=32, drop_last=True)

    for i in dloader:
        print(i.keys())
        print(i['maps'].shape)
        print(i['velocity'].shape)
        break