from scripts.extractor import ROSBagExtractor
from dataset.SCANDDataset import SCANDOGMDataset, DataAugmentation
from torch.utils.data import DataLoader


if __name__ == "__main__":
    # extractor = ROSBagExtractor(base_path='bags/Jackal',
    #                             target='blobs',
    #                             dump_name='Jacal_reverse_aug',
    #                             config='configs/Jackal.yaml')
    # extractor.extract_all()

    # extractor = ROSBagExtractor(base_path='bags/Spot',
    #                             target='blobs',
    #                             dump_name='Spot_reverse_aug',
    #                             config='configs/Spot.yaml')
    # extractor.extract_all()

    data_aug = DataAugmentation(config_path='configs/data_aug.yaml')

    dset = SCANDOGMDataset(hist_seq_len=10,
                           pred_seq_len=10,
                           blob_path='blobs/Spot_reverse_aug.pkl',
                           augmentation=data_aug)

    dloader = DataLoader(dset, batch_size=32, drop_last=True)

    print(len(dloader))

    for i in dloader:
        print(i.keys())
        print(i['maps'].shape)
        print(i['velocity'].shape)
        break