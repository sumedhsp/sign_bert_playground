import os
import re
from glob import glob

import torch
import numpy as np
import lightning.pytorch as pl
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from signbert.utils import read_json, read_txt_as_list


class WLASLDataModule(pl.LightningDataModule):
    """
    A PyTorch Lightning DataModule for the WLASL dataset.

    This class manages data loading and preprocessing for training, validation,
    and testing on the WLASL dataset.

    Attributes:
    - Various paths to data directories and files for WLASL.
    - VIDEO_ID_PATTERN: Regular expression pattern to extract video IDs.
    - PADDING_VALUE: The value used for padding sequences.
    """

    # Define dataset paths
    DPATH = 'sign_bert_playground/'

    RAW_VIDEO_PATH = os.path.join(DPATH, 'WLASL2000')
    PREPROCESS_DPATH = os.path.join(DPATH, 'preprocess')
    
    # The meta-data can be changed.
    META_DATA_FPATH = os.path.join(DPATH, 'finetune/WLASL/preprocess/nslt_100.json')

    CLASSES_JSON_FPATH = os.path.join(DPATH, 'finetune/WLASL/WLASL_class_list.txt')
    SKELETON_DATA_DPATH = os.path.join(DPATH, "wlasl_skeleton")
    
    MEANS_FPATH = os.path.join(PREPROCESS_DPATH, 'means.npy')
    STDS_FPATH = os.path.join(PREPROCESS_DPATH, 'stds.npy')
    VIDEO_ID_PATTERN = r"(?<=v=).{11}"
    PADDING_VALUE = 0.0

    def __init__(self, batch_size, normalize, meta_data_file_name):
        """
        Initialize the WLASLDataModule.

        Parameters:
        batch_size (int): Batch size for data loaders.
        normalize (bool): Whether to normalize the data.
        """
        super().__init__()
        self.batch_size = batch_size
        self.normalize = normalize

        # If provided, update the meta-data file
        if (meta_data_file_name):
            meta_data_path = os.path.join(self.DPATH, 'finetune/WLASL/preprocess')
            self.META_DATA_FPATH = os.path.join(meta_data_path, meta_data_file_name)

    def setup(self, stage):
        """
        Prepares the datasets for the given stage (either 'fit' or 'test').

        Parameters:
        stage (str): The stage for which to prepare the datasets.
        """
        
        if stage == "fit":
            # Loading from the nslt_{class} file. 
            # Modified the code to handle the data structure as above.
            video_info = read_json(WLASLDataModule.META_DATA_FPATH)
            train_info = [video for video in video_info if video_info[video]['subset'] == 'train']
            
            self.train_dataset = WLASLDataset(
                video_info,
                train_info,
                WLASLDataModule.SKELETON_DATA_DPATH,
                self.normalize,
                np.load(WLASLDataModule.MEANS_FPATH),
                np.load(WLASLDataModule.STDS_FPATH)
            )

            val_info = [video for video in video_info if video_info[video]['subset'] == 'val']
            
            self.val_dataset = WLASLDataset(
                video_info,
                val_info,
                WLASLDataModule.SKELETON_DATA_DPATH,
                self.normalize,
                np.load(WLASLDataModule.MEANS_FPATH),
                np.load(WLASLDataModule.STDS_FPATH)
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=my_collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=my_collate_fn
        )


class WLASLDataset(Dataset):
    """
    A PyTorch Dataset for handling data from the WLASL dataset.

    This class loads skeleton data for each sample, normalizes it if required, and
    extracts specific features like arms, left hand, and right hand keypoints.

    Attributes:
    - train_info (list): A list of dictionaries containing information about each sample.
    - skeleton_dpath (str): Path to the directory containing skeleton data files.
    - normalize (bool): Flag indicating whether the data should be normalized.
    - normalize_mean (numpy.ndarray or None): Mean values for normalization.
    - normalize_std (numpy.ndarray or None): Standard deviation values for normalization.
    """

    def __init__(self, video_info, train_info, skeleton_dpath, normalize, normalize_mean=None, normalize_std=None):
        """
        Initialize the WLASLDataset.

        Parameters:
        - train_info (list): Information about each training sample.
        - skeleton_dpath (str): Path to the directory with skeleton data files.
        - normalize (bool): Whether to normalize the data.
        - normalize_mean (numpy.ndarray, optional): Mean values for normalization.
        - normalize_std (numpy.ndarray, optional): Standard deviation values for normalization.
        """
        super().__init__()
        self.video_info = video_info
        self.train_info = train_info
        self.skeleton_dpath = skeleton_dpath
        self.normalize = normalize
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std

    def __len__(self):
        return len(self.train_info)

    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset.

        Parameters:
        - idx (int): Index of the sample.

        Returns:
        - dict: A dictionary containing the sample data.
        """
        video_id = self.train_info[idx]
        class_id = self.video_info[video_id]['action'][0]
        start_video = self.video_info[video_id]['action'][1]
        end_video = self.video_info[video_id]['action'][2]

        skeleton_video_fpath = os.path.join(self.skeleton_dpath, f"{video_id}.npy")
        skeleton_data = np.load(skeleton_video_fpath)[start_video:end_video]

        skeleton_data = skeleton_data[..., :2]
        if self.normalize:
            skeleton_data = (skeleton_data - self.normalize_mean) / self.normalize_std

        arms = skeleton_data[:, 5:11]
        lhand = skeleton_data[:, 91:112]
        rhand = skeleton_data[:, 112:133]

        return {
            "sample_id": idx,
            "class_id": class_id,
            "arms": arms,
            "lhand": lhand,
            "rhand": rhand
        }


def my_collate_fn(original_batch):
    """Custom collate function for WLASL data."""
    sample_id = []
    class_id = []
    arms = []
    lhand = []
    rhand = []
    for ob in original_batch:
        sample_id.append(ob["sample_id"])
        class_id.append(ob["class_id"])
        arms.append(torch.from_numpy(ob["arms"]))
        lhand.append(torch.from_numpy(ob["lhand"]))
        rhand.append(torch.from_numpy(ob["rhand"]))

    arms = pad_sequence(arms, batch_first=True, padding_value=WLASLDataModule.PADDING_VALUE)
    lhand = pad_sequence(lhand, batch_first=True, padding_value=WLASLDataModule.PADDING_VALUE)
    rhand = pad_sequence(rhand, batch_first=True, padding_value=WLASLDataModule.PADDING_VALUE)

    class_id = torch.tensor(class_id, dtype=torch.int64)
    sample_id = torch.tensor(sample_id, dtype=torch.int32)

    return {
        "sample_id": sample_id,
        "class_id": class_id,
        "arms": arms,
        "lhand": lhand,
        "rhand": rhand
    }
