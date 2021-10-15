import os

from torch.tensor import Tensor
from src.data_loader.utils import convert_2_5D_to_3D
from typing import List, Tuple

import cv2
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from src.data_loader.joints import Joints
from src.utils import read_json
from torch.utils.data import Dataset

BOUND_BOX_SCALE = 0.33


class F_DB(Dataset):
    """Class to load samples from the Freihand dataset.
    Inherits from the Dataset class in  torch.utils.data.
    Note: The keypoints are mapped to format used at AIT.
    Refer to joint_mapping.json in src/data_loader/utils.
    """

    def __init__(
        self, root_dir: str, split: str, seed: int = 5, train_ratio: float = 0.9
    ):
        """Initializes the freihand dataset class, relevant paths and the Joints
        class for remapping of freihand formatted joints to that of AIT.

        Args:
            root_dir (str): Path to the directory with image samples.
        """
        self.root_dir = root_dir
        self.split = split
        self.seed = seed
        self.train_ratio = train_ratio
        self.labels = self.get_labels()
        self.scale = self.get_scale()
        self.camera_param = self.get_camera_param()
        self.img_names, self.img_path = self.get_image_names()
        self.indices = self.create_train_val_split()
        # To convert from freihand to AIT format.
        self.joints = Joints()

    def create_train_val_split(self) -> np.array:
        """Creates split for train and val data in freihand

        Raises:
            NotImplementedError: In case the split doesn't match test, train or val.

        Returns:
            np.array: array of indices
        """
        num_unique_images = len(self.camera_param)
        train_indices, val_indices = train_test_split(
            np.arange(num_unique_images),
            train_size=self.train_ratio,
            random_state=self.seed,
        )
        if self.split == "train":
            train_indices = np.sort(train_indices)
            train_indices = np.concatenate(
                (
                    train_indices,
                    train_indices + num_unique_images,
                    train_indices + num_unique_images * 2,
                    train_indices + num_unique_images * 3,
                ),
                axis=0,
            )
            return train_indices
        elif self.split == "val":
            val_indices = np.sort(val_indices)
            val_indices = np.concatenate(
                (
                    val_indices,
                    val_indices + num_unique_images,
                    val_indices + num_unique_images * 2,
                    val_indices + num_unique_images * 3,
                ),
                axis=0,
            )
            return val_indices
        elif self.split == "test":
            return np.arange(len(self.camera_param))
        else:
            raise NotImplementedError

    def get_image_names(self) -> Tuple[List[str], str]:
        """Gets the name of all the files in root_dir.
        Make sure there are only image in that directory as it reads all the file names.

        Returns:
            List[str]: List of image names.
            str: base path for image directory
        """
        if self.split in ["train", "val"]:
            img_path = os.path.join(self.root_dir, "training", "rgb")
        else:
            img_path = os.path.join(self.root_dir, "evaluation", "rgb")
        img_names = next(os.walk(img_path))[2]
        img_names.sort()
        return img_names, img_path

    def get_labels(self) -> list:
        """Extacts the labels(joints coordinates) from the label_json at labels_path
        Returns:
            list: List of all the the coordinates(32650).
        """
        if self.split in ["train", "val"]:
            labels_path = os.path.join(self.root_dir, "training_xyz.json")
            return read_json(labels_path)
        else:
            return None

    def get_scale(self) -> list:
        """Extacts the scale from freihand data."""
        if self.split in ["train", "val"]:
            labels_path = os.path.join(self.root_dir, "training_scale.json")
        else:
            labels_path = os.path.join(self.root_dir, "evaluation_scale.json")
        return read_json(labels_path)

    def get_camera_param(self) -> list:
        """Extacts the camera parameters from the camera_param_json at camera_param_path.
        Returns:
            list: List of camera paramters for all images(32650)
        """
        if self.split in ["train", "val"]:
            camera_param_path = os.path.join(self.root_dir, "training_K.json")
        else:
            camera_param_path = os.path.join(self.root_dir, "evaluation_K.json")
        return read_json(camera_param_path)

    def __len__(self):
        return len(self.indices)

    def create_sudo_bound_box(self, scale) -> Tensor:
        max_bound = torch.tensor([224.0, 224.0])
        min_bound = torch.tensor([0.0, 0.0])
        c = (max_bound + min_bound) / 2.0
        s = ((max_bound - min_bound) / 2.0) * scale
        bound_box = torch.tensor(
            [[0, 0, 0]]
            + [[s[0], s[1], 1]] * 5
            + [[-s[0], s[1], 1]] * 5
            + [[s[0], -s[1], 1]] * 5
            + [[-s[0], -s[1], 1]] * 5
        ) + torch.tensor([c[0], c[1], 0])
        return bound_box.float()

    def __getitem__(self, idx: int) -> dict:
        """Returns a sample corresponding to the index.

        Args:
            idx (int): index

        Returns:
            dict: item with following elements.
                "image" in opencv bgr format.
                "K": camera params
                "joints3D": 3D coordinates of joints in AIT format.
        """

        if torch.is_tensor(idx):
            idx = idx.tolist()
        idx_ = self.indices[idx]
        img_name = os.path.join(self.img_path, self.img_names[idx_])
        img = cv2.cvtColor(cv2.imread(img_name),cv2.COLOR_BGR2RGB)
        if self.labels is not None:
            camera_param = torch.tensor(self.camera_param[idx_ % 32560]).float()
            joints3D = self.joints.freihand_to_ait(
                torch.tensor(self.labels[idx_ % 32560]).float()
            )
        else:
            camera_param = torch.tensor(self.camera_param[idx_]).float()
            joints2d_orthogonal = self.create_sudo_bound_box(scale=BOUND_BOX_SCALE)
            joints3D = convert_2_5D_to_3D(
                joints2d_orthogonal, scale=1.0, K=camera_param.clone()
            )
        joints_valid = torch.ones_like(joints3D[..., -1:])
        sample = {
            "image": img,
            "K": camera_param,
            "joints3D": joints3D,
            "joints_valid": joints_valid,
        }
        return sample
