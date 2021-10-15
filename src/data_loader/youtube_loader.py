import os


from typing import Tuple

import cv2
import torch
import numpy as np
from src.data_loader.utils import get_joints_from_mano_mesh
from src.utils import read_json, save_json
from torch.utils.data import Dataset
from tqdm import tqdm
from src.data_loader.joints import Joints
from src.constants import MANO_MAT
import pandas as pd


class YTB_DB(Dataset):
    """Class to load samples from the youtube dataset.
    Inherits from the Dataset class in  torch.utils.data.
    Not be used for supervised learning!!
    Camera matrix is unity to fit with the sample augmenter.
    """

    def __init__(self, root_dir: str, split: str = "train"):
        self.root_dir = root_dir
        self.split = split
        self.joints_list, self.img_list = self.get_joints_labels_and_images()
        self.img_dict = {item["id"]: item for item in self.img_list}
        self.joints = Joints()
        self.create_ytb_valid_invalid_csv()
        self.indices = self.create_train_val_split()

    def get_joints_labels_and_images(self) -> Tuple[dict, dict]:
        """Returns the dictionary conatinign the bound box of the image and dictionary
        containig image information.

        Returns:
            Tuple[dict, dict]: joints, image_dict
                image_dict
                    - `name` - Image name in the form
                        of `youtube/VIDEO_ID/video/frames/FRAME_ID.png`.
                    - `width` - Width of the image.
                    - `height` - Height of the image.
                    - `id` - Image ID.
                joints
                    - `joints` - 21 joints, containing bound box limits as vertices.
                    - `is_left` - Binary value indicating a right/left hand side.
                    - `image_id` - ID to the corresponding entry in `images`.
                    - `id` - Annotation ID (an image can contain multiple hands).
        """
        data_json_path = os.path.join(self.root_dir, f"youtube_{self.split}.json")
        joints_path = os.path.join(self.root_dir, f"youtube_{self.split}_joints.json")
        images_json_path = os.path.join(
            self.root_dir, f"youtube_{self.split}_images.json"
        )
        if os.path.exists(joints_path) and os.path.exists(images_json_path):
            return read_json(joints_path), read_json(images_json_path)
        else:
            print(
                "JSONs containing condensed keypoints and images id missing!. Creating them."
            )
            data_json = read_json(data_json_path)
            images_dict = data_json["images"]
            save_json(images_dict, images_json_path)
            annotations_dict = data_json["annotations"]
            joints = self.get_joints_from_annotations(annotations_dict)
            save_json(joints, joints_path)
            return joints, images_dict

    def get_joints_from_annotations(self, annotations: dict) -> dict:
        """Converts vertices corresponding to mano mesh to 21 coordinates signifying
        the bound box.

        Args:
            annotations (dict): dictionary containing annotations.

        Returns:
            dict: same dictionary as annotations except 'vertices' is removed and
                'joints' key is added.
        """
        optimized_vertices = []
        mano_matrix = torch.load(MANO_MAT)
        for elem in tqdm(annotations):
            # joints_21 = sudo_joint_bound(elem["vertices"])
            joints_21 = get_joints_from_mano_mesh(
                torch.tensor(elem["vertices"]), mano_matrix
            )
            optimized_vertices.append(
                {
                    **{key: val for key, val in elem.items() if key != "vertices"},
                    **{"joints": joints_21.tolist()},
                }
            )
        return optimized_vertices

    def create_train_val_split(self) -> np.array:
        """Creates split for train and val data in mpii
        Raises:
            NotImplementedError: In case the split doesn't match test, train or val.

        Returns:
            np.array: array of indices
        """
        if self.split == "train":
            valid_index_df = pd.read_csv(
                os.path.join(self.root_dir, f"youtube_{self.split}_invalid_index.csv")
            )
            return valid_index_df[valid_index_df.valid]["joint_idx"].values
        elif self.split == "val":
            valid_index_df = pd.read_csv(
                os.path.join(self.root_dir, f"youtube_{self.split}_invalid_index.csv")
            )
            return valid_index_df[valid_index_df.valid]["joint_idx"].values
        elif self.split == "test":
            valid_index_df = pd.read_csv(
                os.path.join(self.root_dir, f"youtube_{self.split}_invalid_index.csv")
            )
            return valid_index_df[valid_index_df.valid]["joint_idx"].values
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.indices)

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
        img_name = os.path.join(
            self.root_dir, self.img_dict[self.joints_list[idx_]["image_id"]]["name"]
        )
        img = cv2.cvtColor(
            cv2.imread(img_name.replace(".png", ".jpg")), cv2.COLOR_BGR2RGB
        )
        joints3D = self.joints.mano_to_ait(
            torch.tensor(self.joints_list[idx_]["joints"]).float()
        )
        if self.joints_list[idx_]["is_left"] == 1:
            # flipping horizontally to make it right hand
            img = cv2.flip(img, 1)
            # width - x coord
            joints3D[:, 0] = img.shape[1] - joints3D[:, 0]
        joints_raw = joints3D.clone()
        # joints3D = torch.tensor(self.bbox[idx_]["joints"]).float()

        # because image is cropped and rotated with the 2d projections of these coordinates.
        # It needs to have depth as 1.0 to not cause problems. For procrustes use "joints_raw"
        joints3D[..., -1] = 1.0
        camera_param = torch.eye(3).float()
        joints_valid = torch.zeros_like(joints3D[..., -1:])
        sample = {
            "image": img,
            "K": camera_param,
            "joints3D": joints3D,
            "joints_valid": joints_valid,
            "joints_raw": joints_raw,
        }
        return sample

    def create_ytb_valid_invalid_csv(self):
        """
        Iterates through the YTB data to check which images are available.
        It will prepare a csv file for selected split which will contain validity flags for
        all joints based on availaibilty of the images.
        """
        joint_idx = []
        valid = []
        image = []
        counter = 0
        for idx_ in tqdm(range(len(self.joints_list))):
            img_name = os.path.join(
                self.root_dir, self.img_dict[self.joints_list[idx_]["image_id"]]["name"]
            ).replace(".png", ".jpg")
            image.append(self.img_dict[self.joints_list[idx_]["image_id"]]["name"])
            joint_idx.append(idx_)
            if os.path.isfile(img_name):
                valid.append(True)
            else:
                counter += 1
                valid.append(False)
        print(
            f"{counter} out of {len(self.joints_list)} samples not found in {self.split} set"
        )
        df = pd.DataFrame({"joint_idx": joint_idx, "valid": valid, "image": image})
        df.to_csv(
            os.path.join(self.root_dir, f"youtube_{self.split}_invalid_index.csv")
        )
