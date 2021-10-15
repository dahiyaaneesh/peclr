import copy
from math import cos, pi, sin
from typing import Tuple, Union

import numpy as np
import torch
from PIL import Image
from src.data_loader.joints import Joints
from src.types import CAMERA_PARAM, JOINTS_3D, JOINTS_25D, SCALE
from src.constants import MANO_MAT
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

JOINTS = Joints()
PARENT_JOINT = JOINTS.mapping.ait.wrist
CHILD_JOINT = JOINTS.mapping.ait.index_mcp


def convert_to_2_5D(K: CAMERA_PARAM, joints_3D: JOINTS_3D) -> Tuple[JOINTS_25D, SCALE]:
    """Converts coordinates from 3D to 2.5D
    Refer: https://arxiv.org/pdf/1804.09534.pdf

    Args:
        K (CAMERA_PARAM):3x3 Matrix with camera parameters.
        joints_3D (JOINTS_3D): Original 3D coordinates unscaled.

    Returns:
        Tuple[JOINTS_25D, SCALE]: 2.5 D coordinates and scale information.
    """
    scale = (((joints_3D[CHILD_JOINT] - joints_3D[PARENT_JOINT]) ** 2).sum()) ** 0.5
    joints_25D = ((K @ (joints_3D.T)).T) / joints_3D[:, -1:]
    joints_25D[:, -1] = (joints_3D[:, -1] - joints_3D[PARENT_JOINT, -1]) / scale
    return joints_25D, scale


def convert_2_5D_to_3D(
    joints_25D: JOINTS_25D,
    scale: SCALE,
    K: CAMERA_PARAM,
    is_batch: bool = False,
    Z_root_calc: torch.Tensor = None,
) -> JOINTS_3D:
    """Converts coordinates from 2.5 Dimesnions to original 3 Dimensions.
    Refer: https://arxiv.org/pdf/1804.09534.pdf

    Args:
        joints_25D (JOINTS_25D): 2.5 D coordinates.
        scale (SCALE): Eucledian distance between the parent and child joint.
        K (CAMERA_PARAM): 3x3 Matrix with camera parameters.

    Returns:
        JOINTS_3D: Obtained 3D coordinates from 2.5D coordinates and scale information.
    """

    Z_root, K_inv = get_root_depth(joints_25D, K, is_batch)
    Z_root = Z_root_calc if Z_root_calc is not None else Z_root
    camera_projection = joints_25D.clone()
    if is_batch:
        Z_coord = (joints_25D[:, :, -1:] + Z_root.view((-1, 1, 1))) * scale.view(
            (-1, 1, 1)
        )
        camera_projection[:, :, -1] = 1.0
        joints_3D = torch.bmm(camera_projection, torch.transpose(K_inv, 1, 2)) * Z_coord
    else:
        Z_coord = (joints_25D[:, -1:] + Z_root) * scale
        camera_projection[:, -1] = 1.0
        joints_3D = (camera_projection @ (K_inv.T)) * Z_coord
    return joints_3D


def get_root_depth(
    joints_25D: JOINTS_25D, K: CAMERA_PARAM, is_batch: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gets the scale normalized  Z_root from the joints coordinates using the result
    in https://arxiv.org/pdf/1804.09534.pdf equation 6 and 7.
    Note: There is a correction that needs to be made in the paper. x_n, y_n, x_m and y_m
    are the camera projections multiplued with inverted camera parameters.

    Args:
        joints_25D (JOINTS_25D): 21 joint coordinates in 2.5 Dimensions.
        K (CAMERA_PARAM): [description] : camera parameters of the data.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: scaled root Z coordinate and inverted camera parameters.
    """
    K_inv = torch.inverse(K)
    x_n, y_n, Z_n, x_m, y_m, Z_m, C = get_zroot_constraint_terms(
        joints_25D, K_inv, is_batch
    )

    a = (x_n - x_m) ** 2 + (y_n - y_m) ** 2
    b = 2 * (
        Z_n * (x_n ** 2 + y_n ** 2 - x_n * x_m - y_n * y_m)
        + Z_m * (x_m ** 2 + y_m ** 2 - x_n * x_m - y_n * y_m)
    )
    c = (
        (x_n * Z_n - x_m * Z_m) ** 2
        + (y_n * Z_n - y_m * Z_m) ** 2
        + (Z_n - Z_m) ** 2
        - C
    )
    # print("a={},b={},c={}".format(a, b, c))
    # print("x_n={}, y_n={}, Z_n={}".format(x_n, y_n, Z_n))
    # print("x_m={}, y_m={}, Z_m={}".format(x_m, y_m, Z_m))
    Z_root = (
        0.5
        * (-b + (torch.clamp((b ** 2 - 4 * a * c), min=1e-6) ** 0.5))
        / torch.clamp(a, min=1e-6)
    )
    return Z_root, K_inv


def error_in_conversion(true_joints_3D: JOINTS_3D, cal_joints_3D: JOINTS_3D) -> float:
    """Calculates absolutes error between original 3D coordinates and
     the ones recovered from 2.5 Dimensions.

    Args:
        true_joints_3D (JOINTS_3D): Original 3D coordinares from the data, unscaled
        cal_joints_3D (JOINTS_3D): Calculated 3D coordinares from the 2.5D coordinates, unscaled

    Returns:
        float: Maximum percentage error between the conversion and the original.
    """
    error = torch.abs(cal_joints_3D - true_joints_3D)
    # error = torch.sum((cal_joints_3D - true_joints_3D)**2, 0)**0.5
    return torch.max(error)


def get_rotation_matrix(angle) -> torch.Tensor:
    """Retursn 2D rotation matrix

    Args:
        angle (int): Angle in degrees. Measured counterclockwise from the x axis.

    Returns:
        torch.Tensor: A 2x2 Rotation tensor.
    """
    deg = pi / 180
    return torch.tensor(
        [[cos(angle * deg), -sin(angle * deg)], [sin(angle * deg), cos(angle * deg)]]
    )


def sample_rotator(
    image: Image.Image, joints: JOINTS_25D, angle: int
) -> Tuple[Image.Image, JOINTS_25D]:
    """Rotates the sample image and the 2D keypoints by 'angle' in degrees counter clockwise to x axis around
    the image center. The relative depth is not changed.


    Args:
        image (Image.Image): a PIL image, preferable uncropped
        joints (JOINTS_25D): Tensor of all 2.5 D coordinates.
        angle (int): Angle in degrees.

    Returns:
        Tuple[Image.Image, JOINTS_25D]: Rotated image and keypoints.
    """
    rot_mat = get_rotation_matrix(angle)
    joints_rotated = joints.clone()
    # centering joints at image center.
    joints_rotated[:, :-1] = joints_rotated[:, :-1] - image.size[0] / 2
    joints_rotated[:, :-1] = (rot_mat @ joints_rotated[:, :-1].T).T
    # reverting back to original origin i,e. top left corner.
    joints_rotated[:, :-1] = joints_rotated[:, :-1] + image.size[0] / 2
    # Rotate image by the same angle, make sure expand is set to False.
    # Also angle here is measured in clockwise direction make sure to add a minus sign.
    image_rotated = transforms.functional.rotate(image, -angle, expand=False)
    return image_rotated, joints_rotated


def sample_cropper(
    image: Image.Image,
    joints: JOINTS_25D,
    crop_margin: float = 1.5,
    crop_joints: bool = True,
) -> Tuple[Image.Image, JOINTS_25D]:
    top, left = torch.min(joints[:, 1]), torch.min(joints[:, 0])
    bottom, right = torch.max(joints[:, 1]), torch.max(joints[:, 0])
    height, width = bottom - top, right - left
    height = max(height, width)
    width = height
    origin_x = int(left - width * (crop_margin - 1) / 2)
    origin_y = int(top - height * (crop_margin - 1) / 2)
    joints_cropped = joints.clone()
    img_crop = transforms.functional.crop(
        image,
        top=origin_y,
        left=origin_x,
        height=int(height * crop_margin),
        width=int(width * crop_margin),
    )
    if crop_joints:
        joints_cropped[:, 0] = joints_cropped[:, 0] - origin_x
        joints_cropped[:, 1] = joints_cropped[:, 1] - origin_y
    return img_crop, joints_cropped


def sample_resizer(
    image: Image.Image,
    joints: JOINTS_25D,
    shape: Tuple = (128, 128),
    resize_joints: bool = True,
) -> Tuple[Image.Image, JOINTS_25D]:
    """Resizes the sample to given size.

    Args:
        image (Image.Image): A PIL image
        joints (JOINTS_25D): 2.5D joints. The depth is kept as is.
        shape (Tuple, optional): Size to which the image should be reshaped. Defaults to (128, 128).
        resize_joints (bool, optional): To resize the joints along with the image.. Defaults to True.

    Returns:
        Tuple[Image.Image, JOINTS_25D]: REsized image and keypoints.
    """
    width, height = image.size
    image = transforms.functional.resize(image, shape)
    joints_resized = joints.clone()
    if resize_joints:
        joints_resized[:, 0] = joints_resized[:, 0] * 128 / (width)
        joints_resized[:, 1] = joints_resized[:, 1] * 128 / (height)
    return image, joints_resized


def get_train_val_split(
    data: Union[Dataset, ConcatDataset], **kwargs
) -> Tuple[DataLoader, DataLoader]:
    """Creates validation and train dataloader from the Data_set object.

    Args:
        data (Data_Set): Object of the class Data_Set.
    kwargs:
        These arguments are passed as is to the pytorch DataLoader.

    Returns:
        Tuple[DataLoader, DataLoader]: train and validation data loader respectively.
    """

    if isinstance(data, ConcatDataset):
        val_datasets = []
        train_weights, val_weights = [], []
        for i in range(len(data.datasets)):
            train_weights += [1.0 / len(data.datasets[i])] * len(data.datasets[i])
            val_datasets.append(copy.copy(data.datasets[i]))
            val_datasets[-1].is_training(False)
            val_weights += [1.0 / len(val_datasets[-1])] * len(val_datasets[-1])
        val_data = ConcatDataset(val_datasets)
        train_weights = np.array(train_weights) / sum(train_weights)
        val_weights = np.array(val_weights) / sum(val_weights)
        return (
            DataLoader(
                data,
                sampler=WeightedRandomSampler(
                    weights=train_weights,
                    num_samples=len(train_weights),
                    replacement=True,
                ),
                **kwargs
            ),
            DataLoader(
                val_data,
                sampler=WeightedRandomSampler(
                    weights=val_weights, num_samples=len(val_weights), replacement=True
                ),
                **kwargs
            ),
        )
    else:
        data.is_training(True)
        val_data = copy.copy(data)
        val_data.is_training(False)
        return (
            DataLoader(data, **{**kwargs, "shuffle": True}),
            DataLoader(val_data, **{**kwargs, "shuffle": False}),
        )


def get_data(
    data_class, train_param, sources: list, experiment_type: str, split: str = "train"
):
    datasets = []
    sources = ["freihand"] if len(sources) == 0 else sources
    for source in sources:
        datasets.append(
            data_class(
                config=train_param,
                transform=transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize(
                            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                        ),
                    ]
                ),
                split=split,
                experiment_type=experiment_type,
                source=source,
            )
        )

    data = ConcatDataset(datasets)
    return data


def get_zroot_constraint_terms(
    joints_25D: JOINTS_25D, K_inv: torch.Tensor, is_batch: bool
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """returns parent ,child coordinates and C used for calculating in zroot

    Args:
        joints_25D (JOINTS_25D): [description]
        K_inv (torch.Tensor): [description]
        is_batch (bool): [description]

    Returns:
        Tuple[ torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor ]: [description]
    """
    if is_batch:
        batch_size = joints_25D.size()[0]
        joint_n = torch.cat(
            (
                joints_25D[:, PARENT_JOINT, :-1],
                torch.ones_like(joints_25D[:, PARENT_JOINT, -1:]),
            ),
            1,
        ).view(batch_size, 3, 1)
        joint_m = torch.cat(
            (
                joints_25D[:, CHILD_JOINT, :-1],
                torch.ones_like(joints_25D[:, CHILD_JOINT, -1:]),
            ),
            1,
        ).view(batch_size, 3, 1)
        xyz_n = torch.bmm(K_inv, joint_n).view(batch_size, 3)
        xyz_m = torch.bmm(K_inv, joint_m).view(batch_size, 3)
        x_n, y_n = xyz_n[:, 0], xyz_n[:, 1]
        Z_n = joints_25D[:, PARENT_JOINT, -1]
        x_m, y_m = xyz_m[:, 0], xyz_m[:, 1]
        Z_m = joints_25D[:, CHILD_JOINT, -1]
        C = torch.ones_like(x_n)
    else:
        x_n, y_n, _ = K_inv @ torch.cat(
            (joints_25D[PARENT_JOINT, :-1], torch.tensor([1.0])), 0
        )
        Z_n = joints_25D[PARENT_JOINT, -1]
        x_m, y_m, _ = K_inv @ torch.cat(
            (joints_25D[CHILD_JOINT, :-1], torch.tensor([1.0])), 0
        )
        Z_m = joints_25D[CHILD_JOINT, -1]
        C = 1
    return x_n, y_n, Z_n, x_m, y_m, Z_m, C


def sudo_joint_bound(vertices: np.array):
    """Calculates the bound box from the mano mesh vertices of youtube hand dataset.

    Args:
        vertices ([type]): [description]

    Returns:
        [type]: [description]
    """
    max_ver, min_ver = np.max(vertices, axis=0), np.min(vertices, axis=0)
    center_ver = (max_ver + min_ver) / 2
    return np.concatenate(
        (
            np.array([[max_ver[0], max_ver[1], max_ver[2]]] * 5),
            np.array([[min_ver[0], min_ver[1], min_ver[2]]] * 5),
            np.array([[min_ver[0], max_ver[1], min_ver[2]]] * 5),
            np.array([[max_ver[0], min_ver[1], max_ver[2]]] * 5),
            center_ver.reshape((1, -1)),
        )
    )


def get_joints_from_mano_mesh(
    mesh_vertices: torch.Tensor, mano_matrix: torch.Tensor
) -> torch.Tensor:
    """extracts joints from mano mesh. The matrix should be  from MANO_MAT path.
    The joint ordering is in joints_mapping.json.

    Args:
        mesh_vertices (torch.Tensor): mesh of shape 778 *3
        mano_matrix (torch.Tensor): Matrix of shape 16*778

    Returns:
        torch.Tensor: 21 hand joints in 3D.
    """
    joints = mano_matrix @ mesh_vertices
    tips = mesh_vertices[
        [744, 320, 443, 555, 672], :
    ]  # thumb, index, middle, ring, pinky
    joints = torch.cat([joints, tips], 0)
    return joints
