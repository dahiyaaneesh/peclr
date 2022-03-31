"""
NOTE: This script contains some code taken from the FreiHAND directory:
https://github.com/lmb-freiburg/freihand
"""

from __future__ import print_function, unicode_literals
import numpy as np
import json
import os
import skimage.io as io
import cv2 as cv
import torch


""" General util functions. """


def _assert_exist(p):
    msg = "File does not exists: %s" % p
    assert os.path.exists(p), msg


def json_load(p):
    _assert_exist(p)
    with open(p, "r") as fi:
        d = json.load(fi)
    return d


""" Dataset related functions. """


def db_size(set_name):
    """ Hardcoded size of the datasets. """
    if set_name == "training":
        return 32560  # number of unique samples (they exists in multiple 'versions')
    elif set_name == "evaluation":
        return 3960
    else:
        assert 0, "Invalid choice."


class sample_version:
    gs = "gs"  # green screen
    hom = "hom"  # homogenized
    sample = "sample"  # auto colorization with sample points
    auto = (
        "auto"
    )  # auto colorization without sample points: automatic color hallucination

    db_size = db_size("training")

    @classmethod
    def valid_options(cls):
        return [cls.gs, cls.hom, cls.sample, cls.auto]

    @classmethod
    def check_valid(cls, version):
        msg = 'Invalid choice: "%s" (must be in %s)' % (version, cls.valid_options())
        assert version in cls.valid_options(), msg

    @classmethod
    def map_id(cls, id, version):
        cls.check_valid(version)
        return id + cls.db_size * cls.valid_options().index(version)


def read_img(idx, base_path, set_name, version=None):
    if version is None:
        version = sample_version.gs

    if set_name == "evaluation":
        assert (
            version == sample_version.gs
        ), "This the only valid choice for samples from the evaluation split."

    img_rgb_path = os.path.join(
        base_path, set_name, "rgb", "%08d.jpg" % sample_version.map_id(idx, version)
    )
    _assert_exist(img_rgb_path)
    return io.imread(img_rgb_path)


def get_bbox_from_pose(pose):
    assert pose is not None
    x = pose[:, 0]
    x = x[~np.isnan(x)]
    y = pose[:, 1]
    y = y[~np.isnan(y)]
    assert len(x) != 0 and len(y) != 0
    x1 = int(np.min(x))
    y1 = int(np.min(y))
    x2 = int(np.max(x))
    y2 = int(np.max(y))
    person_box = np.array([x1, y1, x2, y2])
    return person_box

def affine_transform_mat(crop_pos, output_size_x, output_size_y, scales):

    shift_to_upper_left = np.identity(3)
    shift_to_center = np.identity(3)

    a = scales[0]

    t = np.identity(3)

    t[0][0] = a
    t[0][1] = 0
    t[1][0] = 0
    t[1][1] = a

    shift_to_upper_left[0][2] = -crop_pos[0]
    shift_to_upper_left[1][2] = -crop_pos[1]
    shift_to_center[0][2] = output_size_x / 2
    shift_to_center[1][2] = output_size_y / 2
    t_form = np.matmul(t, shift_to_upper_left)
    t_form = np.matmul(shift_to_center, t_form)

    return t_form

# def affine_transform_mat(param, crop_pos, output_size_x, output_size_y, scales):
    # # Credit to Umer Rafi

    # shift_to_upper_left = np.identity(3)
    # shift_to_center = np.identity(3)

    # a = scales[0] * param["scale"] * np.cos(param["rot"])
    # b = scales[1] * param["scale"] * np.sin(param["rot"])

    # t = np.identity(3)

    # t[0][0] = a
    # t[0][1] = -b
    # t[1][0] = b
    # t[1][1] = a

    # shift_to_upper_left[0][2] = -crop_pos[0] + param["tx"]
    # shift_to_upper_left[1][2] = -crop_pos[1] + param["ty"]
    # shift_to_center[0][2] = output_size_x / 2
    # shift_to_center[1][2] = output_size_y / 2
    # t_form = np.matmul(t, shift_to_upper_left)
    # t_form = np.matmul(shift_to_center, t_form)

    # return t_form


class HandJoints:
    def __init__(self):
        # number of joints
        self.count = 21
        # indexes of joints
        self.wrist = 0
        self.thumb_mcp = 1
        self.index_mcp = 2
        self.middle_mcp = 3
        self.ring_mcp = 4
        self.pinky_mcp = 5
        self.thumb_pip = 6
        self.index_pip = 7
        self.middle_pip = 8
        self.ring_pip = 9
        self.pinky_pip = 10
        self.thumb_dip = 11
        self.index_dip = 12
        self.middle_dip = 13
        self.ring_dip = 14
        self.pinky_dip = 15
        self.thumb_tip = 16
        self.index_tip = 17
        self.middle_tip = 18
        self.ring_tip = 19
        self.pinky_tip = 20


def convert_order(kp3d):
    output = np.zeros(shape=kp3d.shape, dtype=kp3d.dtype)

    output[0] = kp3d[HandJoints().wrist]

    output[1] = kp3d[HandJoints().thumb_mcp]
    output[2] = kp3d[HandJoints().thumb_pip]
    output[3] = kp3d[HandJoints().thumb_dip]
    output[4] = kp3d[HandJoints().thumb_tip]

    output[5] = kp3d[HandJoints().index_mcp]
    output[6] = kp3d[HandJoints().index_pip]
    output[7] = kp3d[HandJoints().index_dip]
    output[8] = kp3d[HandJoints().index_tip]

    output[9] = kp3d[HandJoints().middle_mcp]
    output[10] = kp3d[HandJoints().middle_pip]
    output[11] = kp3d[HandJoints().middle_dip]
    output[12] = kp3d[HandJoints().middle_tip]

    output[13] = kp3d[HandJoints().ring_mcp]
    output[14] = kp3d[HandJoints().ring_pip]
    output[15] = kp3d[HandJoints().ring_dip]
    output[16] = kp3d[HandJoints().ring_tip]

    output[17] = kp3d[HandJoints().pinky_mcp]
    output[18] = kp3d[HandJoints().pinky_pip]
    output[19] = kp3d[HandJoints().pinky_dip]
    output[20] = kp3d[HandJoints().pinky_tip]

    return output


def move_palm_to_wrist(kp3d):
    palm = kp3d[0]
    middle_mcp = kp3d[3]
    wrist = 2 * palm - middle_mcp
    kp3d[0] = wrist

    return kp3d


def modify_bbox(bbox, scale):
    c_x = (bbox[0] + bbox[2]) / 2
    c_y = (bbox[1] + bbox[3]) / 2
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    width *= scale
    height *= scale
    length = max(height, width)
    bbox[0] = c_x - length / 2
    bbox[1] = c_y - length / 2
    bbox[2] = c_x + length / 2
    bbox[3] = c_y + length / 2

    return bbox


def preprocess(img, K, T, crop_size):
    image_mean = np.array([0.485, 0.456, 0.406])
    image_std = np.array([0.229, 0.224, 0.225])
    # Crop image
    img = cv.warpAffine(
        img,
        T[0:2],
        (crop_size, crop_size),
        borderMode=cv.BORDER_CONSTANT,
        borderValue=[0.485, 0.456, 0.406],
    )
    img = img.astype(np.float32) / 255
    img = np.divide((img - image_mean), image_std)
    img = torch.from_numpy(img.transpose(2, 0, 1)).view(1, 3, crop_size, crop_size)
    # Adjust K
    K = torch.from_numpy(np.matmul(T, K).reshape(1, 3, 3))

    return img, K


def create_affine_transform_from_bbox(bbox, crop_size):
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    l = float(max(height, width))
    # Create the transformation matrix
    target_dist = 0.7
    # param = {"rot": 0, "scale": 1, "tx": 0, "ty": 0}  # scale,
    center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    scales = [target_dist * crop_size / l, target_dist * crop_size / l]
    T = affine_transform_mat(center, crop_size, crop_size, scales)

    return T
