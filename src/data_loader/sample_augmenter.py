from math import factorial
import random
from typing import Tuple

import cv2
import numpy as np
import torch
from easydict import EasyDict as edict
from src.types import JOINTS_25D


class SampleAugmenter:
    def __init__(self, augmentation_flags: edict, augmentation_params: edict):
        """Initialization of the sample augmentor class.

        Args:
            crop (bool): To crop the image around keypoints/joints
            resize (bool): To resize the image according to 'resize_shape' kwargs.
                Default is (128,128).
            rotate (bool): To rotate the image randomly
            seed (int): Seed for random rotation.
        """
        # Augmetation flags.
        self.set_augmentaion_flags(augmentation_flags)
        self.set_augmenation_params(augmentation_params)
        # random parameters.
        self.angle = None
        self.jitter = None
        self.jitter_x = None
        self.jitter_y = None
        self.sigma = None
        self.h = None
        self.s = None
        self.a = None
        self.b = None
        self._crop_margin_scale = 1.5
        # flags
        self._cut_out = False
        self._gaussian_blur = False
        self._rotate = False
        self._crop = False
        self._color_jitter = False
        self._color_drop = False
        self._gaussian_noise = False
        self._sobel_filter = False

    def transform_sample(
        self,
        image: np.array,
        joints: JOINTS_25D,
        override_angle: float = None,
        override_jitter: float = None,
    ) -> Tuple[np.array, JOINTS_25D, np.array]:
        """Transforms  the sample image and the 2D keypoints. The  relative depth is not
        changed.

        Args:
            image (np.array): An Image as a numpy array, preferable uncropped
            joints (JOINTS_25D): Tensor of all 2.5 D coordinates.
            override_angle (float): angle by which the sample should be rotated.
            will rotate irresective of the flag
        Returns:
            Tuple[np.array, JOINTS_25D, np.array]: Transformed image, keypoints and
                corresponding transformation matrix.
        """
        image_, joints_ = image.copy(), joints.clone()
        transformation_matrix = np.identity(3)
        # augmentations to be applied in beginning
        if self.sobel_filter and random.getrandbits(1):
            self._sobel_filter = True
            image_, _ = self.sobel_filter_sample(image_, None)
        else:
            self._sobel_filter = False
        if self.cut_out and random.getrandbits(1):
            self._cut_out = True
            image_, _ = self.cut_out_sample(image_, joints_)
        else:
            self._cut_out = False

        if self.gaussian_blur and random.getrandbits(1):
            self._gaussian_blur = True
            image_, _ = self.gaussian_blur_sample(image_, None)
        else:
            self._gaussian_blur = False

        if self.rotate or override_angle is not None:
            self._rotate = True
            image_, joints_, rot_mat = self.rotate_sample(
                image_, joints_, override_angle
            )
            transformation_matrix = np.concatenate((rot_mat, np.array([[0, 0, 1]])))
        else:
            self._rotate = False

        if self.crop or override_jitter is not None:
            self._crop = True
            image_, joints_, xy_shift = self.crop_sample(
                image_, joints_, override_jitter
            )
            transformation_matrix[0, -1] -= xy_shift[0]
            transformation_matrix[1, -1] -= xy_shift[1]
        else:
            self._crop = False
        # augmentations to be applied in the end.

        if self.resize:
            image_, joints_, factor = self.resize_sample(image_, joints_)
            transformation_matrix[0] = transformation_matrix[0] * factor[0]
            transformation_matrix[1] = transformation_matrix[1] * factor[1]

        if self.color_jitter:
            self._color_jitter = True
            image_, _ = self.color_jitter_sample(image_, None)
        else:
            self._color_jitter = False

        if self.gaussian_noise and random.getrandbits(1):
            self._gaussian_noise = True
            image_, _ = self.gaussian_noise_sample(image_, None)
        else:
            self._gaussian_noise = False

        if self.color_drop and random.getrandbits(1):
            self._color_drop = True
            image_, _ = self.color_drop_sample(image_, None)
        else:
            self._color_drop = False

        return image_, joints_, transformation_matrix

    def sobel_filter_sample(
        self, image: np.array, joints: JOINTS_25D
    ) -> Tuple[np.array, JOINTS_25D, tuple]:
        """Applies sobel filter along x axis and y axis. Converts the image to gray scale before applying the filter.
        Returns the added sobel filter.

        Args:
            image (np.array): Image as an numpy array.
            joints (JOINTS_25D): 2.5D joints. The depth is kept as is.

        Returns:
            Tuple[np.array JOINTS_25D]:  Image with sobel filter applied. The coordiniates are not changed.
        """
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=self.sobel_kernel)
        sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=self.sobel_kernel)
        # replicating same filter across all channels to keep the image channel consistent.
        image[:, :, :] = (sobel_x + sobel_y).reshape(list(image.shape[:2]) + [1])
        return image, joints

    def gaussian_noise_sample(
        self, image: np.array, joints: JOINTS_25D
    ) -> Tuple[np.array, JOINTS_25D, tuple]:
        """Applies gaussian noise to the sample.

        Args:
            image (np.array): Image as an numpy array.
            joints (JOINTS_25D): 2.5D joints. The depth is kept as is.

        Returns:
            Tuple[np.array JOINTS_25D]:  gaussian noise added image. The coordinates are not changed.
        """
        image += cv2.randn(np.zeros(image.shape, np.uint8), (0), (self.noise_std,) * 3)
        return image, joints

    def crop_sample(
        self, image: np.array, joints: JOINTS_25D, jitter: float = None
    ) -> Tuple[np.array, JOINTS_25D, tuple]:
        """Crops the sample around a crop box conataining all key points..

        Args:
            image (np.array): Image as an numpy array.
            joints (JOINTS_25D): 2.5D joints. The depth is kept as is.

        Returns:
            Tuple[np.array JOINTS_25D]: cropped image, adjusted keypoints and tuple of
                (x, y) corrdinates by which the image is movede around.
        """
        origin_x, origin_y, side = self.get_crop_size(joints, jitter)
        joints[:, 0] = joints[:, 0] - origin_x
        joints[:, 1] = joints[:, 1] - origin_y

        return (
            image[origin_y : origin_y + side, origin_x : origin_x + side, :],
            joints,
            (origin_x, origin_y),
        )

    def resize_sample(
        self, image: np.array, joints: JOINTS_25D
    ) -> Tuple[np.array, JOINTS_25D, tuple]:
        """Resizes the sample to given size.

        Args:
            image (np.array): Image as an numpy array.
            joints (JOINTS_25D): 2.5D joints. The depth is kept as is.

        Returns:
            Tuple[np.array JOINTS_25D, tuple]: Resized image, keypoints and tuple of
                factor by which the image is resized along width and height
                respectively.
        """
        height, width = image.shape[:2]
        factor_height = factor_width = 1

        try:
            image = cv2.resize(image, self.resize_shape, interpolation=cv2.INTER_AREA)
            factor_height = self.resize_shape[1] / height
            factor_width = self.resize_shape[0] / width
            joints[:, 0] = joints[:, 0] * factor_width
            joints[:, 1] = joints[:, 1] * factor_height
        except Exception as e:
            print(height, width, self.resize_shape)
            print(e)
        return image, joints, (factor_width, factor_height)

    def rotate_sample(
        self, image: np.array, joints: JOINTS_25D, angle: float = None
    ) -> Tuple[np.array, JOINTS_25D, np.array]:
        """Rotates the sample image and the 2D keypoints by a random angle about the
        crop box center with jitter 0 and crop_margin 1.5. The relative depth is not
        changed.

        Args:
            image (np.array): An Image as a numpy array, preferable uncropped
            joints (JOINTS_25D): Tensor of all 2.5 D coordinates.

        Returns:
            Tuple[np.array, JOINTS_25D, np.array]: Rotated image, keypoints and
                rotation matrix.
        """
        height, width = image.shape[:2]
        # rotating about crop box center with jitter zero and crop_margin as 0.0
        origin_x, origin_y, side = self.get_crop_size(
            joints, jitter=[0, 0], crop_margin=0.0
        )
        center = int(origin_x + side / 2), int(origin_y + side / 2)
        rot_mat = self.get_rotation_matrix(center=center, angle=angle)
        image = cv2.warpAffine(image, rot_mat, (width, height))
        # image[center[1]-2:center[1]+2, center[0]-2:center[0]+2, :] = 0
        joints_ = joints.clone()
        joints_[:, -1] = 1.0
        joints_ = joints_ @ rot_mat.T
        joints[:, :-1] = joints_
        return image, joints, rot_mat

    def color_drop_sample(
        self, image: np.array, joints: JOINTS_25D
    ) -> Tuple[np.array, JOINTS_25D]:
        """Drops the color.
        Joints are not affected by this transform.

        Args:
            image (np.array): Image as an numpy array.
            joints (JOINTS_25D): Not used, can be set as None.

        Returns:
            Tuple[np.array, JOINTS_25D]: Transfomed image and joints as is.
        """

        image[:, :, :] = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).reshape(
            list(image.shape[:2]) + [1]
        )
        return image, joints

    def color_jitter_sample(
        self, image: np.array, joints: JOINTS_25D
    ) -> Tuple[np.array, JOINTS_25D]:
        # randomly jittering the image.
        """Randomly jitters the sample image.
        Joints are not affected by this transform.

        Args:
            image (np.array): Image as an numpy array.
            joints (JOINTS_25D): Not used, can be set as None.

        Returns:
            Tuple[np.array, JOINTS_25D]: Transfomed image and joints as is.
        """

        h, s, a, b = self.get_random_color_jitter_factors()
        self.h = h
        self.s = s
        self.a = a
        self.b = b
        hue, saturation, value = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
        hue = np.clip(hue * h, 0, 255)
        saturation = np.clip(saturation * s, 0, 255)
        value = np.clip(value * a + b, 0, 255)
        image = cv2.cvtColor(
            cv2.merge([hue, saturation, value]).astype(np.uint8), cv2.COLOR_HSV2BGR
        )
        return image, joints

    def gaussian_blur_sample(
        self, image: np.array, joints: JOINTS_25D
    ) -> Tuple[np.array, JOINTS_25D]:
        """Randomly applies gaussian blurrinjg on the image.
        The kernel size and sigma are as per the SimCLR paper

        Args:
            image (np.array): Image as an numpy array.
            joints (JOINTS_25D): Not used, can be set as None.

        Returns:
            Tuple[np.array, JOINTS_25D]: Transfomed image and joints as is.
        """
        kernel_size = tuple(
            [
                i + 1 if i % 2 == 0 else i
                for i in (np.array(image.shape[:2]) * 0.1).astype(int)
            ]
        )
        sigma = random.uniform(0.1, 2.0)
        self.sigma = sigma
        image = cv2.GaussianBlur(image, kernel_size, sigma)
        return image, joints

    def cut_out_sample(
        self, image: np.array, joints: JOINTS_25D
    ) -> Tuple[np.array, JOINTS_25D]:
        """Randomly cuts out a rectangle from the image. The largest
        dimesion of the rectangle is 50% of the image dimesions.

        Args:
            joints (JOINTS_25D): Not used, can be set as None.

        Returns:
            Tuple[np.array, JOINTS_25D]: Transfomed image and joints as is.
        """
        # randomly dropping color
        if True:
            # randomly picking a hand coordiate to occlude.
            hand_center = np.random.randint(0, 20, 1)
            dim0_bounds, dim1_bounds = self.get_random_cut_out_box(
                image.shape[0],
                image.shape[1],
                joints[hand_center, 0],
                joints[hand_center, 1],
            )
            image[
                dim0_bounds[0] : dim0_bounds[1], dim1_bounds[0] : dim1_bounds[1]
            ] = np.uint8(np.random.randint(0, 255, 1))
        return image, joints

    def get_random_cut_out_box(
        self,
        image_dim0: int,
        image_dim1: int,
        hand_center_dim0: int,
        hand_center_dim1: int,
    ) -> Tuple[list, list]:
        """Helper function to obtain the bound box for cut out.

        Args:
            image_dim0 (int): Image's first dimension.
            image_dim1 (int): Image's second dimension.
            hand_center_dim0 (int): coordinate around which box should move.
            hand_center_dim1 (int): coordinate around which box should move.

        Returns:
            Tuple[list, list]: bounds along dim0 and dim1 respectively
        """
        ratio = random.uniform(self.cut_out_fraction[0], self.cut_out_fraction[1])
        cut_out_dim0 = int(image_dim0 * ratio)
        cut_out_dim1 = int(image_dim1 * ratio)
        top_corner_dim0 = int(
            random.uniform(
                hand_center_dim0 - cut_out_dim0 / 2, hand_center_dim0 - cut_out_dim0 / 2
            )
        )
        top_corner_dim1 = int(
            random.uniform(
                hand_center_dim1 - cut_out_dim1 / 2, hand_center_dim1 - cut_out_dim1 / 2
            )
        )
        return (
            np.clip([top_corner_dim0, top_corner_dim0 + cut_out_dim0], 0, image_dim0),
            np.clip([top_corner_dim1, top_corner_dim1 + cut_out_dim1], 0, image_dim1),
        )

    def get_random_crop_margin(self) -> float:
        """Gets random crop margin from the crop margin range

        Returns:
            float: random crop margin.
        """
        return random.uniform(self.crop_margin_range[0], self.crop_margin_range[1])

    def get_random_color_jitter_factors(self) -> Tuple[float, float, float, float]:
        """Gets the random color jitter factors.

        Returns:
            Tuple[float, float, float, float]: hue, saturation, alpha and beta factors.
        """
        hue_factor = random.uniform(*self.hue_factor_range)
        sat_factor = random.uniform(*self.sat_factor_range)
        alpha_factor = random.uniform(*self.value_factor_alpha_range)
        beta_factor = random.uniform(*self.value_factor_beta_range)
        return hue_factor, sat_factor, alpha_factor, beta_factor

    def get_rotation_matrix(self, center: Tuple[int, int], angle: float) -> np.array:
        """Function to get the roation matrix according to randomly sampled angle.

        Args:
            center (Tuple[int, int]): center coordinates (x, y)
            angle (float): angle by which the sample shoould be rotated.
                If none random value is chosen

        Returns:
            np.array: A 2 x 3 rotation matrix.
        """
        if angle is None:
            angle = random.uniform(self.min_angle, self.max_angle) // 1
        self.angle = angle
        return cv2.getRotationMatrix2D(center, angle, 1.0)

    def get_crop_size(
        self, joints: JOINTS_25D, jitter: float = None, crop_margin: float = None
    ) -> Tuple[int, int, int]:
        """Function to obtain the top left corner of the crop square and the side.

        Args:
            joints (JOINTS_25D): 2.5D joints Only 2D image coordinates are used.

        Returns:
            Tuple[int, int, int]:  Top left coordinates of the crop box and the side of
                the crop box.
        """
        if crop_margin is not None:
            crop_margin = crop_margin
        elif self.random_crop:
            crop_margin = self.get_random_crop_margin()
        else:
            crop_margin = self.crop_margin
        self._crop_margin_scale = crop_margin
        center_y, center_x = (
            int(torch.mean(joints[:, 1])),
            int(torch.mean(joints[:, 0])),
        )
        side = int(
            (
                torch.max(
                    ((joints[:, 1] - center_y) ** 2 + (joints[:, 0] - center_x) ** 2)
                )
            )
            ** 0.5
            * crop_margin
        )
        if jitter is None:
            jitter = [
                int(random.uniform(0, self.crop_box_jitter[1])),
                int(random.uniform(0, self.crop_box_jitter[1])),
            ]
            # jitter = [
            #     self.crop_box_jitter[0],
            #     self.crop_box_jitter[1]

            # ]
            # print(f"jitter {jitter}")
        self.jitter = jitter
        origin_x = max(center_x - side + jitter[0], 0)
        origin_y = max(center_y - side + jitter[1], 0)
        self.jitter_x = center_x - side - origin_x
        self.jitter_y = center_y - side - origin_y
        # print(f"{origin_x},{origin_y}, {side}, {center_x}, {center_y}, jitter {jitter}, {self.jitter_x}, {self.jitter_y}")
        # print(f" {center_x}, {center_y}, jitter {jitter}")
        return origin_x, origin_y, int(2 * side)

    def set_augmenation_params(self, augmentation_params: edict):
        """Helper method to set the augmentation params

        Args:
            augmentation_params (edict): Edict containing the augmentation params.
        """
        self.min_angle = augmentation_params.max_angle
        self.max_angle = augmentation_params.min_angle
        self.crop_margin_range = augmentation_params.crop_margin_range
        self.hue_factor_range = augmentation_params.hue_factor_range
        self.sat_factor_range = augmentation_params.sat_factor_range
        self.value_factor_alpha_range = augmentation_params.value_factor_alpha_range
        self.value_factor_beta_range = augmentation_params.value_factor_beta_range
        self.cut_out_fraction = augmentation_params.cut_out_fraction
        self.crop_margin = augmentation_params.crop_margin
        self.resize_shape = tuple(augmentation_params.resize_shape)
        self.crop_box_jitter = augmentation_params.crop_box_jitter
        self.sobel_kernel = augmentation_params.sobel_kernel
        self.noise_std = augmentation_params.noise_std

    def set_augmentaion_flags(self, augmentation_flags: edict):
        """Helper function to set the augmentation flags

        Args:
            augmentation_flags (edict): Edict containing the augmentation flags.
        """
        self.color_drop = augmentation_flags.color_drop
        self.color_jitter = augmentation_flags.color_jitter
        self.crop = augmentation_flags.crop
        self.resize = augmentation_flags.resize
        self.rotate = augmentation_flags.rotate
        self.gaussian_blur = augmentation_flags.gaussian_blur
        self.cut_out = augmentation_flags.cut_out
        self.random_crop = augmentation_flags.random_crop
        self.gaussian_noise = augmentation_flags.gaussian_noise
        self.sobel_filter = augmentation_flags.sobel_filter
