import torch
import torchvision
from easydict import EasyDict as edict
from src.constants import FREIHAND_DATA, YOUTUBE_DATA
from src.data_loader.freihand_loader import F_DB
from src.data_loader.sample_augmenter import SampleAugmenter
from src.data_loader.utils import convert_2_5D_to_3D, convert_to_2_5D, JOINTS
from src.data_loader.youtube_loader import YTB_DB
from torch.utils.data import Dataset


class Data_Set(Dataset):
    def __init__(
        self,
        config: edict,
        transform: torchvision.transforms,
        split: str = "train",
        experiment_type: str = "supervised",
        source: str = "freihand",
    ):
        """This class acts as overarching data_loader.
        It coordinates the indices that must go to the train and validation set.
        Note: To switch between train and validation switch the mode using ``is_training(True)`` for
        training and is_training(False) for validation.
        To create simulatenous instances of validation and training, make a shallow copy and change the
        mode with ``is_training()``

        See 01-Data_handler.ipynb for visualization.
        Args:
            config (e): Configuraction dict must have  "seed" and "train_ratio".
            transforms ([type]): torch transforms or composition of them.
            train_set (bool, optional): Flag denoting which samples are returned. Defaults to True.
            experiment_type (str, optional): Flag denoting how to decide what should be the sample format.
                Default is "supervised". For SimCLR change to "simclr"
        """
        # Individual data loader initialization.
        self.config = config
        self.source = source
        self.db = None
        self._split = split
        self.initialize_data_loaders()

        self.transform = transform
        self.experiment_type = experiment_type

        if self.experiment_type == "hybrid1":
            # Two augmenters are used when hybrid experiment data params are passed.
            self.pairwise_augmenter = self.get_sample_augmenter(
                config.augmentation_params, config.pairwise.augmentation_flags
            )
            self.contrastive_augmenter = self.get_sample_augmenter(
                config.augmentation_params, config.contrastive.augmentation_flags
            )
        else:
            self.augmenter = self.get_sample_augmenter(
                config.augmentation_params, config.augmentation_flags
            )

    def initialize_data_loaders(self):
        if self.source == "freihand":
            self.db = F_DB(
                root_dir=FREIHAND_DATA,
                split=self._split,
                train_ratio=self.config.train_ratio,
            )
        elif self.source == "youtube":
            # TODO: this data set has already existing validation set, hence no need for train ratio
            self.db = YTB_DB(root_dir=YOUTUBE_DATA, split=self._split)

    def __getitem__(self, idx: int):

        sample = self.db[idx]
        # Returning data as per the experiment.
        if self.experiment_type == "simclr":
            # sample = self.prepare_simclr_sample(sample, self.augmenter)
            sample = self.prepare_experiment4_pretraining(sample, self.augmenter)
        elif self.experiment_type == "experiment4_pretraining":
            # for simclr ablative, for nips A1
            sample = self.prepare_experiment4_pretraining(sample, self.augmenter)
        elif self.experiment_type == "hybrid2":
            sample = self.prepare_hybrid2_sample(sample, self.augmenter)
        else:
            sample = self.prepare_supervised_sample(sample, self.augmenter)
        return sample

    def __len__(self):
        return len(self.db)

    def get_sample_augmenter(
        self, augmentation_params: edict, augmentation_flags: edict
    ) -> SampleAugmenter:
        return SampleAugmenter(
            augmentation_params=augmentation_params,
            augmentation_flags=augmentation_flags,
        )

    def prepare_simclr_sample(self, sample: dict, augmenter: SampleAugmenter) -> dict:
        """Prepares sample according to SimCLR experiment.
        For each sample two transformations of an image are returned.
        Note: Rotation and jitter is kept same in both the transformations.
        Args:
            sample (dict): Underlying data from dataloader class.
            augmenter (SampleAugmenter): Augmenter used to transform sample

        Returns:
            dict: sample containing 'transformed_image1' and 'transformed_image2'
        """
        joints25D, _ = convert_to_2_5D(sample["K"], sample["joints3D"])
        img1, _, _ = augmenter.transform_sample(sample["image"], joints25D.clone())

        # To keep rotation and jitter consistent between the two transformations.
        override_angle = augmenter.angle
        overrride_jitter = augmenter.jitter

        img2, _, _ = augmenter.transform_sample(
            sample["image"], joints25D.clone(), override_angle, overrride_jitter
        )

        # Applying only image related transform
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return {"transformed_image1": img1, "transformed_image2": img2}

    def prepare_experiment4_pretraining(
        self, sample: dict, augmenter: SampleAugmenter
    ) -> dict:
        """Prepares samples for ablative studies on Simclr. This function isolates the
        effect of each transform. Make sure no other transformation is applied except
        the one you want to isolate. (Resize is allowed). Samples are not
        artificially increased by changing rotation and jitter for both samples.

        Args:
            sample (dict): Underlying data from dataloader class.
            augmenter (SampleAugmenter): Augmenter used to transform sample

        Returns:
            dict: sample containing 'transformed_image1' and 'transformed_image2'
        """

        joints25D, _ = convert_to_2_5D(sample["K"], sample["joints3D"])
        if augmenter.crop:
            override_jitter = None
        else:
            # Zero jitter is added incase the cropping is off. It is done to trigger the
            # cropping but always with no translation in image.
            override_jitter = [0, 0]
        if augmenter.rotate:
            override_angle = None
        else:
            override_angle = None
            # override_angle = random.uniform(1, 360)
            # uncomment line above to add this rotation  to both channels

        img1, _, _ = augmenter.transform_sample(
            sample["image"], joints25D.clone(), override_angle, override_jitter
        )
        img2, _, _ = augmenter.transform_sample(
            sample["image"], joints25D.clone(), override_angle, override_jitter
        )

        # Applying only image related transform
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return {"transformed_image1": img1, "transformed_image2": img2}

    def prepare_pairwise_sample(self, sample: dict, augmenter: SampleAugmenter) -> dict:
        """Prepares samples according to pairwise experiment, i.e. transforming the
        image and keepinf track of the relative parameters.
        Note: Gaussian blur and Flip are treated as boolean. Also it was decided not to
        use them for experiment.
        The effects of transformations are isolated.

        Args:
            sample (dict): Underlying data from dataloader class.
            augmenter (SampleAugmenter): Augmenter used to transform sample

        Returns:
            dict: sample containing following elements
                'transformed_image1'
                'transformed_image2'
                'joints1' (2.5D joints)
                'joints2' (2.5D joints)
                'rotation'
                'jitter' ...
        """
        joints25D, _ = convert_to_2_5D(sample["K"], sample["joints3D"])

        img1, joints1, _ = augmenter.transform_sample(
            sample["image"], joints25D.clone()
        )
        param1 = self.get_random_augment_param(augmenter)

        img2, joints2, _ = augmenter.transform_sample(
            sample["image"], joints25D.clone()
        )
        param2 = self.get_random_augment_param(augmenter)

        # relative transform calculation.
        rel_param = self.get_relative_param(augmenter, param1, param2)

        # Applying only image related transform
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return {
            **{
                "transformed_image1": img1,
                "transformed_image2": img2,
                "joints1": joints1,
                "joints2": joints2,
            },
            **rel_param,
        }

    def prepare_pairwise_ablative(
        self, sample: dict, augmenter: SampleAugmenter
    ) -> dict:
        """Prepares samples according to pairwise experiment, i.e. transforming the
        image and keeping track of the relative parameters. Augmentations are isolated.
        Args:
            sample (dict): Underlying data from dataloader class.
            augmenter (SampleAugmenter): Augmenter used to transform sample

        Returns:
            dict: sample containing following elements
                'transformed_image1'
                'transformed_image2'
                'joints1' (2.5D joints)
                'joints2' (2.5D joints)
                'rotation'
                'jitter' ...
        """
        joints25D, _ = convert_to_2_5D(sample["K"], sample["joints3D"])
        if augmenter.crop:
            override_jitter = None
        else:
            # Zero jitter is added incase the cropping is off. It is done to trigger the
            # cropping but always with no translation in image.
            override_jitter = [0, 0]
        if augmenter.rotate:
            override_angle = None
        else:
            override_angle = None
            # override_angle = random.uniform(1, 360)
            # uncomment line above to add this rotation  to both channels
        img1, joints1, _ = augmenter.transform_sample(
            sample["image"], joints25D.clone(), override_angle, override_jitter
        )
        param1 = self.get_random_augment_param(augmenter)

        img2, joints2, _ = augmenter.transform_sample(
            sample["image"], joints25D.clone(), override_angle, override_jitter
        )
        param2 = self.get_random_augment_param(augmenter)

        # relative transform calculation.
        rel_param = self.get_relative_param(augmenter, param1, param2)

        # Applying only image related transform
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return {
            **{
                "transformed_image1": img1,
                "transformed_image2": img2,
                "joints1": joints1,
                "joints2": joints2,
            },
            **rel_param,
        }

    def prepare_supervised_sample(
        self, sample: dict, augmenter: SampleAugmenter
    ) -> dict:
        """Prepares samples for supervised experiment with keypoints.

        Args:
            sample (dict): Underlying data from dataloader class.
            augmenter (SampleAugmenter): Augmenter used to transform sample

        Returns:
            dict: sample containing following elements
                'image'
                'joints'
                'joints3D'
                'K'
                'scale'
                'joints3D_recreated'
        """
        joints25D_raw, scale = convert_to_2_5D(sample["K"], sample["joints3D"])
        joints_raw = (
            sample["joints_raw"]
            if "joints_raw" in sample.keys()
            else sample["joints3D"].clone()
        )
        image, joints25D, transformation_matrix = augmenter.transform_sample(
            sample["image"], joints25D_raw
        )
        sample["K"] = torch.Tensor(transformation_matrix) @ sample["K"]
        if self.config.use_palm:
            sample["joints3D"] = self.move_wrist_to_palm(sample["joints3D"])
            joints25D, scale = convert_to_2_5D(sample["K"], sample["joints3D"])

        joints3D_recreated = convert_2_5D_to_3D(joints25D, scale, sample["K"])
        # This variable is for procrustes analysis, only relevant when youtube data is used

        if self.config.use_palm:
            joints_raw = self.move_wrist_to_palm(joints_raw)

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "joints": joints25D,
            "joints3D": sample["joints3D"],
            "K": sample["K"],
            "scale": scale,
            "joints3D_recreated": joints3D_recreated,
            "joints_valid": sample["joints_valid"],
            "joints_raw": joints_raw,
            "T": torch.Tensor(transformation_matrix),
        }

    def prepare_hybrid1_sample(
        self,
        sample: dict,
        pairwise_augmenter: SampleAugmenter,
        contrastive_augmenter: SampleAugmenter,
    ) -> dict:
        """Prepares samples for basic Hybrid model

        Args:
            sample (dict): Underlying data from dataloader class.
            pairwise_augmenter (SampleAugmenter): Augmenter used to transform sample for
                Pairwise model
            contrastive_augmenter (SampleAugmenter): Augmenter used to transform sample
                for contrastive model.

        Returns:
            dict : sample_containing
                    contrastive_sample.
                    pariwise_sample.
        """
        pairwise_sample = self.prepare_pairwise_ablative(sample, pairwise_augmenter)
        contrastive_sample = self.prepare_experiment4_pretraining(
            sample, contrastive_augmenter
        )
        return {"contrastive": contrastive_sample, "pairwise": pairwise_sample}

    def prepare_hybrid2_sample(self, sample: dict, augmenter: SampleAugmenter) -> dict:
        joints25D, _ = convert_to_2_5D(sample["K"], sample["joints3D"])
        if augmenter.crop:
            override_jitter = None
        else:
            # Zero jitter is added incase the cropping is off. It is done to trigger the
            # cropping but always with no translation in image.
            override_jitter = [0, 0]
        img1, joints1, _ = augmenter.transform_sample(
            sample["image"], joints25D.clone(), None, override_jitter
        )
        param1 = self.get_random_augment_param(augmenter)

        img2, joints2, _ = augmenter.transform_sample(
            sample["image"], joints25D.clone(), None, override_jitter
        )
        param2 = self.get_random_augment_param(augmenter)

        # Applying only image related transform
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return {
            **{"transformed_image1": img1, "transformed_image2": img2},
            **{f"{k}_1": v for k, v in param1.items() if v is not None},
            **{f"{k}_2": v for k, v in param2.items() if v is not None},
        }

    def is_training(self, value: bool):
        """Switches the mode of the data.

        Args:
            value (bool): If value is True then training samples are returned else
        validation samples are returned.
        """
        if value and self._split != "train":
            self._split = "train"
            self.initialize_data_loaders()
        elif not value and self._split != "val":
            self._split = "val"
            self.initialize_data_loaders()

    def get_random_augment_param(self, augmenter: SampleAugmenter) -> dict:
        """Reads the random parameters from the augmenter for calulation of relative
        transformation
        Args:
            augmenter (SampleAugmenter): Augmenter used to transform the sample.
        Returns:
            dict: Containsangle
                    'jitter_x' (translation of centriod of hand)
                    'jitter_y' (translation of centriod of hand)
                    'h' (hue factor)
                    's' (sat factor)
                    'a' (brightness factor)
                    'b' (brightness additive term)
                    'blur_flag'
        """
        angle = augmenter.angle
        jitter_x = augmenter.jitter_x
        jitter_y = augmenter.jitter_y
        h = augmenter.h
        s = augmenter.s
        a = augmenter.a
        b = augmenter.b
        blur_flag = augmenter._gaussian_blur
        crop_margin_scale = augmenter._crop_margin_scale
        return {
            "angle": angle,
            "jitter_x": jitter_x,
            "jitter_y": jitter_y,
            "h": h,
            "s": s,
            "a": a,
            "b": b,
            "blur_flag": blur_flag,
            "crop_margin_scale": crop_margin_scale,
        }

    def get_relative_param(self, augmenter, param1: dict, param2: dict) -> dict:
        """Calculates relative parameters between two set of augmentation params.

        Args:
            augmenter (SampleAugmenter): Augmenter used to transform sample
            param1 (dict): 1st image  augmetation parameters
                            (from get_random_augment_param())
            param2 (dict): 2nd image augmentation parameters

        Returns:
            dict: relative transformation param
        """
        rel_param = {}

        if augmenter.crop:
            jitter_x = param1["jitter_x"] - param2["jitter_x"]
            jitter_y = param1["jitter_y"] - param2["jitter_y"]
            rel_param.update({"jitter": torch.tensor([jitter_x, jitter_y])})

        if augmenter.color_jitter:
            h = param1["h"] - param2["h"]
            s = param1["s"] - param2["s"]
            a = param1["a"] - param2["a"]
            b = param1["b"] - param2["b"]
            rel_param.update({"color_jitter": torch.tensor([h, s, a, b])})

        if augmenter.gaussian_blur:
            blur_flag = param1["blur_flag"] ^ param2["blur_flag"]
            rel_param.update({"blur": torch.Tensor([blur_flag * 1])})

        if augmenter.rotate:
            angle = (param1["angle"] - param2["angle"]) % 360
            rel_param.update({"rotation": torch.Tensor([angle])})
        return rel_param

    def move_wrist_to_palm(self, joints3D: torch.Tensor) -> torch.Tensor:
        wrist_idx = JOINTS.mapping.ait.wrist
        index_mcp_idx = JOINTS.mapping.ait.index_mcp
        joints3D[wrist_idx] = (joints3D[wrist_idx] + joints3D[index_mcp_idx]) / 2
        return joints3D
