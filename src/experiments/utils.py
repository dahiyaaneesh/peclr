import argparse
import os
from typing import List, Tuple

import torch
from comet_ml import Experiment
from easydict import EasyDict as edict
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from src.constants import SAVED_META_INFO_PATH, SAVED_MODELS_BASE_PATH
from src.data_loader.data_set import Data_Set
from src.experiments.evaluation_utils import evaluate
from src.models.callbacks.upload_comet_logs import UploadCometLogs
# from src.models.semisupervised.denoised_supervised_head_model import (
#     DenoisedSupervisedHead,
# )
# from src.models.semisupervised.heatmap_head_model import HeatmapHead
# from src.models.semisupervised.supervised_head_model import SupervisedHead
# from src.models.supervised.baseline_model import BaselineModel
# from src.models.supervised.denoised_baseline import DenoisedBaselineModel
from src.models.unsupervised.hybrid2_model import Hybrid2Model
from src.models.unsupervised.simclr_model import SimCLR
from src.models.utils import get_latest_checkpoint
from src.models.callbacks.model_checkpoint import UpdatedModelCheckpoint
from src.utils import get_console_logger
from torch.utils.data import WeightedRandomSampler
from torch.utils.data.dataset import ConcatDataset


def get_general_args(
    description: str = "Script for training baseline supervised model",
) -> argparse.Namespace:
    """Function to parse the arguments given as input to a general experiment.
    only parses augmentation flag and data parameters like training ratio, num_workers,
    batchsize, epochs.
    Returns:
       argparse.Namespace: Parsed arguments as namespace.
    """

    parser = argparse.ArgumentParser(description=description)

    # Augmentation flags
    parser.add_argument(
        "--color_drop", action="store_true", help="To enable random color drop"
    )
    parser.add_argument(
        "--color_jitter", action="store_true", help="To enable random jitter"
    )
    parser.add_argument("--crop", action="store_true", help="To enable cropping")
    parser.add_argument(
        "--cut_out", action="store_true", help="To enable random cur out"
    )
    parser.add_argument("--flip", action="store_true", help="To enable random flipping")
    parser.add_argument(
        "--gaussian_blur", action="store_true", help="To enable gaussina blur"
    )
    parser.add_argument(
        "--rotate", action="store_true", help="To rotate samples randomly"
    )
    parser.add_argument(
        "--random_crop", action="store_true", help="To enable random cropping"
    )
    parser.add_argument("--resize", action="store_true", help="To enable resizing")
    parser.add_argument(
        "--sobel_filter", action="store_true", help="To enable sobel filtering"
    )
    parser.add_argument(
        "--gaussian_noise", action="store_true", help="To add gaussian noise."
    )
    parser.add_argument("-tag", action="append", help="Tag for comet", default=[])
    # Training  and data loader params.
    parser.add_argument("-batch_size", type=int, help="Batch size")
    parser.add_argument("-epochs", type=int, help="Number of epochs")
    parser.add_argument("-seed", type=int, help="To add seed")
    parser.add_argument(
        "-num_workers", type=int, help="Number of workers for Dataloader."
    )
    parser.add_argument(
        "-train_ratio", type=float, help="Ratio of train:validation split."
    )
    parser.add_argument(
        "-accumulate_grad_batches",
        type=int,
        help="Number of batches to accumulate gradient.",
    )
    parser.add_argument("-lr", type=float, help="learning rate", default=None)
    parser.add_argument(
        "-optimizer",
        type=str,
        help="Select optimizer",
        default=None,
        choices=["LARS", "adam"],
    )
    parser.add_argument(
        "--denoiser", action="store_true", help="To enable denoising", default=False
    )
    parser.add_argument(
        "--heatmap", action="store_true", help="To enable heatmap model", default=False
    )
    parser.add_argument(
        "-sources",
        action="append",
        help="Data sources to use.",
        default=[],
        choices=["freihand", "interhand", "mpii", "youtube"],
    )
    parser.add_argument(
        "-log_interval",
        type=str,
        help="To enable denoising",
        default="epoch",
        choices=["step", "epoch"],
    )
    parser.add_argument(
        "-experiment_key",
        type=str,
        help="Experiment key of pretrained encoder",
        default=None,
    )
    parser.add_argument(
        "-checkpoint", type=str, help="checkpoint name to restore.", default=""
    )
    parser.add_argument(
        "-meta_file",
        type=str,
        help="File to save the name of the experiment.",
        default=None,
    )
    parser.add_argument(
        "-experiment_name", type=str, help="experiment name for logging", default=""
    )
    parser.add_argument(
        "-save_period",
        type=int,
        help="interval at which experiments should be saved",
        default=1,
    )
    parser.add_argument(
        "-save_top_k", type=int, help="Top snapshots to save", default=3
    )
    parser.add_argument(
        "--encoder_trainable",
        action="store_true",
        help="To enable encoder training in SSL",
        default=False,
    )
    parser.add_argument(
        "-resnet_size",
        type=str,
        help="Resnet size",
        default="18",
        choices=["18", "34", "50", "101", "152"],
    )
    parser.add_argument(
        "-lr_max_epochs", type=int, help="Top snapshots to save", default=None
    )
    parser.add_argument(
        "--use_palm",
        action="store_true",
        help="To regress plam instead of wrist.",
        default=False,
    )
    args = parser.parse_args()
    return args


def get_hybrid1_args(
    description: str = "Script for training hybrid1 model",
) -> argparse.Namespace:
    """Function to parse the arguments given as input to a hybrid1 experiment.
    Returns:
       argparse.Namespace: Parsed arguments as namespace.
    """

    parser = argparse.ArgumentParser(description=description)

    # Augmentation flags
    parser.add_argument(
        "-contrastive",
        action="append",
        help="Add augmentations for contrastive sample.",
        choices=["rotate", "crop", "color_jitter"],
        default=[],
    )
    parser.add_argument(
        "-pairwise",
        action="append",
        help="Add augmentations for pairwise sample.",
        choices=["rotate", "crop", "color_jitter"],
        default=[],
    )
    parser.add_argument("-batch_size", type=int, help="Batch size")
    parser.add_argument("-tag", action="append", help="Tag for comet", default=[])
    parser.add_argument("-epochs", type=int, help="Number of epochs")
    parser.add_argument("-seed", type=int, help="To add seed")
    parser.add_argument(
        "-num_workers", type=int, help="Number of workers for Dataloader."
    )
    parser.add_argument(
        "-train_ratio", type=float, help="Ratio of train:validation split."
    )
    parser.add_argument(
        "-accumulate_grad_batches",
        type=int,
        help="Number of batches to accumulate gradient.",
    )
    parser.add_argument(
        "-optimizer",
        type=str,
        help="Select optimizer",
        default=None,
        choices=["LARS", "adam"],
    )
    parser.add_argument("-lr", type=float, help="learning rate", default=None)
    parser.add_argument(
        "--denoiser", action="store_true", help="To enable denoising", default=False
    )
    parser.add_argument(
        "--heatmap", action="store_true", help="To enable heatmap model", default=False
    )
    parser.add_argument(
        "-sources",
        action="append",
        help="Data sources to use.",
        default=[],
        choices=["freihand", "interhand", "mpii", "youtube"],
    )
    parser.add_argument(
        "-log_interval",
        type=str,
        help="To enable denoising",
        default="epoch",
        choices=["step", "epoch"],
    )
    parser.add_argument(
        "-meta_file",
        type=str,
        help="File to save the name of the experiment.",
        default=None,
    )
    parser.add_argument(
        "-save_period",
        type=int,
        help="interval at which experiments should be saved",
        default=1,
    )
    parser.add_argument(
        "-save_top_k", type=int, help="Top snapshots to save", default=3
    )
    args = parser.parse_args()
    return args


def update_hybrid1_train_args(args: argparse.Namespace, train_param: edict) -> edict:
    if args.pairwise is not None:
        for item in args.pairwise:
            train_param.pairwise.augmentation_flags[item] = True
    if args.contrastive is not None:
        for item in args.contrastive:
            train_param.contrastive.augmentation_flags[item] = True
    if args.train_ratio is not None:
        train_param.train_ratio = (args.train_ratio * 100 % 100) / 100.0
    if args.train_ratio is not None:
        train_param.train_ratio = (args.train_ratio * 100 % 100) / 100.0
    if args.accumulate_grad_batches is not None:
        train_param.accumulate_grad_batches = args.accumulate_grad_batches
    train_param.update(
        update_param(
            args,
            train_param,
            ["batch_size", "epochs", "train_ratio", "num_workers", "seed"],
        )
    )
    return train_param


def update_train_params(args: argparse.Namespace, train_param: edict) -> edict:
    """Updates and returns the training hyper paramters as per args

    Args:
        args (argparse.Namespace): Arguments from get_experiement_args().
        train_param (edict): Default training parameter.

    Returns:
        edict: Updated training parameters.
    """
    if args.train_ratio is not None:
        train_param.train_ratio = (args.train_ratio * 100 % 100) / 100.0
    train_param.update(
        update_param(
            args,
            train_param,
            ["batch_size", "epochs", "train_ratio", "num_workers", "seed", "use_palm"],
        )
    )
    train_param.augmentation_flags = update_param(
        args,
        train_param.augmentation_flags,
        [
            "color_drop",
            "color_jitter",
            "crop",
            "cut_out",
            "flip",
            "gaussian_blur",
            "random_crop",
            "resize",
            "rotate",
            "sobel_filter",
            "gaussian_noise",
        ],
    )
    if args.accumulate_grad_batches is not None:
        train_param.accumulate_grad_batches = args.accumulate_grad_batches
    return train_param


def update_param(args: argparse.Namespace, config: edict, params: List[str]) -> edict:
    """Update the config according to the argument.

    Args:
        args (edict): script arguments
        config (edict): configuration as read from json
        params (List[str]): Name of paramters that must be edited.

    Returns:
        edict: Updated config.
    """
    args_dict = vars(args)
    for param in params:
        if args_dict[param] is not None:
            config[param] = args_dict[param]
    return config


def prepare_name(prefix: str, train_param: edict, hybrid_naming: bool = False) -> str:
    """Encodes the train paramters into string for appropraite naming of experiment.

    Args:
        prefix (str): prefix to attach to the name example sup , simclr, ssl etc.
        train_param (edict): train params used for the experiment.

    Returns:
        str: name of the experiment.
    """
    codes = {
        "color_drop": "CD",
        "color_jitter": "CJ",
        "crop": "C",
        "cut_out": "CO",
        "flip": "F",
        "gaussian_blur": "GB",
        "random_crop": "RC",
        "resize": "Re",
        "rotate": "Ro",
        "sobel_filter": "SF",
        "gaussian_noise": "GN",
    }
    if hybrid_naming:
        pairwise_augmentations = "_".join(
            sorted(
                [
                    codes[key]
                    for key, value in train_param.pairwise.augmentation_flags.items()
                    if value
                ]
            )
        )
        contrastive_augmentations = "_".join(
            sorted(
                [
                    codes[key]
                    for key, value in train_param.contrastive.augmentation_flags.items()
                    if value
                ]
            )
        )
        return (
            f"{prefix}{train_param.batch_size}_rel_{pairwise_augmentations}"
            f"_con_{contrastive_augmentations}"
        )

    else:
        augmentations = "_".join(
            sorted(
                [
                    codes[key]
                    for key, value in train_param.augmentation_flags.items()
                    if value
                ]
            )
        )

        return f"{prefix}{train_param.batch_size}{augmentations}"


def save_experiment_key(
    experiment_name: str, experiment_key: str, filename="default.csv"
):
    """Writes the experiemtn name and key in a  file for quick reference to use the
    saved models.

    Args:
        experiment_name (str]): Name of the experiment. from prepare_name()
        experiment_key (str): comet generated experiment key.
        filename (str, optional): Name of the file where the info should be appended.
         Defaults to "default.csv".
    """
    with open(os.path.join(SAVED_META_INFO_PATH, filename), "a") as f:
        f.write(f"{experiment_name},{experiment_key}\n")


def get_nips_a1_args():
    parser = argparse.ArgumentParser(
        description="Experiment NIPS A1: SIMCLR ablative studies"
    )
    parser.add_argument(
        "augmentation", type=str, default=None, help="Select augmentation to apply"
    )
    args = parser.parse_args()
    return args


def get_nips_a2_args():
    parser = argparse.ArgumentParser(
        description="Experiment NIPS A2: Pairwise ablative studies"
    )
    parser.add_argument(
        "augmentation", type=str, default=None, help="Select augmentation to apply"
    )
    args = parser.parse_args()
    return args


def get_downstream_args():
    parser = argparse.ArgumentParser(description="Downstream training experiment")
    parser.add_argument("experiment_key", type=str, default=None, help="Experiment key")
    parser.add_argument(
        "experiment_name",
        type=str,
        default=None,
        help="Name of the pretrained experiment",
    )
    parser.add_argument(
        "experiment_type",
        type=str,
        default=None,
        help="Type of experiment for tagging.",
    )
    parser.add_argument(
        "--denoiser", action="store_true", help="To enable denoising", default=False
    )
    parser.add_argument(
        "-num_of_checkpoints",
        type=int,
        help="Numberof checkpoints to fine tune",
        default=-1,
    )
    args = parser.parse_args()
    args = parser.parse_args()
    return args


def downstream_evaluation(
    model,
    data: Data_Set,
    num_workers: int,
    batch_size: int,
    logger: Experiment,
    max_crop_jitter: float = 0.0,
    max_rotate_angle: float = 0.0,
    min_rotate_angle: float = 0.0,
    seed: int = 5,
) -> Tuple[dict, dict]:
    """Returns train and validate results respectively.

    Args:
        model ([type]): [description]
        data (Data_Set): [description]
        num_workers (int): [description]
        batch_size (int): [description]
        logger (Experiment):

    Returns:
        Tuple[dict, dict]: [description]
    """
    torch.manual_seed(seed)
    model.eval()
    if isinstance(data, ConcatDataset) and len(data.datasets) > 1:
        val_weights = []
        val_datasets = []
        for i in range(len(data.datasets)):
            val_dataset = data.datasets[i]
            val_dataset.config.augmentation_params.max_angle = max_rotate_angle
            val_dataset.config.augmentation_params.min_angle = min_rotate_angle
            val_dataset.config.augmentation_flags.random_crop = False
            val_dataset.config.augmentation_params.crop_box_jitter = [
                0.0,
                max_crop_jitter,
            ]
            val_dataset.augmenter = val_dataset.get_sample_augmenter(
                val_dataset.config.augmentation_params,
                val_dataset.config.augmentation_flags,
            )
            val_datasets.append(val_dataset)
            val_datasets[-1].is_training(False)
            val_weights += [1.0 / len(val_datasets[-1])] * len(val_datasets[-1])
        data = ConcatDataset(val_datasets)
        validate_results = evaluate(
            model,
            data,
            num_workers=num_workers,
            batch_size=batch_size,
            sampler=WeightedRandomSampler(
                weights=val_weights, num_samples=len(val_weights), replacement=True
            ),
        )
    else:
        data = data.datasets[0] if isinstance(data, ConcatDataset) else data
        data.is_training(False)
        data.config.augmentation_params.max_angle = max_rotate_angle
        data.config.augmentation_params.min_angle = min_rotate_angle
        data.config.augmentation_params.crop_box_jitter = [0.0, max_crop_jitter]
        data.config.augmentation_flags.random_crop = False
        data.augmenter = data.get_sample_augmenter(
            data.config.augmentation_params, data.config.augmentation_flags
        )
        validate_results = evaluate(
            model, data, num_workers=num_workers, batch_size=batch_size
        )
    with logger.experiment.validate():
        logger.experiment.log_metrics(validate_results)
    # return validate_results


def restore_model(model, experiment_key: str, checkpoint: str = ""):
    """Restores the experiment with the most recent checkpoint.

    Args:
        experiment_key (str): experiment key
    """
    saved_state_dict = torch.load(get_latest_checkpoint(experiment_key, checkpoint))[
        "state_dict"
    ]
    print(f"REstroing {get_latest_checkpoint(experiment_key, checkpoint)}")
    model.load_state_dict(saved_state_dict)
    return model


def get_checkpoints(experiment_key: str, number: int = 3) -> List[str]:
    """Returns last 'n' checkpoints.

    Args:
        experiment_key (str): [description]
        number (int, optional): [description]. Defaults to 3.

    Returns:
        List[str]: Name of last n checkpoints.
    """
    return sorted(
        os.listdir(os.path.join(SAVED_MODELS_BASE_PATH, experiment_key, "checkpoints"))
    )[::-1][:number]


def get_model(experiment_type: str, heatmap_flag: bool, denoiser_flag: bool):
    if experiment_type == "simclr":
        if heatmap_flag:
            raise "Not implemented error"
        else:
            return SimCLR
    elif experiment_type == "hybrid2":
        if heatmap_flag:
            raise "Not implemented error"
        else:
            return Hybrid2Model
    elif experiment_type == "semisupervised":
        raise "Not implemented error"
        # if heatmap_flag and not denoiser_flag:
        #     return HeatmapHead
        # elif heatmap_flag and denoiser_flag:
        #     return DenoisedHeatmapHead
        # elif denoiser_flag:
        #     return DenoisedSupervisedHead
        # else:
        #     return SupervisedHead


def get_callbacks(
    logging_interval: str,
    experiment_type: str,
    save_top_k: int = 1,
    period: int = 1,
    monitor: str = "checkpoint_saving_loss",
):
    upload_comet_logs = UploadCometLogs(
        logging_interval, get_console_logger("callback"), experiment_type
    )
    lr_monitor = LearningRateMonitor(logging_interval=logging_interval)
    # saving the best model as per the validation loss.
    checkpoint_callback = UpdatedModelCheckpoint(
        save_top_k=save_top_k, period=period, monitor=monitor
    )
    return {
        "callbacks": [lr_monitor, upload_comet_logs],
        "checkpoint_callback": checkpoint_callback,
    }


def update_model_params(model_param: edict, args, data_length: int, train_param: edict):
    model_param = update_param(
        args, model_param, ["optimizer", "lr", "resnet_size", "lr_max_epochs"]
    )
    model_param.num_samples = data_length
    model_param.batch_size = train_param.batch_size
    model_param.num_of_mini_batch = train_param.accumulate_grad_batches
    return model_param
