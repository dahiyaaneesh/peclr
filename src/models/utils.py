import os
from typing import Dict, Tuple, Union

import kornia
import numpy as np
import torch
import torchvision
from comet_ml import Experiment
from easydict import EasyDict as edict
from src.constants import SAVED_MODELS_BASE_PATH
from src.data_loader.utils import convert_2_5D_to_3D
from src.models.resnet_model import ResNetModel
from src.visualization.visualize import (plot_hybrid2_images,
                                         plot_pairwise_images,
                                         plot_simclr_images,
                                         plot_truth_vs_prediction)
from torch import Tensor, nn


def cal_l1_loss(
    pred_joints: Tensor, true_joints: Tensor, scale: Tensor, joints_valid: Tensor
) -> Tuple[Tensor, Tensor, Tensor]:
    """Calculates L1 loss between the predicted and true joints.  The relative unscaled
    depth (Z) is penalized seperately.

    Args:
        pred_joints (Tensor): Predicted 2.5D joints.
        true_joints (Tensor): True 2.5D joints.
        scale (Tensor): Scale to unscale the z coordinate. If not provide unscaled
            loss_z is returned, otherwise scaled loss_z is returned.

    Returns:
        Tuple[Tensor, Tensor, Tensor]: 2d loss, scaled z relative
            loss and unscaled z relative loss.
    """
    if joints_valid is None:
        joints_valid = torch.ones_like(true_joints[..., -1:])
    pred_uv = pred_joints[..., :-1]
    pred_z = pred_joints[..., -1:]
    true_uv = true_joints[..., :-1]
    true_z = true_joints[..., -1:]
    loss = nn.L1Loss(reduction="none")
    joints_weight = joints_valid / joints_valid.sum()
    loss_2d = (
        loss(pred_uv, true_uv) * joints_weight
    ).sum() / 2  # because there are two values for 2d
    loss_z = loss(pred_z, true_z) * joints_weight
    loss_z_unscaled = (loss_z * scale.view(-1, 1, 1)).sum()
    loss_z = loss_z.sum()
    return (loss_2d, loss_z, loss_z_unscaled)


def calculate_metrics(
    y_pred: Tensor, y_true: Tensor, step: str = "train"
) -> Dict[str, Tensor]:
    """Calculates the metrics on a batch of predicted and true labels.

    Args:
        y_pred (Tensor): Batch of predicted labels.
        y_true (Tensor): Batch of True labesl.
        step (str, optional): This argument specifies whether the metrics are caclulated
            for train or val set. Appends suitable name to the keys in returned
            dictionary. Defaults to "train".

    Returns:
        dict: Calculated metrics as a dictionary.
    """
    distance_joints = (
        torch.sum(((y_pred - y_true) ** 2), 2) ** 0.5
    )  # shape: (batch, 21)
    mean_distance = torch.mean(distance_joints)
    median_distance = torch.median(distance_joints)
    return {f"EPE_mean_{step}": mean_distance, f"EPE_median_{step}": median_distance}


def cal_3d_loss(
    predicton: Tensor,
    joints3d_gt: Tensor,
    scale: Tensor,
    camera_param: Tensor,
    joints_valid: Tensor,
    Z_root_calc: Tensor = None,
) -> Tensor:
    """calculates 3d MAE loss over the predicted 2.5 d joints

    Args:
        predicton (Tensor): batch x 21 x3 , 2.5 d joint predictions
        joints3d_gt (Tensor): batch x 21 x 2, 3D joint ground truth.
        scale (Tensor): batch x1, scale, bone between wrist and index mcp
        camera_param (Tensor): batch x 3 x 3 , camera parametrs
        joints_valid (Tensor): batch x 21 x 1, flags to show if the joints are valid or not
        Z_root_calc (Tensor, optional): If model trained with denoiser, refined z_root. Defaults to None.

    Returns:
        Tensor: MAE between all keypoints.
    """
    prediction3d = convert_2_5D_to_3D(
        predicton, scale, camera_param, is_batch=True, Z_root_calc=Z_root_calc
    )
    joints_weight = joints_valid / joints_valid.sum()
    loss3d = (
        nn.L1Loss(reduction="none")(prediction3d, joints3d_gt) * joints_weight
    ).sum() / 3
    return loss3d


def log_metrics(metrics: dict, comet_logger: Experiment, epoch: int, context_val: bool):
    if context_val:
        with comet_logger.validate():
            comet_logger.log_metrics(metrics, epoch=epoch)
    else:
        with comet_logger.train():
            comet_logger.log_metrics(metrics, epoch=epoch)


def log_image(
    prediction: Tensor,
    y: Tensor,
    x: Tensor,
    gpu: bool,
    context_val: bool,
    comet_logger: Experiment,
):
    if gpu:
        pred_label = prediction.data[0].cpu().numpy()
        true_label = y.data[0].cpu().detach().numpy()
    else:
        pred_label = prediction[0].detach().numpy()
        true_label = y[0].detach().numpy()
    if context_val:
        with comet_logger.validate():
            plot_truth_vs_prediction(
                pred_label, true_label, x.data[0].cpu(), comet_logger
            )
    else:
        with comet_logger.train():
            plot_truth_vs_prediction(
                pred_label, true_label, x.data[0].cpu(), comet_logger
            )


def log_simclr_images(
    img1: Tensor, img2: Tensor, context_val: bool, comet_logger: Experiment
):

    if context_val:
        with comet_logger.validate():
            plot_simclr_images(img1.data[0].cpu(), img2.data[0].cpu(), comet_logger)
    else:
        with comet_logger.train():
            plot_simclr_images(img1.data[0].cpu(), img2.data[0].cpu(), comet_logger)


def vanila_contrastive_loss(z1: Tensor, z2: Tensor, temperature: float = 0.5) -> Tensor:
    """Calculates the contrastive loss as mentioned in SimCLR paper
        https://arxiv.org/pdf/2002.05709.pdf.
    Parts of the code adapted from pl_bolts nt_ext_loss.

    Args:
        z1 (Tensor): Tensor of normalized projections of the images.
            (#samples_in_batch x vector_dim).
        z2 (Tensor): Tensor of normalized projections of the same images but with
            different transformation.(#samples_in_batch x vector_dim)
        temperature (float, optional): Temperature term in the contrastive loss.
            Defaults to 0.5. In SimCLr paper it was shown t=0.5 is good for training
            with small batches.

    Returns:
        Tensor: Contrastive loss (1 x 1)
    """
    z = torch.cat([z1, z2], dim=0)
    n_samples = len(z)

    # Full similarity matrix
    cov = torch.mm(z, z.t().contiguous())
    sim = torch.exp(cov / temperature)

    # Negative similarity
    mask = ~torch.eye(n_samples, device=sim.device).bool()
    neg = sim.masked_select(mask).view(n_samples, -1).sum(dim=-1)

    # Positive similarity :
    pos = torch.exp(torch.sum(z1 * z2, dim=-1) / temperature)
    pos = torch.cat([pos, pos], dim=0)
    loss = -torch.log(pos / neg).mean()
    return loss


def get_latest_checkpoint(experiment_name: str, checkpoint: str = "") -> str:
    """Path to the last saved checkpoint of the trained model.

    Args:
        experiment_name (str): experiment name.
        checkpoint (str): checkpoint name eg. 'epoch=99.ckpt'
    Returns:
        str: absolute path to the latest checkpoint
    """
    checkpoint_path = os.path.join(
        SAVED_MODELS_BASE_PATH, experiment_name, "checkpoints"
    )
    if checkpoint == "":
        checkpoints = os.listdir(checkpoint_path)
        latest_checkpoint = sorted(checkpoints, key=lambda x: int(x[6:-5]))[-1]
    else:
        latest_checkpoint = checkpoint
    return os.path.join(checkpoint_path, latest_checkpoint)


def get_encoder_state_dict(saved_model_path: str, checkpoint: str) -> dict:
    """state dict of encoder of the saved pretrained model at 'saved_model_path'.

    Args:
        experiment_name (str): experiment name.
        checkpoint (str): checkpoint name eg. 'epoch=99.ckpt'

    Returns:
        dict: saved encoder weights.
    """
    saved_state_dict = torch.load(get_latest_checkpoint(saved_model_path, checkpoint))[
        "state_dict"
    ]
    encoder_state_dict = {
        key[8:]: value for key, value in saved_state_dict.items() if "encoder" in key
    }
    return encoder_state_dict


def log_pairwise_images(
    img1: Tensor,
    img2: Tensor,
    gt_pred: Dict[str, Tensor],
    context_val: bool,
    comet_logger: Experiment,
):
    gt_pred = {
        k: [v[0].data[0].cpu().numpy(), v[1].data[0].cpu().numpy()]
        for k, v in gt_pred.items()
    }
    if context_val:
        with comet_logger.validate():
            plot_pairwise_images(
                img1.data[0].cpu(), img2.data[0].cpu(), gt_pred, comet_logger
            )
    else:
        with comet_logger.train():
            plot_pairwise_images(
                img1.data[0].cpu(), img2.data[0].cpu(), gt_pred, comet_logger
            )


def log_hybrid2_images(
    img1: Tensor,
    img2: Tensor,
    params: Dict[str, Tensor],
    context_val: bool,
    comet_logger: Experiment,
):
    params = {k: v.data[0].cpu() for k, v in params.items()}
    if context_val:
        with comet_logger.validate():
            plot_hybrid2_images(
                img1.data[0].cpu(), img2.data[0].cpu(), params, comet_logger
            )
    else:
        with comet_logger.train():
            plot_hybrid2_images(
                img1.data[0].cpu(), img2.data[0].cpu(), params, comet_logger
            )


def get_rotation_2D_matrix(
    angle: Tensor, center_x: Tensor, center_y: Tensor, scale: Tensor
) -> Tensor:
    """Generates 2D rotation matrix transpose. the matrix generated is for the whole batch.
    The implementation of 2D matrix is same as that in openCV.

    Args:
        angle (Tensor): 1D tensor of rotation angles for the batch
        center_x (Tensor): 1D tensor of x coord of center of the keypoints.
        center_y (Tensor): 1D tensor of x coord of center of the keypoints.
        scale (Tensor): Scale, set it to 1.0.

    Returns:
        Tensor: Returns a tensor of 2D rotation matrix for the batch.
    """
    # convert to radians
    angle = angle * np.pi / 180
    alpha = scale * torch.cos(angle)
    beta = scale * torch.sin(angle)
    rot_mat = torch.zeros((len(angle), 3, 2))
    rot_mat[:, :, 0] = torch.stack(
        [alpha, beta, (1 - alpha) * center_x - beta * center_y], dim=1
    )
    rot_mat[:, :, 1] = torch.stack(
        [-beta, alpha, (1 - alpha) * center_y + beta * center_x], dim=1
    )

    return rot_mat


def rotate_encoding(encoding: Tensor, angle: Tensor) -> Tensor:
    """Function to 2D rotate a batch of encodings by a batch of angles.
    The third dimension is n not changed.

    Args:
        encoding (Tensor): Encodings of shape (batch_size,m,3)
        angle ([type]): batch of angles (batch_size,)

    Returns:
        Tensor: Rotated batch of keypoints.
    """
    center_xyz = torch.mean(encoding.detach(), 1)
    rot_mat = get_rotation_2D_matrix(
        angle, center_xyz[:, 0], center_xyz[:, 1], scale=1.0
    )
    rot_mat = rot_mat.to(encoding.device)
    encoding[..., :2] = torch.bmm(
        torch.cat((encoding[..., :2], torch.ones_like(encoding[..., -1:])), dim=2),
        rot_mat,
    )
    return encoding



def translate_encodings(
    encoding: Tensor, translate_x: Tensor, translate_y: Tensor
) -> Tensor:
    """Translates the encodings along first two dimensions with linear scaling

    Args:
        encoding (Tensor): image encodings/projections from the network
        translate_x (Tensor): normlaized jitter along x axis of the input image
        translate_y (Tensor): normalized jitter along y axis of the input image.

    Returns:
        Tensor: Translated encodings based on scaled normalized jitter.
    """
    max_encodings = torch.max(encoding.detach(), dim=1).values
    min_encodings = torch.min(encoding.detach(), dim=1).values
    encoding[..., 0] += (
        translate_x * (max_encodings[:, 0] - min_encodings[:, 0])
    ).view((-1, 1))
    encoding[..., 1] += (
        translate_y * (max_encodings[:, 1] - min_encodings[:, 1])
    ).view((-1, 1))
    return encoding


def translate_encodings2(
    encoding: Tensor, translate_x: Tensor, translate_y: Tensor
) -> Tensor:
    """New strategy, exact translation.

    Args:
        encoding (Tensor): image encodings/projections from the network
        translate_x (Tensor): normlaized jitter along x axis of the input image
        translate_y (Tensor): normalized jitter along y axis of the input image.

    Returns:
        Tensor: Translated encodings based on scaled normalized jitter.
    """
    encoding[..., 0] += translate_x.view((-1, 1))
    encoding[..., 1] += translate_y.view((-1, 1))
    return encoding


def normalize_heatmap(heatmap: Tensor, beta: Tensor = None):
    n, c, _, _ = heatmap.size()
    beta = (
        torch.ones(size=(1, c, 1, 1), requires_grad=False).to(heatmap.device)
        if beta is None
        else beta
    )
    heatmap = torch.exp(heatmap) * beta
    channel_sum = torch.sum(heatmap, dim=[2, 3])
    return heatmap / channel_sum.view([n, c, 1, 1])


def get_denoiser():
    return nn.Sequential(
        nn.Linear(21 * 3 + 1, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Linear(128, 1),
    )


def get_resnet(resnet_size: str, **kwargs):
    if resnet_size == "18":
        model = torchvision.models.resnet18(**kwargs)
        model.fc = nn.Sequential()
    elif resnet_size == "34":
        model = torchvision.models.resnet34(**kwargs)
        model.fc = nn.Sequential()
    elif resnet_size == "50":
        model = torchvision.models.resnet50(**kwargs)
        model.fc = nn.Linear(2048, 512)
    elif resnet_size == "101":
        model = torchvision.models.resnet101(**kwargs)
        model.fc = nn.Linear(2048, 512)
    elif resnet_size == "152":
        model = torchvision.models.resnet152(**kwargs)
        model.fc = nn.Linear(2048, 512)
    else:
        raise NotImplementedError
    return model


def get_wrapper_model(config: edict, pretrained: bool, wrapper:bool=False):
    cfg = edict(
        {
            "model": {
                "backend_model": "resnet" + config.resnet_size,
                "norm_layer": "bn",
                "use_var": False,
                "pretrained": pretrained,
            },
            "dataset": {"np": 21},
            "loss": {"hmap": {"enabled": False}},
        }
    )
    if wrapper:
        return WrapperModel(cfg, mode="pretraining")
    else:
        return ResNetModel(config=cfg, mode="pretraining")


def get_heatmap_transformation_matrix(
    jitter_x: Tensor,
    jitter_y: Tensor,
    scale: Tensor,
    angle: Tensor,
    heatmap_dim: Tensor,
) -> Tensor:
    """
    Generates transfromation matric to revert the transformation on heatmap.

    Args:
        jitter_x (Tensor): x Pixels by which heatmap should be jittered (batch)
        jitter_y (Tensor): y Pixels by which heatmap should be jittered (batch)
        scale (Tensor): Scale factor from crop margin (batch).
        angle (Tensor): Rotation angle (batch)
        heatmap_dim (Tensor): Height and width of heatmap (1x2)

    Returns:
        [Tensor]: Transformation matrix (batch x 2 x3).
    """
    # Making a translation matrix
    translations = torch.cat(
        [jitter_x.view(-1, 1), jitter_y.view(-1, 1)], axis=1
    ).float()
    origin = torch.zeros_like(translations)
    zero_angle = torch.zeros_like(jitter_x[:, 0])
    unit_scale = torch.ones_like(translations)
    # NOTE: The function below returns a batch x 3 x 3 matrix.
    translation_matrix = kornia.get_affine_matrix2d(
        translations=translations, center=origin, angle=zero_angle, scale=unit_scale
    )
    # Making a rotation matrix.
    center_of_rotation = torch.ones_like(translations) * ((heatmap_dim / 2).view(1, 2))
    # NOTE: The function below returns a batch x 2 x 3 matrix.
    rotation_matrix = kornia.get_rotation_matrix2d(
        center=center_of_rotation.float(),
        angle=angle.float(),
        scale=scale.repeat(1, 2).float(),
    )
    # Applying transformations in the order.
    return torch.bmm(rotation_matrix, translation_matrix)


def affine_mat_to_theta(
    affine_mat: Tensor, w: Union[int, float], h: Union[int, float]
) -> Tensor:
    """
    Converts affine matrix in opencv format to theta expected by torch.functional.affine_grid().
    Implementation inspired from
    https://discuss.pytorch.org/t/how-to-convert-an-affine-transform-matrix-into-theta-to-use-torch-nn-functional-affine-grid/24315/2

    Args:
        affine_mat (Tensor): Affine matrix of shape (batch x 2  x 3)
        w (Union[int, float]): width of image/heatmap
        h (Union[int, float]): width of image/heatmap

    Returns:
       theta (Tensor): Affine matrix expected by torch.nn.functional.affine_grid()
    """
    theta = torch.zeros_like(affine_mat)
    theta[:, 0, 0] = affine_mat[:, 0, 0]
    theta[:, 0, 1] = affine_mat[:, 0, 1] * h / w
    theta[:, 0, 2] = (
        affine_mat[:, 0, 2] * 2 / w + affine_mat[:, 0, 0] + affine_mat[:, 0, 1] - 1
    )
    theta[:, 1, 0] = affine_mat[:, 1, 0] * w / h
    theta[:, 1, 1] = affine_mat[:, 1, 1]
    theta[:, 1, 2] = (
        affine_mat[:, 1, 2] * 2 / h + affine_mat[:, 1, 0] + affine_mat[:, 1, 1] - 1
    )
    return theta
