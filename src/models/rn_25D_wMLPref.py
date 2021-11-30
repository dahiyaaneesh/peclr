import torch
import torch.nn as nn
from torchvision import models


class ZrootMLP_ref(nn.Module):
    """
    Zroot refinement module taken from: https://arxiv.org/abs/2003.09282
    Given 21 2D and zrel keypoints, plus a zroot estimate, refines the zroot estimate
    via:
    zroot_ref = zroot_est + mlp(2D, zrel, zroot_est)
    """

    def __init__(self):
        super().__init__()

        zroot_ref = nn.Sequential(
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 1),
        )
        self.norm_bone_idx = (3, 8)
        self.zroot_ref = zroot_ref
        epsilon = 1e-8
        self.register_buffer("eps", torch.tensor(epsilon), persistent=False)

    def forward(self, kp3d_unnorm, zrel, K):
        eps = self.eps
        # Recover the scale normalized zroot using https://arxiv.org/pdf/1804.09534.pdf
        m = self.norm_bone_idx[0]
        n = self.norm_bone_idx[1]
        X_m = kp3d_unnorm[:, m : m + 1, 0:1]
        Y_m = kp3d_unnorm[:, m : m + 1, 1:2]
        X_n = kp3d_unnorm[:, n : n + 1, 0:1]
        Y_n = kp3d_unnorm[:, n : n + 1, 1:2]
        zrel_m = zrel[:, m : m + 1]
        zrel_n = zrel[:, n : n + 1]
        # Eq (6)
        a = (X_n - X_m) ** 2 + (Y_n - Y_m) ** 2
        b = 2 * (
            zrel_n * (X_n ** 2 + Y_n ** 2 - X_n * X_m - Y_n * Y_m)
            + zrel_m * (X_m ** 2 + Y_m ** 2 - X_n * X_m - Y_n * Y_m)
        )
        c = (
            (X_n * zrel_n - X_m * zrel_m) ** 2
            + (Y_n * zrel_n - Y_m * zrel_m) ** 2
            + (zrel_n - zrel_m) ** 2
            - 1
        )
        d = (b ** 2) - (4 * a * c)
        # Push sufficiently far away from zero to ensure numerical stability
        a = torch.max(eps, a)
        d = torch.max(eps, d)
        # Eq (7)
        zroot = ((-b + torch.sqrt(d)) / (2 * a)).detach()
        # Refine zroot estimate via an MLP using: https://arxiv.org/abs/2003.09282
        zroot = torch.clamp(zroot, 4.0, 50.0)
        mlp_input = torch.cat(
            (
                zrel.reshape(-1, 21),
                kp3d_unnorm[..., :2].reshape(-1, 42),
                zroot.reshape(-1, 1),
            ),
            dim=1,
        )
        zroot = zroot + self.zroot_ref(mlp_input).reshape(zroot.shape)

        return zroot


class RN_25D_wMLPref(nn.Module):
    def __init__(self, backend_model="rn50"):
        super().__init__()
        # Initialize a torchvision resnet
        if backend_model == "rn50":
            model_func = models.resnet50
        elif backend_model == "rn152":
            model_func = models.resnet152
        else:
            raise Exception(f"Unknown backend_model: {backend_model}")
        backend_model = model_func()
        num_feat = backend_model.fc.in_features
        # 2D + zrel for 21 keypoints: 3 * 21. Please ignore +1, it is no longer used
        backend_model.fc = nn.Linear(num_feat, 3 * 21 + 1)
        # Initialize the zroot refinement module
        zroot_ref = ZrootMLP_ref()

        self.backend_model = backend_model
        self.zroot_ref = zroot_ref
        self.register_buffer(
            "K_default",
            torch.Tensor(
                [
                    [388.9018310596544, 0.0, 112.0],
                    [0.0, 388.71231836584275, 112.0],
                    [0.0, 0.0, 1.0],
                ]
            ).reshape(1,3,3),
            persistent=False,
        )

    def forward(self, img, K=None):

        if K is None:
            # Use a default camera matrix
            K = self.K_default

        out = self.backend_model(img)
        kp25d = out[:, :-1].view(-1,21,3)
        kp2d = kp25d[..., :2]
        zrel = kp25d[..., 2:3]
        # We know that zrel of root is 0
        zrel[:, 0] = 0
        # Acquire refined zroot
        kp2d_h = torch.cat(
            (kp2d, torch.ones((kp2d.shape[0], 21, 1), device=K.device)), dim=2
        )
        kp3d_unnorm = torch.matmul(kp2d_h, K.inverse().transpose(1, 2))
        zroot = self.zroot_ref(kp3d_unnorm, zrel, K)
        # Compute the scale-normalized 3D keypoints using
        # https://arxiv.org/pdf/1804.09534.pdf
        kp3d = kp3d_unnorm * (zrel + zroot)

        output = {}
        output["kp3d"] = kp3d
        output["zrel"] = zrel
        output["kp2d"] = kp2d
        output['kp25d'] = kp25d

        return output
