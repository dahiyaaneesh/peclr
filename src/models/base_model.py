import math
from typing import Dict, Iterator, List, Tuple, Union

import torch
from easydict import EasyDict as edict
from pl_bolts.optimizers.lars_scheduling import LARSWrapper
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pytorch_lightning.core.lightning import LightningModule
from src.models.utils import get_resnet, get_wrapper_model
from torch.optim.lr_scheduler import CosineAnnealingLR


class BaseModel(LightningModule):
    """This is the base class inherited by all the models used in the thesis.
    It on the other hand inherits from Lightening module. It defines functions for
    setting up optimizer, schedulers.
    """

    def __init__(self, config: edict):
        super().__init__()
        if "resnet_size" in config.keys():
            # self.encoder = get_resnet(config.resnet_size, pretrained=True)
            self.encoder = get_wrapper_model(config, pretrained=True)
        self.config = config
        self.train_metrics_epoch = {}
        self.train_metrics = {}
        self.validation_metrics_epoch = {}
        self.plot_params = {}

    def exclude_from_wt_decay(
        self,
        named_params: Iterator[Tuple[str, torch.Tensor]],
        weight_decay: float,
        skip_list: List[str] = ["bias", "bn"],
    ) -> List[Dict[str, Union[list, float]]]:

        params = []
        excluded_params = []

        for name, param in named_params:
            if not param.requires_grad:
                continue
            elif any(layer_name in name for layer_name in skip_list):
                excluded_params.append(param)
            else:
                params.append(param)

        return [
            {"params": params, "weight_decay": weight_decay},
            {"params": excluded_params, "weight_decay": 0.0},
        ]

    def setup(self, stage: str):
        global_batch_size = self.trainer.world_size * self.config.batch_size
        self.train_iters_per_epoch = self.config.num_samples // global_batch_size

    def configure_optimizers(self) -> Tuple[list, list]:

        parameters = self.exclude_from_wt_decay(
            self.named_parameters(), weight_decay=self.config.opt_weight_decay
        )
        optimizer = torch.optim.Adam(
            parameters,
            lr=self.config.lr
            * math.sqrt(self.config.batch_size * self.config.num_of_mini_batch),
        )
        warmup_epochs = (
            self.config.warmup_epochs
            * self.train_iters_per_epoch
            // self.config.num_of_mini_batch
        )
        # updating the max epochs for learning rate scheduler for fair comparision of fine-tunes and fully
        # supervised models.
        if (
            "lr_max_epochs" in self.config.keys()
            and self.config["lr_max_epochs"] is not None
        ):
            max_epochs = (
                self.config["lr_max_epochs"]
                * self.train_iters_per_epoch
                // self.config.num_of_mini_batch
            )
        else:
            max_epochs = (
                self.trainer.max_epochs
                * self.train_iters_per_epoch
                // self.config.num_of_mini_batch
            )

        if self.config.optimizer == "LARS":
            optimizer = LARSWrapper(optimizer)
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=warmup_epochs,
                max_epochs=max_epochs,
                warmup_start_lr=0,
                eta_min=0,
            )
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs)

        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return [optimizer], [scheduler]

    def training_epoch_end(self, outputs: List[dict]):
        metric_keys = outputs[0].keys()
        self.train_metrics_epoch = {
            key: torch.stack([x[key] for x in outputs]).mean() for key in metric_keys
        }
        # NOTE: Comment this part and uncomment the similar block to stop based on validation metrics
        if "loss_3d" in metric_keys:
            self.log("checkpoint_saving_loss", self.train_metrics_epoch["loss_3d"])
        else:
            self.log("checkpoint_saving_loss", self.train_metrics_epoch["loss"])

    def validation_epoch_end(self, outputs: List[dict]):
        metric_keys = outputs[0].keys()
        metrics = {
            key: torch.stack([x[key] for x in outputs]).mean() for key in metric_keys
        }
        # NOTE: Comment this part and uncomment the similar block to stop based on train metrics
        # if "loss_3d" in metrics.keys():
        #     self.log("checkpoint_saving_loss", metrics["loss_3d"])
        # else:
        #     self.log("checkpoint_saving_loss", metrics["loss"])
        self.validation_metrics_epoch = metrics
