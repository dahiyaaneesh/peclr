from typing import Dict

import torch
from easydict import EasyDict as edict
from src.models.base_model import BaseModel
from src.models.utils import vanila_contrastive_loss
from torch import Tensor, nn


class SimCLR(BaseModel):
    """
    SimcLR implementation inspired from paper https://arxiv.org/pdf/2002.05709.pdf.
    The code is adapted from pl_bolts library.
    """

    def __init__(self, config: edict):
        super().__init__(config)
        self.projection_head = self.get_projection_head()

    def get_projection_head(self) -> nn.Sequential:
        projection_head = nn.Sequential(
            nn.Linear(
                self.config.projection_head_input_dim,
                self.config.projection_head_hidden_dim,
                bias=True,
            ),
            nn.BatchNorm1d(self.config.projection_head_hidden_dim),
            nn.ReLU(),
            nn.Linear(
                self.config.projection_head_hidden_dim,
                self.config.output_dim,
                bias=False,
            ),
        )
        return projection_head

    def contrastive_step(self, batch: Dict[str, Tensor]) -> Tensor:
        batch_size = batch["transformed_image1"].size()[0]
        concat_batch = torch.cat(
            (batch["transformed_image1"], batch["transformed_image2"]), dim=0
        )
        concat_encoding = self.get_encodings(concat_batch)
        concat_projections = self.projection_head(concat_encoding)
        projection1, projection2 = (
            nn.functional.normalize(concat_projections[:batch_size]),
            nn.functional.normalize(concat_projections[batch_size:]),
        )
        loss = vanila_contrastive_loss(projection1, projection2)
        return loss

    def get_encodings(self, batch_images: Tensor) -> Tensor:
        return self.encoder(batch_images)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        embedding = self.encoder(x)
        projection = self.projection_head(embedding)
        return {"embedding": self.encoder(x), "projection": projection}

    def training_step(self, batch: dict, batch_idx: int) -> Dict[str, Tensor]:
        loss = self.contrastive_step(batch)
        self.train_metrics = {**self.train_metrics, **{"loss": loss}}
        self.plot_params = {
            "image1": batch["transformed_image1"],
            "image2": batch["transformed_image2"],
            "params": {k: v for k, v in batch.items() if "image" not in k},
        }
        return self.train_metrics

    def validation_step(self, batch: dict, batch_idx: int) -> Dict[str, Tensor]:
        loss = self.contrastive_step(batch)
        self.plot_params = {
            "image1": batch["transformed_image1"],
            "image2": batch["transformed_image2"],
            "params": {k: v for k, v in batch.items() if "image" not in k},
        }
        return {"loss": loss}
