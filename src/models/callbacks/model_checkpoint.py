from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import torch


class UpdatedModelCheckpoint(ModelCheckpoint):
    def _save_model(self, filepath: str, trainer, pl_module):
        print(f"Saving checkpoint at {filepath}")
        # Note: To only save state dict.
        # torch.save(pl_module.encoder.state_dict(), filepath)
        ModelCheckpoint._save_model(self, filepath, trainer, pl_module)
