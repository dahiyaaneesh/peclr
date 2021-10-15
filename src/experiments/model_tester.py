from src.experiments.utils import (
    get_model
)
from easydict import EasyDict as edict
from src.constants import HYBRID2_CONFIG
from src.utils import get_console_logger, read_json

def main():
    model_param_path =  HYBRID2_CONFIG
    model_param = edict(read_json(model_param_path))
    model_param.augmentation = []
    model = get_model(
        experiment_type="hybrid2",
        heatmap_flag=False,
        denoiser_flag=False,
    )(config=model_param)
    print(model)

main()