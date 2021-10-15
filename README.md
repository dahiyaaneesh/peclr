# Self-Supervised 3D Hand Pose Estimation from monocular RGB via Contrastive Learning
[Paper](https://arxiv.org/abs/2106.05953)|[Project Page](https://ait.ethz.ch/projects/2021/PeCLR/)

<p align="center">
<img src="teaser.png" alt="drawing" width="500"/>
</p>

This repository contains the code for the paper [Self-Supervised 3D Hand Pose Estimation from monocular RGB via Contrastive Learning](https://arxiv.org/abs/2106.05953).

# Installation
This code has been tested on Ubuntu 18.04.5 and python 3.8.10

1. Setup python environment.

```
cd path_to_peclr_repo
python3 -m venv ~/peclr_env
source ~/peclr_env/bin/activate
```

2.  Install pytorch (1.7.0) and other requirements. More info on installation of  pytorch 1.7.0 can be found [here](https://pytorch.org/get-started/previous-versions/) .
```
pip install torch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 
pip install -r requirements.txt
```


3.  Define environment variables.

```
export BASE_PATH='<path_to_repo>'
export COMET_API_KEY=''
export COMET_PROJECT=''
export COMET_WORKSPACE=''
export PYTHONPATH="$BASE_PATH"
export DATA_PATH="$BASE_PATH/data/raw/"
export SAVED_MODELS_BASE_PATH="$BASE_PATH/data/models/peclr"
export SAVED_META_INFO_PATH="$BASE_PATH/data/models" 
```

4. Download the data.

We perform training and evaluation on [FreiHand](https://lmb.informatik.uni-freiburg.de/projects/freihand/) and youtube3D(https://github.com/arielai/youtube_3d_hands). These datasets should be downloaded in ``data/raw/freihand_dataset`` and  ``data/raw/youtube_3d_hands``folder of peclr directory.

5. Download models.

Note: The pre-trained models will be released soon.

 The models should be downloaded in ``data/models`` folder of peclr directory.




# Training 

Note: [Comet](https://www.comet.ml/) is the logging service used to monitor the training of the models. Setting up comet is optional. It doesn't effect model training.

## Pretraining
Following are the commands used to train the  PeCLR model in Table 2 and 3 of the paper.

### PeCLR
1. Resnet 152:
``python src/experiments/peclr_training.py --color_jitter --random_crop --rotate --crop -resnet_size 152  -sources freihand -sources youtube  --resize   -epochs 100 -batch_size 128  -accumulate_grad_batches 16 -save_top_k 1  -save_period 1   -num_workers 8``

2. ResNet 50:
``python src/experiments/peclr_training.py --color_jitter --random_crop --rotate --crop -resnet_size 50  -sources freihand -sources youtube  --resize   -epochs 100 -batch_size 128  -accumulate_grad_batches 16 -save_top_k 1  -save_period 1   -num_workers 8``

## Loading PeCLR weights into a ResNet model

The pretrained PeCLR model can be easily loaded into a resnet model from torchvision.models. This can be then used for fine-tuning for one or more datasets with labels.
```
from src.models.port_model import peclr_to_torchvision
import torchvision


resnet152 = torchvision.models.resnet152(pretrained=True)
peclr_to_torchvision(resnet152, "path_to_peclr_with_resnet_152_base")
# Note: The last 'fc' layer in resnet model is not updated
```


# Citation
```
@inproceedings{spurr2021self,
  title={Self-Supervised 3D Hand Pose Estimation from monocular RGB via Contrastive Learning},
  author={Spurr, Adrian and Dahiya, Aneesh and Wang, Xi and Zhang, Xucong and Hilliges, Otmar},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={11230--11239},
  year={2021}
}
```