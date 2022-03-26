# Self-Supervised 3D Hand Pose Estimation from monocular RGB via Contrastive Learning
[Paper](https://arxiv.org/abs/2106.05953) | [Project Page](https://ait.ethz.ch/projects/2021/peclr/) | [Blog Post](https://eth-ait.medium.com/peclr-leverage-unlabeled-pose-data-with-pose-equivariant-contrastive-learning-2ca624083614)

<p align="center">
<img src="peclr.gif" alt="drawing" width="1000"/>
</p>

This is the official repository containing the code for the paper [Self-Supervised 3D Hand Pose Estimation from monocular RGB via Contrastive Learning](https://arxiv.org/abs/2106.05953).

# Installation
The code has been tested on Ubuntu 18.04.5 and python 3.8.10

1. Setup python environment.

```
cd path_to_peclr_repo
python3 -m venv ~/peclr_env
source ~/peclr_env/bin/activate
```

2.  Install pytorch (1.7.0) and other requirements. More info on installation of pytorch 1.7.0 can be found [here](https://pytorch.org/get-started/previous-versions/) .
```
pip install torch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 
pip install -r requirements.txt
```


3.  Define the environment variables.

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

4. Download [FreiHand](https://lmb.informatik.uni-freiburg.de/projects/freihand/) and [youtube3Dhands](https://github.com/arielai/youtube_3d_hands) and extract the datasets into ``data/raw/freihand_dataset`` and  ``data/raw/youtube_3d_hands`` of the main PeCLR directory respectively.

# Training 

Note: [Comet](https://www.comet.ml/) is the logging service used to monitor the training of the models. Setting up comet is optional. It does not affect model training.

Following commands can be used to train the best performing PeCLR model of the main paper.

### ResNet-50
```
python src/experiments/peclr_training.py --color_jitter --random_crop --rotate --crop -resnet_size 50  -sources freihand -sources youtube  --resize   -epochs 100 -batch_size 128  -accumulate_grad_batches 16 -save_top_k 1  -save_period 1   -num_workers 8
```

### Resnet-152
```
python src/experiments/peclr_training.py --color_jitter --random_crop --rotate --crop -resnet_size 152  -sources freihand -sources youtube  --resize   -epochs 100 -batch_size 128  -accumulate_grad_batches 16 -save_top_k 1  -save_period 1   -num_workers 8
```

# Loading PeCLR weights into a Torchvision ResNet model

The pre-trained PeCLR weights acquired from training can be easily loaded into a ResNet model from torchvision.models. The pre-trained weights can then be used for fine-tuning on labeled datasets.
```
from src.models.port_model import peclr_to_torchvision
import torchvision


resnet152 = torchvision.models.resnet152(pretrained=True)
peclr_to_torchvision(resnet152, "path_to_peclr_with_resnet_152_base")
# Note: The last 'fc' layer in resnet model is not updated
```

# Pre-trained PeCLR models
We offer ResNet-50 and ResNet-152 pre-trained on FreiHAND and YT3DH using PeCLR. The models can be downloaded [here](https://dataset.ait.ethz.ch/downloads/guSEovHBpR/) and unpacked via `tar`:
```
# Download pre-trained ResNet-50
wget https://dataset.ait.ethz.ch/downloads/guSEovHBpR/peclr_rn50.tar.gz
tar -xvzf peclr_rn50.tar.gz
```
```
# Download pre-trained ResNet-152
wget https://dataset.ait.ethz.ch/downloads/guSEovHBpR/peclr_rn152.tar.gz
tar -xvzf peclr_rn152.tar.gz
```
The models have been converted to torchvision's model description and can be loaded directly:
```
import torch
import torchvision.models as models
# For ResNet-50
rn50 = models.resnet50()
peclr_weights = torch.load('peclr_rn50_yt3dh_fh.pth')
rn50.load_state_dict(peclr_weights['state_dict'])
# For ResNet-152
rn152 = models.resnet152()
peclr_weights = torch.load('peclr_rn152_yt3dh_fh.pth')
rn152.load_state_dict(peclr_weights['state_dict'])
```

# Fine-tuned PeCLR models
We offer ResNet-50 and ResNet-152 fine-tuned on FreiHAND from the above PeCLR pre-trained weights. The models can be downloaded [here](https://dataset.ait.ethz.ch/downloads/guSEovHBpR/) and unpacked via `tar`:
```
# Download fine-tuned ResNet-50
wget https://dataset.ait.ethz.ch/downloads/guSEovHBpR/rn50_peclr_yt3d-fh_pt_fh_ft.tar.gz
tar -xvzf rn50_peclr_yt3d-fh_pt_fh_ft.tar.gz
```
```
# Download fine-tuned ResNet-152
wget https://dataset.ait.ethz.ch/downloads/guSEovHBpR/rn152_peclr_yt3d-fh_pt_fh_ft.tar.gz
tar -xvzf rn152_peclr_yt3d-fh_pt_fh_ft.tar.gz
```
The model weights follow the model description of `src/models/rn_25D_wMLPref.py`. Thus, one can load them in the following manner:
```
import torch
from src.models.rn_25D_wMLPref import RN_25D_wMLPref
# For RN50
model_type = 'rn50'
# For RN152
model_type = 'rn152'
model = RN_25D_wMLPref(backend_model=model_type)
model_path = f'{model_type}_peclr_yt3d-fh_pt_fh_ft.pth'
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['state_dict'])
```
These model weights achieve the following performance on the FreiHAND leaderboard:
```
ResNet-50 + PeCLR
Evaluation 3D KP results:
auc=0.357, mean_kp3d_avg=4.71 cm
Evaluation 3D KP ALIGNED results:
auc=0.860, mean_kp3d_avg=0.71 cm
```
```
ResNet-152 + PeCLR
Evaluation 3D KP results:
auc=0.360, mean_kp3d_avg=4.56 cm
Evaluation 3D KP ALIGNED results:
auc=0.868, mean_kp3d_avg=0.66 cm
```
# Citation
If this repository has been useful for your project, please cite the following work:
```
@inproceedings{spurr2021self,
  title={Self-Supervised 3D Hand Pose Estimation from monocular RGB via Contrastive Learning},
  author={Spurr, Adrian and Dahiya, Aneesh and Wang, Xi and Zhang, Xucong and Hilliges, Otmar},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={11230--11239},
  year={2021}
}
```
If the RN_25D_wMLPref model description was useful for your project, please cite the following works:
```
@inproceedings{iqbal2018hand,
  title={Hand pose estimation via latent 2.5 d heatmap regression},
  author={Iqbal, Umar and Molchanov, Pavlo and Gall, Thomas Breuel Juergen and Kautz, Jan},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={118--134},
  year={2018}
}
```
```
@inproceedings{spurr2020weakly,
  title={Weakly supervised 3d hand pose estimation via biomechanical constraints},
  author={Spurr, Adrian and Iqbal, Umar and Molchanov, Pavlo and Hilliges, Otmar and Kautz, Jan},
  booktitle={Computer Vision--ECCV 2020: 16th European Conference, Glasgow, UK, August 23--28, 2020, Proceedings, Part XVII 16},
  pages={211--228},
  year={2020},
  organization={Springer}
}
```
