import torch
import torch.nn as nn
import torchvision.models as models


class ResNetModel(nn.Module):
    """Adapted ResNet model with layers renamed.
    """

    def __init__(self, config, mode=""):
        super().__init__()
        self.mode = mode
        resnet_name = config.model.backend_model.lower()
        model_function = self.get_resnet(resnet_name)
        model = model_function(pretrained=config.model.pretrained, norm_layer =nn.BatchNorm2d)
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
            nn.AdaptiveAvgPool2d(output_size=(1,1)),
        )
        self.final_layer = nn.Sequential(
                    nn.Linear(model.fc.in_features,21*3+1),
                    )
    
    def get_resnet(self, resnet_name):
        if "resnet18" == resnet_name:
            return models.resnet18
        elif "resnet34" == resnet_name:
            return models.resnet34
        elif "resnet50" == resnet_name:
            return models.resnet50
        elif "resnet101" == resnet_name:
            return models.resnet101
        elif "resnet152" == resnet_name:
            return models.resnet152
        else:
            raise NotImplementedError
    
    def forward(self, x):
        """Forward method, return embeddings when the mode is pretraining.
        and return 2.5D keypoints, None and scale otherwise.
        """
        z = self.features(x)
        z = z.flatten(start_dim=1)
        if self.mode=="pretraining":
            return z
        else:
            z = self.final_layer(z)
            return z[:,:21*3], None, z[:,-1]
        
        