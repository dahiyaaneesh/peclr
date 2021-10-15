import torch
import torchvision
import torch
from tqdm import tqdm


def peclr_to_torchvision(resnet_model, path_to_peclr_weights):
    """Copies parameters from trained peclr model to a corresponding resent from torchvision.
    All the weights until the fc layer of the resent are copied.
    
    Note: Make sure the saved PeCLR model shares same resent size as the "resnet_model" 

    Args:
        resnet_model (tochvision.models.ResNet): A resnet model
        path_to_peclr_weights (str)

    Raises:
        Exception: When the passed model is not of ResNet type
    """
    peclr_weights = torch.load(path_to_peclr_weights, map_location=torch.device("cpu"))
    print(peclr_weights.keys())
    peclr_state_dict = peclr_weights["state_dict"]
    if isinstance(resnet_model, torchvision.models.ResNet):
        resnet_state_dict_list = list(resnet_model.state_dict().items())
        peclr_state_dict_list = [
            (key, peclr_state_dict[key])
            for key in peclr_state_dict
            if "features" in key
        ]
        last_feature_idx = len(peclr_state_dict_list)
        own_state = resnet_model.state_dict()
        for idx in tqdm(range(last_feature_idx)):
            if (
                resnet_state_dict_list[idx][0].split(".")[-1]
                != peclr_state_dict_list[idx][0].split(".")[-1]
            ):
                print("PeCLR layers don't match with Resnet layer ")
                break
            name = resnet_state_dict_list[idx][0]
            param = peclr_state_dict_list[idx][1]
            try:
                own_state[name].copy_(param)
            except Exception as e:
                print("The models are not compatible!")
                print(f"Exception :{e}")
                break
    else:
        raise Exception(f"The selected model is not of type ResNet from torch vision!")


def main():
    resnet152 = torchvision.models.resnet152(pretrained=True)
    # NOTE: This path is just for demonstration.
    peclr_to_torchvision(resnet152,"data/models/637ab5816c43483181143d6189cc39ff/checkpoints/epoch=3.ckpt")
   
if __name__=='__main__':
    main()
