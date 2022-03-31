from __future__ import print_function, unicode_literals
import sys
import json

sys.path.append(".")
import argparse
from tqdm import tqdm
import torch
import subprocess
import numpy as np
import os

from src.models.rn_25D_wMLPref import RN_25D_wMLPref
from testing.fh_utils import (
    json_load,
    db_size,
    get_bbox_from_pose,
    read_img,
    convert_order,
    move_palm_to_wrist,
    modify_bbox,
    preprocess,
    create_affine_transform_from_bbox,
)

BBOX_SCALE = 0.33
CROP_SIZE = 224
DS_PATH = "/hdd/Datasets/freihand_dataset/"


def main(base_path, pred_func, out_name, set_name=None):
    """
    Main eval loop: Iterates over all evaluation samples and saves the corresponding 
    predictions.
    """
    # default value
    if set_name is None:
        set_name = "evaluation"
    # init output containers
    xyz_pred_list, verts_pred_list = list(), list()
    K_list = json_load(os.path.join(base_path, "%s_K.json" % set_name))
    scale_list = json_load(os.path.join(base_path, "%s_scale.json" % set_name))
    # iterate over the dataset once
    for idx in tqdm(range(db_size(set_name))):
        if idx >= db_size(set_name):
            break

        # load input image
        img = read_img(idx, base_path, set_name)
        # use some algorithm for prediction
        xyz, verts = pred_func(img, np.array(K_list[idx]), scale_list[idx])
        xyz_pred_list.append(xyz)
        verts_pred_list.append(verts)

    # dump results
    dump(xyz_pred_list, verts_pred_list, out_name)


def dump(xyz_pred_list, verts_pred_list, out_name):
    """ Save predictions into a json file. """
    # make sure its only lists
    xyz_pred_list = [x.tolist() for x in xyz_pred_list]
    verts_pred_list = [x.tolist() for x in verts_pred_list]

    # Filter out ID
    out_ID = out_name.split("_")[-1]
    if not os.path.isdir("out"):
        os.mkdir("out")
    # save to a json
    json_name = f"out/pred_{out_ID}"
    with open(f"{json_name}.json", "w") as fo:
        json.dump([xyz_pred_list, verts_pred_list], fo)
    print(
        "Dumped %d joints and %d verts predictions to %s"
        % (len(xyz_pred_list), len(verts_pred_list), "%s.json" % json_name)
    )
    subprocess.call(["zip", "-j", "%s.zip" % json_name, "%s.json" % json_name])


def pred(img_orig, K_orig, scale, model, T):
    """ 
    Predict joints and vertices from a given sample.
    img: (224, 224, 30 RGB image.
    K: (3, 3) camera intrinsic matrix.
    scale: () scalar metric length of the reference bone.
    1. Get 2D predictions of IMG
    2. Create bbox based on 2D prediction
    3. Reproject bbox into original image
    4. Adjust it how it is done in training
    5. Re-crop hand based on adjusted bbox
    6. Perform prediction again on new crop
    """
    img, K = preprocess(img_orig, K_orig, T, CROP_SIZE)
    # Create feed dict
    feed = {"image": img.float().to(dev), "K": K.float().to(dev)}
    # Predict
    with torch.no_grad():
        output = model(feed)
    kp2d = output["kp25d"][:, :21, :2][0]
    bbox = get_bbox_from_pose(kp2d.cpu().numpy())
    # Apply inverse affine transform
    bbox = np.concatenate((bbox.reshape(2, 2).T, np.ones((1, 2))), axis=0)
    bbox = np.matmul(np.linalg.inv(T)[:2], bbox)
    bbox = bbox.T.reshape(4)
    # Recreate affine transform
    T = create_affine_transform_from_bbox(bbox, CROP_SIZE)
    img, K = preprocess(img_orig, K_orig, T, CROP_SIZE)
    # Create feed dict
    feed = {"image": img.float().to(dev), "K": K.float().to(dev)}
    # Predict again
    with torch.no_grad():
        output = model(feed)

    kp3d = output["kp3d"].view(-1, 3)[:21].cpu().numpy().astype(np.float64)
    # Move palm to wrist
    kp3d = move_palm_to_wrist(kp3d)
    # Convert to Zimmermanns representation
    kp3d = convert_order(kp3d)
    # Unscale (scale is in meters)
    kp3d = kp3d * scale
    # We do not care about vertices
    verts = np.zeros((778, 3))

    assert not np.any(np.isnan(kp3d)), "NaN detected"

    return kp3d, verts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()

    model_path = args.model_path
    dev = torch.device("cuda")
    if "rn50" in model_path:
        model_type = "rn50"
    elif "rn152" in model_path:
        model_type = "rn152"
    else:
        raise Exception(
            "Cannot infer model_type from model_path. Did you rename the .pth file?"
        )
    model_ = RN_25D_wMLPref(backend_model=model_type)
    checkpoint = torch.load(model_path)
    model_.load_state_dict(checkpoint["state_dict"])
    model_.eval()
    model_.to(dev)
    model = lambda feed: model_(feed["image"], feed["K"])
    # Create initial bbox
    bbox = np.array([0, 0, CROP_SIZE, CROP_SIZE], dtype=np.float32)
    bbox = modify_bbox(bbox, BBOX_SCALE)
    T = create_affine_transform_from_bbox(bbox, CROP_SIZE)
    # call with a predictor function
    main(
        DS_PATH,
        pred_func=lambda img, K, scale: pred(img, K, scale, model, T),
        out_name=model_type,
        set_name="evaluation",
    )
