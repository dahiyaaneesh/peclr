from typing import NewType, Tuple

import torch

#  Standard types to be used in the codebase.
JOINTS_25D = NewType(
    (
        "Hand coordinates of shape 21 x 3, where first two columns are x and y"
        "in image plane and third columnis scaled relative depth."
    ),
    torch.Tensor,
)
SCALE = NewType("scale, distance between the root and index mcp", torch.Tensor)
JOINTS_3D = NewType("Hand coordinates of shape 21 x 3", torch.Tensor)
CAMERA_PARAM = NewType("Camera params of dim 2 x 3", torch.Tensor)
