import torch
import torch.nn as nn
import numpy as np

class AxisAdaptiveLayer(nn.Module):
    def __init__(self, side: str = "right"):
        super().__init__()
        self.side = side

        # Joints used for axis construction
        self.joints_mapping = [1, 2, 3,    # index
                       4, 5, 6,    # middle
                       10, 11, 12, # ring
                       7, 8, 9,    # pinky
                       13, 14, 15] # thumb

        self.parent_joints_mapping = [0, 1, 2,
                                    0, 4, 5,
                                    0, 10, 11,
                                    0, 7, 8,
                                    0, 13, 14]


        # Define default "up" direction: +Y for fingers, (±1,1,1) for thumb
        if side == "right":
            up_axis_base = np.vstack([[0, 1, 0] for _ in range(13)] + [[1, 1, 1] for _ in range(3)])
        else:  # left
            up_axis_base = np.vstack(([0, 1, 0] for _ in range(13)) + [[-1, 1, 1] for _ in range(3)])

        self.register_buffer("up_axis_base", torch.from_numpy(up_axis_base).float().unsqueeze(0))  # (1, 16, 3)

    def forward(self, joints: torch.Tensor, transforms: torch.Tensor):
        """
        Compute anatomical axes per joint.

        Args:
            joints: (B, 21, 3) - MANO joint positions
            transforms: (B, 16, 4, 4) - MANO absolute transforms

        Returns:
            b_axis, u_axis, l_axis: each (B, 16, 3) - normalized vectors
        """
        B = transforms.shape[0]

        # Compute bone direction vectors from parent to child joints
        bone_vec = joints[:, self.parent_joints_mapping] - joints[:, self.joints_mapping]  # (B, 15, 3)

        # Rotate bone vectors into local joint frame using inverse rotation
        local_rot = transforms[:, 1:, :3, :3].transpose(2, 3)  # (B, 15, 3, 3)
        b_axis = torch.matmul(local_rot, bone_vec.unsqueeze(-1)).squeeze(-1)  # (B, 15, 3)

        # Add wrist reference manually (identity orientation, pointing +X)
        b_axis_0 = torch.tensor([1.0, 0.0, 0.0], device=b_axis.device).view(1, 1, 3).repeat(B, 1, 1)
        b_axis = torch.cat([b_axis_0, b_axis], dim=1)  # (B, 16, 3)

        # Get up vector and compute cross products to form coordinate frame
        u_base = self.up_axis_base.expand(B, 16, 3)  # (B, 16, 3)
        l_axis = torch.cross(b_axis, u_base, dim=-1)
        u_axis = torch.cross(l_axis, b_axis, dim=-1)

        # Normalize
        b_axis = b_axis / (torch.norm(b_axis, dim=-1, keepdim=True) + 1e-8)
        u_axis = u_axis / (torch.norm(u_axis, dim=-1, keepdim=True) + 1e-8)
        l_axis = l_axis / (torch.norm(l_axis, dim=-1, keepdim=True) + 1e-8)

        return b_axis, u_axis, l_axis
