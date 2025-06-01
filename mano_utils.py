import torch
import numpy as np
from config import RIGHT_MANO_PATH
from manopth.manolayer import ManoLayer

class MANOHandMesh:
    """
    Wrapper class for MANO model using manopth.
    Applies pose, shape, and orientation to produce mesh and joint outputs.
    """

    def __init__(self, model_path=RIGHT_MANO_PATH, use_pca=False):
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mano_layer = ManoLayer(
            mano_root='mano',
            use_pca=use_pca,
            ncomps=45,
            flat_hand_mean=True,
            side='right'
        ).to(self.device)

    def apply_transformations_to_mano(self, pose_params, shape_params):
        """
        Apply pose and shape to MANO model.

        Args:
            pose_params (np.ndarray or torch.Tensor): [1, 45] axis-angle hand pose.
            shape_params (np.ndarray or torch.Tensor): [1, 10] shape coefficients.

        Returns:
            vertices (np.ndarray): [778, 3] mesh vertices.
            joints (np.ndarray): [21, 3] joint positions.
            faces (np.ndarray): [1538, 3] mesh triangle indices.
        """
        pose_tensor = torch.as_tensor(pose_params, dtype=torch.float32, device=self.device)
        shape_tensor = torch.as_tensor(shape_params, dtype=torch.float32, device=self.device)

        if pose_tensor.ndim == 1:
            pose_tensor = pose_tensor.unsqueeze(0)
        if shape_tensor.ndim == 1:
            shape_tensor = shape_tensor.unsqueeze(0)

        verts, joints = self.mano_layer(
            th_pose_coeffs=pose_tensor,
            th_betas=shape_tensor
        )

        vertices = verts.detach().cpu().numpy().squeeze()
        joints = joints.detach().cpu().numpy().squeeze()
        faces = self.mano_layer.th_faces.cpu().numpy()

        return vertices, joints, faces
