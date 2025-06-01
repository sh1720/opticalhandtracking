import numpy as np
import torch

from mano_utils import MANOHandMesh
from collections import deque
from transforms3d.quaternions import quat2mat
from transforms3d.axangles import mat2axangle

class MANOHandJoints:
  n_joints = 21

  labels = [
    'W', #0
    'I0', 'I1', 'I2', #3
    'M0', 'M1', 'M2', #6
    'L0', 'L1', 'L2', #9
    'R0', 'R1', 'R2', #12
    'T0', 'T1', 'T2', #15
    'I3', 'M3', 'L3', 'R3', 'T3' #20, tips are manually added (not in MANO)
  ]

  # finger tips are not joints in MANO, we label them on the mesh manually
  mesh_mapping = {16: 333, 17: 444, 18: 672, 19: 555, 20: 744}

  parents = [
    None,
    0, 1, 2,
    0, 4, 5,
    0, 7, 8,
    0, 10, 11,
    0, 13, 14,
    3, 6, 9, 12, 15
  ]


class MPIIHandJoints:
  n_joints = 21

  labels = [
    'W', #0
    'T0', 'T1', 'T2', 'T3', #4
    'I0', 'I1', 'I2', 'I3', #8
    'M0', 'M1', 'M2', 'M3', #12
    'R0', 'R1', 'R2', 'R3', #16
    'L0', 'L1', 'L2', 'L3', #20
  ]

  parents = [
    None,
    0, 1, 2, 3,
    0, 5, 6, 7,
    0, 9, 10, 11,
    0, 13, 14, 15,
    0, 17, 18, 19
  ]

def to_dict(joints):
    temp_dict = dict()
    for i in range(21):
        temp_dict[i] = joints[:, [i]]

    return temp_dict



def extract_mano_template():
    """
    Extracts the MANO template joint positions in the neutral pose (zero shape, zero pose).

    Parameters:
        model_path (str): Path to the MANO .pkl file (e.g. RIGHT_MANO_PATH)

    Returns:
        joints (np.ndarray): [21, 3] joint positions in the neutral template pose.
        vertices (np.ndarray): [778, 3] mesh vertices in the neutral pose.
        faces (np.ndarray): Mesh faces (for visualization).
    """
    hand_model = MANOHandMesh()
    
    # Zero pose: [1, 45] for MANO (15 joints × 3 axis-angle)
    pose_params = np.zeros((1, 48), dtype=np.float32)

    
    # Zero shape parameters: [1, 10]
    shape_params = np.zeros((1, 10), dtype=np.float32)
    
    v , j, f = hand_model.apply_transformations_to_mano(
        pose_params=pose_params,
        shape_params=shape_params
    )

     # Base 16 joints
    # joints = j[0, :16]  # [16, 3]
    vertices = v  # [778, 3]

    # Fingertip joints from mesh vertex indices
    tips = [vertices[idx] for idx in MANOHandJoints.mesh_mapping.values()]  # [5, 3]

    # Combine to form 21-joint array
    joints_21 = np.vstack([j, np.array(tips)])  # [21, 3]
    
    return joints_21, vertices, f  # Return first 21 joints only

def compute_angle_axis_from_rotmat(R: np.ndarray) -> np.ndarray:
    """
    Converts a 3x3 rotation matrix to a 3D axis-angle representation.
    
    Parameters:
        R (np.ndarray): [3, 3] rotation matrix.
        
    Returns:
        axis_angle (np.ndarray): [3,] axis-angle vector.
    """
    assert R.shape == (3, 3), "Input must be a 3x3 rotation matrix"

    # Compute rotation angle from trace
    cos_theta = (np.trace(R) - 1) / 2
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Numerical stability
    theta = np.arccos(cos_theta)

    if np.isclose(theta, 0.0):
        return np.zeros(3, dtype=R.dtype)

    # Compute rotation axis
    rx = R[2, 1] - R[1, 2]
    ry = R[0, 2] - R[2, 0]
    rz = R[1, 0] - R[0, 1]
    axis = np.array([rx, ry, rz], dtype=R.dtype)
    axis /= (2 * np.sin(theta) + 1e-8)

    return axis * theta  # axis-angle vector
