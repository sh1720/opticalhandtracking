import numpy as np

from mano_utils import MANOHandMesh
from collections import deque
from config import MANOHandJoints, MAX_JOINT_ANGLES_MANO
from scipy.spatial.transform import Rotation as R


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


def clip_pose_R_per_joint(pose_R):
    """
    pose_R: (1, 16, 3, 3) array of rotation matrices
    Returns: (1, 16, 3, 3) array with clipped rotations
    """
    clipped_rotmats = pose_R.copy()

    for i in range(16):
        rotmat = pose_R[0, i]
        limits = MAX_JOINT_ANGLES_MANO[i]  # [(minX, maxX), (minY, maxY), (minZ, maxZ)]

        # Convert to Euler angles (in radians)
        r = R.from_matrix(rotmat)
        euler = r.as_euler('XYZ', degrees=False)
        r = R.from_matrix(pose_R[0, i])
        print(f"Euler XYZ: {r.as_euler('XYZ')}")
        print(f"Euler ZYX: {r.as_euler('ZYX')}")


        print(f"[DEBUG] Euler: {euler}")
        # Clamp each axis
        for axis in range(3):
            euler[axis] = np.clip(euler[axis], limits[axis][0], limits[axis][1])
            print (f"[DEBUG]: CLIPPED! ")

        # Convert back to rotation matrix
        r_clipped = R.from_euler('XYZ', euler, degrees=False)
        clipped_rotmats[0, i] = r_clipped.as_matrix()

    return clipped_rotmats
