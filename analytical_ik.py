import numpy as np
import transforms3d 
import transforms3d

from utils import compute_angle_axis_from_rotmat, to_dict, clip_pose_R_per_joint
from config import MANOHandJoints, MAX_JOINT_ANGLES_MANO, SPHERICAL_ANGLE_LIMITS_MANO

# import torch
# import numpy as np
# from scipy.spatial.transform import Rotation as R
# from axis_layer import AxisAdaptiveLayer  # your own class

# import numpy as np
# import transforms3d
# from transforms3d.axangles import axangle2mat
# from transforms3d.axangles import mat2axangle

# def spherical_soft_penalty(theta, phi, idx, weight=1.0):
#     min_theta, max_theta = SPHERICAL_ANGLE_LIMITS_MANO[idx][0]
#     min_phi, max_phi = SPHERICAL_ANGLE_LIMITS_MANO[idx][1]

#     penalty = 0
#     if theta < min_theta:
#         penalty += weight * (theta - min_theta) ** 2
#     elif theta > max_theta:
#         penalty += weight * (theta - max_theta) ** 2

#     if phi < min_phi:
#         penalty += weight * (phi - min_phi) ** 2
#     elif phi > max_phi:
#         penalty += weight * (phi - max_phi) ** 2

#     return penalty


# def cartesian_to_spherical(x, y, z):
#     r = np.linalg.norm([x, y, z])
#     theta = np.arctan2(y, x)       # azimuth
#     phi = np.arcsin(z / r)         # elevation
#     return theta, phi, r

# def spherical_to_cartesian(theta, phi, r=1.0):
#     x = r * np.cos(phi) * np.cos(theta)
#     y = r * np.cos(phi) * np.sin(theta)
#     z = r * np.sin(phi)
#     return np.array([x, y, z])

# def rotation_matrix_from_vectors(a, b):
#     a = a / np.linalg.norm(a)
#     b = b / np.linalg.norm(b)
#     v = np.cross(a, b)
#     c = np.dot(a, b)
#     s = np.linalg.norm(v)
#     if s < 1e-8:
#         return np.eye(3) if c > 0 else -np.eye(3)
#     vx = np.array([[0, -v[2], v[1]],
#                    [v[2], 0, -v[0]],
#                    [-v[1], v[0], 0]])
#     R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))
#     return R

# def adaptive_IK(T_, P_):
#     T = T_.copy().astype(np.float64).T
#     P = P_.copy().astype(np.float64).T

#     P = to_dict(P)
#     T = to_dict(T)

#     R, R_pa_k, q = {}, {}, {}
#     q[0] = T[0]

#     P_0 = np.concatenate([P[i] - P[0] for i in [1, 5, 9, 13, 17]], axis=-1)
#     T_0 = np.concatenate([T[i] - T[0] for i in [1, 5, 9, 13, 17]], axis=-1)
#     H = np.matmul(T_0, P_0.T)
#     U, S, V_T = np.linalg.svd(H)
#     V = V_T.T
#     R0 = np.matmul(V, U.T)

#     if np.linalg.det(R0) < 0:
#         V[:, 2] *= -1
#         R0 = np.matmul(V, U.T)

#     R[0] = R0
#     for i in [1, 5, 9, 13, 17]:
#         R[i] = R[0].copy()

#     joint_indices = list(range(len(MANOHandJoints.parents)))
#     kinematic_tree = [i for i in joint_indices if MANOHandJoints.parents[i] is not None]
#     snap_parents = {i: p for i, p in enumerate(MANOHandJoints.parents) if p is not None}
#     penalty = 0

#     for k in kinematic_tree:
#         pa = snap_parents[k]
#         ref_idx = snap_parents.get(pa, 0)
#         q[pa] = R[pa] @ (T[pa] - T[ref_idx]) + q[ref_idx]
#         delta_p_k = np.linalg.inv(R[pa]) @ (P[k] - q[pa])
#         delta_t_k = T[k] - T[pa]

#         delta_t_k = np.asarray(delta_t_k).reshape(3,)
#         delta_p_k = np.asarray(delta_p_k).reshape(3,)       
#         axis = np.cross(delta_t_k, delta_p_k)
#         axis /= np.linalg.norm(axis) + 1e-8

#         cos_alpha = np.dot(delta_t_k, delta_p_k) / (
#             (np.linalg.norm(delta_t_k) + 1e-8) * (np.linalg.norm(delta_p_k) + 1e-8)
#         )
#         alpha = np.arccos(np.clip(cos_alpha, -1.0, 1.0))

#         R_pa_k[k] = axangle2mat(axis, alpha)
#         R[k] = R[pa] @ R_pa_k[k]

#     id2rot = {
#         2: 13, 3: 14, 4: 15,
#         6: 1, 7: 2, 8: 3,
#         10: 4, 11: 5, 12: 6,
#         14: 10, 15: 11, 16: 12,
#         18: 7, 19: 8, 20: 9,
#     }

#     pose_R = np.zeros((1, 16, 3, 3))
#     pose_R[0, 0] = R[0]
#     for key, value in id2rot.items():
#         pose_R[0, value] = R_pa_k[key]

#     axis_angle = np.zeros((1, 16, 3))
#     for i in range(16):
#         rotmat = pose_R[0, i]
#         axis, angle = mat2axangle(rotmat)
#         x, y, z = axis
#         r = np.linalg.norm([x, y, z])
#         theta = np.arccos(z / r) if r > 1e-8 else 0.0
#         phi = np.arctan2(y, x)

#         print(f"[DEBUG] theta: {theta}")
#         print(f"[DEBUG] phi: {phi}")


#         theta_min, theta_max = SPHERICAL_ANGLE_LIMITS_MANO[i][0]
#         phi_min, phi_max = SPHERICAL_ANGLE_LIMITS_MANO[i][1]

#         theta = np.clip(theta, theta_min, theta_max)
#         phi = np.clip(phi, phi_min, phi_max)

#         penalty += spherical_soft_penalty(theta, phi, i)


#         clamped_axis = np.array([
#             np.sin(theta) * np.cos(phi),
#             np.sin(theta) * np.sin(phi),
#             np.cos(theta)
#         ])
#         axis_angle[0, i] = clamped_axis * angle

#     return axis_angle, pose_R


angels0 = np.zeros((1,21))

def adaptive_IK(T_, P_):
    '''
    Computes pose parameters given template and predictions.
    We think the twist of hand bone could be omitted.

    :param T: template ,21*3
    :param P: target, 21*3
    :return: pose params.
    '''

    T = T_.copy().astype(np.float64)
    P = P_.copy().astype(np.float64)

    P = P.transpose(1, 0)
    T = T.transpose(1, 0)

    # to dict
    P = to_dict(P)
    T = to_dict(T)

    # some globals
    R = {}
    R_pa_k = {}
    q = {}

    q[0] = T[0]  # in fact, q[0] = P[0] = T[0].

    # compute R0, here we think R0 is not only a Orthogonal matrix, but also a Rotation matrix.
    # you can refer to paper "Least-Squares Fitting of Two 3-D Point Sets. K. S. Arun; T. S. Huang; S. D. Blostein"
    # It is slightly different from  https://github.com/Jeff-sjtu/HybrIK/blob/main/hybrik/utils/pose_utils.py#L4, in which R0 is regard as orthogonal matrix only.
    # Using their method might further boost accuracy.
    P_0 = np.concatenate([P[1] - P[0], P[5] - P[0],
                          P[9] - P[0], P[13] - P[0],
                          P[17] - P[0]], axis=-1)
    T_0 = np.concatenate([T[1] - T[0], T[5] - T[0],
                          T[9] - T[0], T[13] - T[0],
                          T[17] - T[0]], axis=-1)
    H = np.matmul(T_0, P_0.T)

    U, S, V_T = np.linalg.svd(H)
    V = V_T.T
    R0 = np.matmul(V, U.T)

    det0 = np.linalg.det(R0)

    if abs(det0 + 1) < 1e-6:
        V_ = V.copy()

        if (abs(S) < 1e-4).sum():
            V_[:, 2] = -V_[:, 2]
            R0 = np.matmul(V_, U.T)

    R[0] = R0

    # the bone from 1,5,9,13,17 to 0 has same rotations
    R[1] = R[0].copy()
    R[5] = R[0].copy()
    R[9] = R[0].copy()
    R[13] = R[0].copy()
    R[17] = R[0].copy()

    # Get list of all joint indices
    joint_indices = list(range(len(MANOHandJoints.parents)))

    # Build kinematic tree (excluding the root joint 0)
    kinematic_tree = [i for i in joint_indices if MANOHandJoints.parents[i] is not None]

    # Build parent lookup dictionary (skip root since it has no parent)
    snap_parents = {i: p for i, p in enumerate(MANOHandJoints.parents) if p is not None}
    # compute rotation along kinematics
    for k in kinematic_tree:
        pa = snap_parents[k]
        pa_pa = snap_parents.get(pa)
        origin_idx = 0  # root joint index
        pa_pa = snap_parents.get(pa)
        ref_idx = origin_idx if pa_pa is None else pa_pa
        
        q[pa] = np.matmul(R[pa], (T[pa] - T[ref_idx])) + q[ref_idx]
        delta_p_k = np.matmul(np.linalg.inv(R[pa]), P[k] - q[pa])
        delta_p_k = delta_p_k.reshape((3,))
        delta_t_k = T[k] - T[pa]
        delta_t_k = delta_t_k.reshape((3,))
        temp_axis = np.cross(delta_t_k, delta_p_k)
        axis = temp_axis / (np.linalg.norm(temp_axis, axis=-1) + 1e-8)
        temp = (np.linalg.norm(delta_t_k, axis=0) + 1e-8) * (np.linalg.norm(delta_p_k, axis=0) + 1e-8)
        cos_alpha = np.clip(np.dot(delta_t_k, delta_p_k) / temp, -1.0, 1.0)
        alpha = np.arccos(cos_alpha)
        # # Reflect angle if the axis would rotate in the opposite direction
        # if np.dot(np.cross(delta_t_k, delta_p_k), axis) < 0:
        #     alpha *= -1

        twist = delta_t_k
        D_sw = transforms3d.axangles.axangle2mat(axis=axis, angle=alpha, is_normalized=False)
        D_tw = transforms3d.axangles.axangle2mat(axis=twist, angle=angels0[:, k], is_normalized=False)
        R_pa_k[k] = np.matmul(D_sw, D_tw)
        R[k] = np.matmul(R[pa], R_pa_k[k])

    id2rot = {
        2: 13, 3: 14, 4: 15,
        6: 1, 7: 2, 8: 3,
        10: 4, 11: 5, 12: 6,
        14: 10, 15: 11, 16: 12,
        18: 7, 19: 8, 20: 9,
    }
    
    pose_R = np.zeros((1, 16, 3, 3))
    pose_R[0, 0] = R[0]


    for key in id2rot.keys():
        value = id2rot[key]
        pose_R[0, value] = R_pa_k[key]

    # print(f"[DEBUG] {pose_R}")

    # pose_R = clip_pose_R_per_joint(pose_R)
    axis_angle = np.zeros((1, 16, 3))

    for i in range(16):
        rotmat = pose_R[0, i]  # shape (3, 3)
        axis_angle[0, i] = compute_angle_axis_from_rotmat(rotmat)


    return axis_angle, pose_R
