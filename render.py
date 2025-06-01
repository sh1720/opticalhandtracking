import cv2
import numpy as np 
import trimesh 
import pyrender 
import pickle as pickle
import matplotlib.pyplot as plt
import numpy as np

from utils import *
from config import DETNET_SKELETON



def draw_hand_keypoints(image, image_points, xyz_points, skeleton=DETNET_SKELETON, blocking=False):
    for start, end in skeleton:
        if start < len(image_points) and end < len(image_points):
            cv2.line(image, image_points[start], image_points[end], (0, 255, 255), 2)

    for i, (u, v) in enumerate(image_points):
        cv2.circle(image, (u, v), 3, (0, 255, 0), -1)
        coordinate_text = f"({round(xyz_points[i][0], 1)}, {round(xyz_points[i][1], 1)}, {round(xyz_points[i][2], 1)})"
        cv2.putText(image, coordinate_text, (u + 5, v + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)

    return image



def model_to_mano_window(vertices, faces): 
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process = False)
    # Convert Trimesh to Pyrender mesh
    render_mesh = pyrender.Mesh.from_trimesh(mesh)

    return render_mesh
    # # Create scene and add mesh
    # scene = pyrender.Scene()
    # scene.add(render_mesh)

    # # Render viewer
    # mesh = open3d.geometry.triangle
#     # pyrender.Viewer(scene, use_raymond_lighting=True)
#     mesh.vertices = open3d.utility.Vector3dVector(vertices)
#     mesh.paint_uniform_color([228 / 255, 178 / 255, 148 / 255])
#     mesh.compute_triangle_normals()
#     mesh.compute_vertex_normals()
#     viewer.update_geometry(mesh)
#     viewer.poll_events()

# import open3d as o3d
# import numpy as np

# def model_to_mano_window(vertices, faces):
#     """
#     Visualize MANO mesh using Open3D.

#     Args:
#         vertices (np.ndarray): (N, 3) vertex array.
#         faces (np.ndarray): (M, 3) face indices.
#     """
#     mesh = o3d.geometry.TriangleMesh()
#     mesh.vertices = o3d.utility.Vector3dVector(vertices)
#     mesh.triangles = o3d.utility.Vector3iVector(faces)
#     mesh.paint_uniform_color([228 / 255, 178 / 255, 148 / 255])
#     mesh.compute_triangle_normals()
#     mesh.compute_vertex_normals()

#     # Create visualizer window
#     viewer = o3d.visualization.Visualizer()
#     viewer.create_window(window_name='MANO Mesh', width=640, height=480)
#     viewer.add_geometry(mesh)
#     viewer.update_renderer()
#     viewer.run()
#     viewer.destroy_window()


def view_hand_matplotlib_free_axes(joints_xyz, parents = MPIIHandJoints.parents):
    joints_xyz = np.array(joints_xyz)
    bones = []
    annotations = []

    if parents is not None:
        for i, p in enumerate(parents):
            if p is not None:
                bone = np.array([joints_xyz[p], joints_xyz[i]])
                bones.append(bone)

    for i, (x, y, z) in enumerate(joints_xyz):
        label = f'{i} ({x:.2f},{y:.2f},{z:.2f})'
        annotations.append((i, x, y, z, label))

    return {
        'joints': joints_xyz,
        'bones': bones,
        'annotations': annotations
    }



# def view_hand_matplotlib_free_axes(joints_xyz, parents=MPIIHandJoints.parents, show_axes=True, annotate=True): 
#     """
#     Plot 3D hand joints and bones with natural axes based on joint positions.

#     Args:
#         joints_xyz (np.ndarray): shape (N, 3), joint positions
#         parents (list): list of parent indices
#         show_axes (bool): whether to show XYZ axis lines
#         annotate (bool): whether to label each joint
#     """
#     joints_xyz = np.array(joints_xyz)
#     fig = plt.figure(figsize=(8, 8))
#     ax = fig.add_subplot(111, projection='3d')

#     # Plot joints
#     ax.scatter(joints_xyz[:, 0], joints_xyz[:, 1], joints_xyz[:, 2], c='red', s=40)

#     # Plot bones
#     if parents is not None:
#         for i, p in enumerate(parents):
#             if p is None:
#                 continue
#             xs = [joints_xyz[i, 0], joints_xyz[p, 0]]
#             ys = [joints_xyz[i, 1], joints_xyz[p, 1]]
#             zs = [joints_xyz[i, 2], joints_xyz[p, 2]]
#             ax.plot(xs, ys, zs, c='green')

#     # Annotate joints
#     if annotate:
#         for i, (x, y, z) in enumerate(joints_xyz):
#             ax.text(x, y, z, f'{i}\n({x:.2f},{y:.2f},{z:.2f})',
#                     fontsize=8, ha='center', va='center')

#     # Show directional XYZ arrows from origin
#     if show_axes:
#         axis_len = 0.1 * np.linalg.norm(joints_xyz.max(axis=0) - joints_xyz.min(axis=0))
#         ax.quiver(0, 0, 0, axis_len, 0, 0, color='r', linewidth=2)
#         ax.quiver(0, 0, 0, 0, axis_len, 0, color='g', linewidth=2)
#         ax.quiver(0, 0, 0, 0, 0, axis_len, color='b', linewidth=2)
#         ax.text(axis_len, 0, 0, 'X', color='r', fontsize=12)
#         ax.text(0, axis_len, 0, 'Y', color='g', fontsize=12)
#         ax.text(0, 0, axis_len, 'Z', color='b', fontsize=12)

#     ax.set_xlabel("X")
#     ax.set_ylabel("Y")
#     ax.set_zlabel("Z")
#     ax.view_init(elev=20, azim=-75)
#     ax.set_title("3D Hand Joints (Dynamic Axes)")

#     plt.tight_layout()
#     plt.show()


