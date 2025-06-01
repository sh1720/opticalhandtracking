from camera import process_frame

import torch 
import numpy as np
import cv2
import open3d as o3d
import time
from models.detnet.detnet import detnet
from mano_utils import MANOHandMesh
import torchvision.transforms as transforms

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((128, 128)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
# ])

# DetNet = detnet().to(device)
# detnet_ckpt = torch.load('networks/detnet/ckp_detnet_68.pth', map_location=device)
# DetNet.load_state_dict(detnet_ckpt)
# DetNet.eval()

# img_path = r"D:\FYP Datasets\archive (HAGRID Classification)\hagrid-classification-512p\fist\0a1ef746-1db8-4a20-9fa5-fc1744e222f8.jpeg"
# frame = cv2.imread(img_path)

viewer = o3d.visualization.Visualizer()
viewer.create_window(window_name='MANO Mesh', width=640, height=480, visible=True)
viewer.run()

mano = MANOHandMesh()

while True: 
    random_shape = torch.rand(1, 10)
    root_pose = torch.tensor([[0, np.pi / 2, 0]]).repeat(1, 1)
    finger_pose = torch.zeros(1, 45)
    hand_pose = torch.cat([root_pose, finger_pose], dim=1)

    vertices, joints, faces = mano.apply_transformations_to_mano(
        pose_params=hand_pose, 
        shape_params=random_shape
    )
    # crop, uv_pts, xyz_pts, _, _, vertices, faces = process_frame(frame, DetNet, device, transform)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.paint_uniform_color([228 / 255, 178 / 255, 148 / 255])
    mesh.compute_vertex_normals()
    viewer.add_geometry(mesh)

    # Render once
    viewer.update_geometry(mesh)
    viewer.poll_events()
    viewer.update_renderer()
viewer.destroy_window()
