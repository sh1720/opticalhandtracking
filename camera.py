import cv2
import torch
import numpy as np
from tqdm import tqdm
import open3d as o3d
import cv2
import trimesh
import time
import pyrender

from queue import Queue
from mano_utils import *
from render import *
from utils import *
from analytical_ik import adaptive_IK



def process_frame(frame, detnet, device, transform):
    timings = {}

    t0 = time.perf_counter()

    # 1. Crop image
    h, w, _ = frame.shape
    side = min(h, w)
    cx, cy = w // 2, h // 2
    crop = frame[cy - side//2:cy + side//2, cx - side//2:cx + side//2]

    t1 = time.perf_counter()
    timings['crop'] = t1 - t0

    # 2. Transform to tensor
    img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    input_tensor = transform(img_rgb).unsqueeze(0).to(device)

    t2 = time.perf_counter()
    timings['transform'] = t2 - t1

    # 3. Run DetNet
    with torch.no_grad():
        output = detnet(input_tensor)
        xyz = output['xyz']
        uv = output['uv'][0].cpu().numpy()

    t3 = time.perf_counter()
    timings['detnet'] = t3 - t2

    # 4. Prepare template
    j, _, _ = extract_mano_template()
    template = torch.tensor(j, dtype=torch.float32) / 1000
    predicted = xyz[0].detach().cpu()
    template_np = template.numpy()
    predicted_np = predicted.numpy()

    ref_idx = [0, 9]
    pred_len = np.linalg.norm(predicted_np[ref_idx[1]] - predicted_np[ref_idx[0]])
    temp_len = np.linalg.norm(template_np[ref_idx[1]] - template_np[ref_idx[0]])
    scale = temp_len / (pred_len + 1e-8)
    scaled_pred = predicted_np * scale
    translated_pred = scaled_pred - scaled_pred[0] + template_np[0]

    t4 = time.perf_counter()
    timings['normalize'] = t4 - t3

    # 5. Estimate pose with IK
    thetas, _ = adaptive_IK(template_np, translated_pred)

    t5 = time.perf_counter()
    timings['adaptive_IK'] = t5 - t4

    # 6. Process 2D/3D joint info
    xyz_keypoints = [tuple(coord.tolist()) for coord in xyz[0]]
    image_points = []
    for u, v in uv:
        x = int((v / 32) * crop.shape[1])
        y = int((u / 32) * crop.shape[0])
        image_points.append((x, y))

    t6 = time.perf_counter()
    timings['postprocess_joints'] = t6 - t5

    # 7. Build MANO mesh
    thetas = torch.tensor(thetas, dtype=torch.float32).reshape(-1, 1).to(device)
    mano = MANOHandMesh()
    betas = np.array([
        -0.8101279, -0.77720827, -1.9707527, -0.35107753, -0.64986867,
        -2.711966, 1.1160069, -0.6333117, 0.75000185, -0.4505857
    ], dtype=np.float32)
    pose_parameters = thetas.reshape(1, -1)
    betas = torch.from_numpy(betas).unsqueeze(0)
    vertices, _, faces = mano.apply_transformations_to_mano(pose_params=pose_parameters, shape_params=betas)

    t7 = time.perf_counter()
    timings['mano'] = t7 - t6
    timings['total'] = t7 - t0

    print("Timing breakdown (s):", timings)

    return crop, image_points, xyz_keypoints, vertices, faces


def run_webcam(detnet, device, transform):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return


    viewer = o3d.visualization.Visualizer()
    viewer.create_window(window_name='MANO Mesh', width=640, height=480, visible=True)
    viewer.run()

    frame_counter = 0
    while True:
        start_total = time.perf_counter()

        viewer.clear_geometries()
        start_read = time.perf_counter()
        ret, frame = cap.read()
        end_read = time.perf_counter()
        if not ret:
            break

        start_process = time.perf_counter()
        crop, uv_pts, xyz_pts, vertices, faces = process_frame(frame, detnet, device, transform)
        end_process = time.perf_counter()

        start_draw = time.perf_counter()
        image = draw_hand_keypoints(crop.copy(), uv_pts, xyz_pts)
        cv2.imshow("Image", image)
        end_draw = time.perf_counter()

        vertices = np.asarray(vertices)
        vertices[:, [0, 2]] *= -1

        start_mesh = time.perf_counter()
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices * -1)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.paint_uniform_color([228 / 255, 178 / 255, 148 / 255])
        mesh.compute_vertex_normals()
        viewer.add_geometry(mesh)
        viewer.update_geometry(mesh)
        viewer.poll_events()
        viewer.update_renderer()
        end_mesh = time.perf_counter()

        end_total = time.perf_counter()

        tqdm.write(f"[Frame {frame_counter}] "
                   f"Read: {(end_read - start_read)*1000:.1f}ms | "
                   f"Process: {(end_process - start_process)*1000:.1f}ms | "
                   f"Draw: {(end_draw - start_draw)*1000:.1f}ms | "
                   f"Mesh: {(end_mesh - start_mesh)*1000:.1f}ms | "
                   f"Total: {(end_total - start_total)*1000:.1f}ms")

        key = cv2.waitKey(1)
        if key == ord('q'):
            viewer.destroy_window()
            break

        frame_counter += 1

    cap.release()
    cv2.destroyAllWindows()



def run_video(video_path, detnet, device, transform):
    """ 
    Processes a pre-recorded video file using a hand keypoint detector.

    Parameters: 
    ----------
        video_path (str): Path to the video file.
        detnet: Neural network for keypoint detection.
        device: Torch device (e.g., 'cpu' or 'cuda').
        transform: Image transformation function (e.g., normalization, resize).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Could not open video:", video_path)
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process each frame: crop image, predict keypoints, estimate pose
        crop, uv_pts, xyz_pts, vertices, faces = process_frame(frame, detnet, device, transform)


        image = draw_hand_keypoints(crop.copy(), uv_pts, xyz_pts)


        cv2.imshow("Image", image)
        cv2.waitKey(1)

        
        # render_mesh = model_to_mano_window(vertices, faces)

        mesh = trimesh.Trimesh(vertices, faces, process = False)
        scene = pyrender.Scene() 
        scene.add(mesh)
        pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=True)


def run_image(image_path, detnet, device, transform):
    """Runs inference on a single image and displays 2D keypoints and 3D mesh."""

    # Load image
    frame = cv2.imread(image_path)
    if frame is None:
        print("Could not read image:", image_path)
        return

    # Run detection and get hand mesh data
    crop, uv_pts, xyz_pts, vertices, faces = process_frame(frame, detnet, device, transform)

    # 2D keypoints overlay
    image = draw_hand_keypoints(crop.copy(), uv_pts, xyz_pts)

    # --- OpenCV display (2D) ---
    cv2.imshow("DetNet - 2D Keypoints", image)

    # --- Open3D display (3D mesh) ---
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.paint_uniform_color([228 / 255, 178 / 255, 148 / 255])
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()

    viewer = o3d.visualization.Visualizer()
    viewer.create_window(window_name='MANO Mesh', width=640, height=480, visible=True)
    viewer.add_geometry(mesh)

    # Render once (non-blocking)
    viewer.update_geometry(mesh)
    viewer.poll_events()
    viewer.update_renderer()

    # Wait for key press to close OpenCV window
    print("Press any key in the image window to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    viewer.destroy_window()