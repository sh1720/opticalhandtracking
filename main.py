# main.py
import torch
import torchvision.transforms as transforms
import argparse
from models.detnet.detnet import detnet
from camera import run_webcam, run_video, run_image
from mano_utils import *

# Command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, choices=['webcam', 'video', 'image'], default='webcam')
parser.add_argument('--source', type=str, help="Path to video or image (required for video/image mode)")
args = parser.parse_args()

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load DetNet
DetNet = detnet().to(device)
detnet_ckpt = torch.load('networks/detnet/ckp_detnet_68.pth', map_location=device)
DetNet.load_state_dict(detnet_ckpt)
DetNet.eval()

# Transform for image input
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# Run the selected mode
if args.mode == 'webcam':
    run_webcam(DetNet,  device, transform)
elif args.mode == 'video':
    if not args.source:
        print("Error: --source must be specified for video mode.")
    else:
        run_video(args.source, DetNet,  device, transform)
elif args.mode == 'image':
    if not args.source:
        print("Error: --source must be specified for image mode.")
    else:
        run_image(args.source, DetNet,  device, transform)

