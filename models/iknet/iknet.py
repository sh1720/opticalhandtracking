import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat

# -------------------------------
# IKNet Network
# -------------------------------

class DenseBN(nn.Module):
    def __init__(self, inc, ouc):
        super().__init__()
        self.dense = nn.Sequential(
            nn.Linear(inc, ouc, bias=True),
            nn.BatchNorm1d(ouc),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.dense(x)


#TO DO - fix iknet module - need to retrain to get pose parameters
class iknet(nn.Module):
    pose = 7
    dof = 4
    min_dim = 10
    max_dim = 500
    min_dropout = 0.1
    max_dropout = 0.5

    def __init__(self, trial=None):
        super().__init__()

        self.input_dims = [400, 300, 200, 100, 50]
        self.dropout = 0.1
        if trial is not None:
            for i in range(0, 5):
                self.input_dims[i] = trial.suggest_int(
                    f"fc{i+2}_input_dim", self.min_dim, self.max_dim
                )
            self.dropout = trial.suggest_float(
                "dropout", self.min_dropout, self.max_dropout
            )


        layers = []
        input_dim = self.pose
        for output_dim in self.input_dims:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))
            input_dim = output_dim
        layers.append(nn.Linear(input_dim, self.dof))
        self.layers = nn.Sequential(*layers)


    def forward(self, x):
        return self.layers(x)

# -------------------------------
# Conversion Utility
# -------------------------------

def convert_tf_ckpt_to_pth(ckpt_path: str, output_pth: str = "iknet.pth"):
    """
    Converts a TensorFlow .ckpt file to a PyTorch .pth file for IKNet.
    """
    import tensorflow as tf

    print(f"Reading TensorFlow checkpoint from: {ckpt_path}")
    reader = tf.train.load_checkpoint(ckpt_path)
    variable_map = reader.get_variable_to_dtype_map()

    tf_weights = {}
    for key in variable_map:
        if "Adam" in key or "global_step" in key:
            continue
        tf_weights[key] = reader.get_tensor(key)

    # Map TensorFlow keys to PyTorch keys
    pt_weights = {}
    for key, value in tf_weights.items():
        pt_key = key.replace("/", ".")
        pt_key = pt_key.replace("kernel", "weight").replace("bias", "bias")
        pt_key = pt_key.replace("moving_mean", "running_mean").replace("moving_variance", "running_var")
        pt_key = pt_key.replace("batch_normalization", "dense.1")

        if "weight" in pt_key and value.ndim == 2:
            value = rearrange(value, "i o -> o i")

        pt_weights[pt_key] = torch.from_numpy(np.array(value))


    # Load into IKNet
    model = iknet()
    missing, unexpected = model.load_state_dict(pt_weights, strict=False)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    torch.save(model.state_dict(), output_pth)
    print(f"✅ IKNet PyTorch weights saved to: {output_pth}")

# -------------------------------
# Optional CLI Usage
# -------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True, help='Path to TensorFlow .ckpt file')
    parser.add_argument('--out', type=str, default='iknet.pth', help='Output .pth file path')
    args = parser.parse_args()

    convert_tf_ckpt_to_pth(args.ckpt, args.out)
