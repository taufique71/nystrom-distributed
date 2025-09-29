import numpy as np
import argparse
from torchvision.datasets import CIFAR10

parser = argparse.ArgumentParser()
parser.add_argument("-path", "--path", type=str, help="Path to save binary file", default="./data")
parser.add_argument("-kernel", "--kernel", type=str, help="Kernel to generate gram matrix", default="linear")

args = parser.parse_args()

# Download dataset (train + test)
rawData = CIFAR10(root=args.path, train=True, download=True, transform=None)
allSamples = [np.array(img) for img, _ in rawData] 
A_Mat = np.stack(allSamples, axis=0)
A_Mat = A_Mat.reshape(-1, 3072)  # (n, 3072)
print(f"Shape of Input matrix: {A_Mat.shape}")

if args.kernel == "linear":
    A_Mat = A_Mat @ A_Mat.T
elif args.kernel == "multiquadric":
    X_norm = np.sum(A_Mat**2, axis=1).reshape(-1, 1)
    sq_dists = X_norm + X_norm.T - 2 * A_Mat @ A_Mat.T
    sq_dists = np.maximum(sq_dists, 0)  # numerical stability
    A_Mat = np.sqrt(sq_dists)

data = A_Mat.flatten().astype(np.float64)

with open(args.path+"/cifar10.bin", "wb") as f:
    # then write matrix values
    data.tofile(f)