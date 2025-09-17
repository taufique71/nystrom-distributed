import numpy as np
import argparse
from torchvision.datasets import CIFAR10

parser = argparse.ArgumentParser()
parser.add_argument("-path", "--path", type=str, help="Path to save binary file", default="./data")

args = parser.parse_args()

# Download dataset (train + test)
rawData = CIFAR10(root=args.path, train=True, download=True, transform=None)
allSamples = [np.array(img) for img, _ in rawData] 
samples = np.stack(allSamples, axis=0)
samples = samples.reshape(-1, 3072)  # (n, 3072)
print(f"Shape of Input matrix: {samples.shape}")

samples = samples @ samples.T

data = samples.flatten().astype(np.float64)

with open(args.path+"/cifar10.bin", "wb") as f:
    # then write matrix values
    data.tofile(f)