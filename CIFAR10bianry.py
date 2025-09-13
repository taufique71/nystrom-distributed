import numpy as np
import argparse
from torchvision.datasets import CIFAR10

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--nuberOfSamples", type=int, help="Number of rows and columns of the matrix", default=100)

args = parser.parse_args()
nuberOfSamples = args.nuberOfSamples

print(f"Generating CIFAR10 Binary at ./data/cifar10.bin")
# Download dataset (train + test)
rawData = CIFAR10(root="./data", train=True, download=True, transform=None)
allSamples = [np.array(img) for img, _ in rawData] 
samples = np.stack(allSamples[:nuberOfSamples], axis=0)
samples = samples.reshape(nuberOfSamples, -1)  # (n, 3072)

samples = samples @ samples.T

data = samples.flatten().astype(np.float64)

with open("./data/cifar10.bin", "wb") as f:
    # then write matrix values
    data.tofile(f)