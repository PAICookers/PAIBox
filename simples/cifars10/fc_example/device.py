import torch

from torch import nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
