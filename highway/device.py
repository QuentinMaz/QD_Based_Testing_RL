import torch

"""Used throughout the code to access the same Pytorch device."""

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
