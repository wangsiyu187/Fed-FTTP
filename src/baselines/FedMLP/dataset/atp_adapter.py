"""
ATP Data Adapter: reads Fed-OCTTP's preprocessed pickle data
and exposes it in FedMLP's expected Dataset format.

Data alignment:
- Same step_2_16 client partition (from ATP partition pickle)
- Same gaussian_noise corruption on Off-site test set (from ATP corruption pickle)
- Same preprocessing: Resize(256)+ToTensor, NO ImageNet normalization
- Same model: ResNet18 from scratch (pretrained=False)
"""
import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset


def load_atp_data(corruption_path, partition_path):
    with open(corruption_path, 'rb') as f:
        data = pickle.load(f)
    with open(partition_path, 'rb') as f:
        partition = pickle.load(f)
    return data['X'], data['Y'], partition[0], partition[1]


class ATPClientDataset(Dataset):
    """Dataset for FedMLP training clients. DatasetSplit passes global indices
    directly, so __getitem__ uses idx as a direct index into X/Y."""

    def __init__(self, X, Y, indices, mode='train'):
        self.X = X
        self.Y = Y.numpy().astype(np.float32)
        self.indices = indices
        self.mode = mode
        self.targets = self.Y[indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image = self.X[idx]
        label = torch.tensor(self.Y[idx], dtype=torch.float32)

        if self.mode == 'train':
            return {
                "image_aug_1": image, "image_aug_2": image,
                "target": label, "index": idx, "image_id": str(idx),
            }
        else:
            return {
                "image": image, "target": label,
                "index": idx, "image_id": str(idx),
            }

    def __deepcopy__(self, memo):
        # Share X/Y tensors to avoid OOM (6GB+ each deepcopy)
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        result.X = self.X
        result.Y = self.Y
        result.indices = list(self.indices)
        result.mode = self.mode
        result.targets = self.targets
        return result


class ATPWrapperDataset(Dataset):
    """Dataset for globaltest evaluation. DataLoader iterates from 0..N-1,
    so __getitem__ maps via self.indices to the actual global X index."""

    def __init__(self, X, Y, indices):
        self.X = X
        self.Y = Y
        self.indices = indices
        self.targets = Y[indices].numpy().astype(np.float32)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        gidx = self.indices[idx]
        return {
            "image": self.X[gidx],
            "target": torch.tensor(self.Y[gidx], dtype=torch.float32),
            "index": idx, "image_id": str(gidx),
        }
