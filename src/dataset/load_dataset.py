# -*- coding: utf-8 -*-
# @Description: load_dataset.py
# @Author: Yanhan Hu
# @Date: 2025-09-30
# @LastEditTime: 2025-09-30

from torchvision import transforms
from torch.utils.data import Dataset
import os
from PIL import Image
import torch
import yaml
from glob import glob
import pandas as pd
import numpy as np

# Load parameters
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)


# ========================= ODIR Dataset Loader =========================
DEFAULT_ODIR_CLASSES = ['A','C','D','G','H','M','N','O']

def _scan_dir_to_records(data_dir, class_names):
    """
    Scan images organized by class_name subfolders under data_dir,
    Return record list: [{"image_id":..., "path":..., "label":..., "label_idx":...}, ...]
    """
    records = []
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    for cls in class_names:
        cls_dir = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        patterns = [os.path.join(cls_dir, '*.jpg'),
                    os.path.join(cls_dir, '*.jpeg'),
                    os.path.join(cls_dir, '*.png')]
        img_paths = []
        for p in patterns:
            img_paths.extend(glob(p))
        for p in img_paths:
            image_id = os.path.splitext(os.path.basename(p))[0]
            records.append({
                "image_id": image_id,
                "path": p,
                "label": cls,
                "label_idx": class_to_idx[cls]
            })
    return records

def load_odir(train_dir, onsite_test_dir, offsite_test_dir, class_names=None):
    """
    Load ODIR dataset from two separate paths (train and test)
    Returns (train_df, test_df), both with columns: ['image_id', 'path', 'label', 'label_idx']
    train_dir and onsite_test_dir offsite_test_dir should be organized by class subfolders (e.g. train_dir/A/*.jpg)
    """
    class_names = class_names or DEFAULT_ODIR_CLASSES

    train_records = _scan_dir_to_records(train_dir, class_names)
    if len(train_records) == 0:
        raise RuntimeError(f"No training images found under {train_dir}. Check directory structure.")

    onsite_test_records = _scan_dir_to_records(onsite_test_dir, class_names)
    if len(onsite_test_records) == 0:
        raise RuntimeError(f"No testing images found under {test_dir}. Check directory structure.")

    offsite_test_records = _scan_dir_to_records(offsite_test_dir, class_names)
    if len(offsite_test_records) == 0:
        raise RuntimeError(f"No testing images found under {test_dir}. Check directory structure.")

    train_df = pd.DataFrame(train_records)
    onsite_test_df = pd.DataFrame(onsite_test_records)
    offsite_test_df = pd.DataFrame(offsite_test_records)

    return train_df.reset_index(drop=True), onsite_test_df.reset_index(drop=True), offsite_test_df.reset_index(drop=True)

# ========================= ODIR Dataset Loader =========================

class ODIRDataFrame(Dataset):
    """
    DataFrame -> Dataset wrapper for compatibility with upstream code.
    - df: must contain 'path' and 'label' or 'label_idx' columns
    - return_multihot: if True, returns multi-hot labels (float tensor); else returns integer label (torch.long)
    - self.targets is set as integer list for partitioner
    """
    def __init__(self, df, transform=None, class_names=None, return_multihot=False):
        self.df = df.reset_index(drop=True).copy()
        self.transform = transform
        self.class_names = class_names or DEFAULT_ODIR_CLASSES
        self.class_to_idx = {c:i for i,c in enumerate(self.class_names)}
        if 'label_idx' in self.df.columns:
            self.targets = [int(x) for x in self.df['label_idx'].tolist()]
        else:
            self.targets = [int(self.class_to_idx[l]) for l in self.df['label'].tolist()]

        self.return_multihot = return_multihot
        self.num_classes = len(self.class_names)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['path']).convert('RGB')
        if self.transform:
            img = self.transform(img)

        label_idx = int(self.targets[idx])
        if self.return_multihot:
            lab = torch.zeros(self.num_classes, dtype=torch.float32)
            lab[label_idx] = 1.0
            labels = lab
        else:
            labels = torch.tensor(label_idx, dtype=torch.long)

        # return {'image': img, 'labels': labels}
        return img, labels

def get_transforms():
    """Get data augmentation and preprocessing transforms"""
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Pad(3),
        transforms.RandomRotation(10),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ])
    
    test_transforms = transforms.Compose([
        transforms.Pad(3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    
    return train_transforms, test_transforms

# ========================= Dataset split function using fedlab =========================
import os
import torch
from torch.utils.data import Dataset, Subset
from fedlab.utils.dataset.partition import CIFAR10Partitioner
import pickle

def federated_dataset_split(dataset, 
                            num_clients=5, 
                            partition_mode="iid", 
                            balance=None, 
                            dir_alpha=0.3, 
                            num_shards=None, 
                            unbalance_sgm=0.5, 
                            seed=42,
                            save_dir="splits",
                            load_existing=True,
                            local_test_ratio=0.1):
    """
    General federated dataset split function, supports saving and loading split results
    
    Args:
        dataset (Dataset): Any Dataset, as long as __getitem__ returns dict or (image, label)
        num_clients (int): number of clients
        partition_mode (str): split strategy, options: ["iid", "dirichlet", "shards"]
        balance (bool): whether each client has the same number of samples
        dir_alpha (float): Dirichlet distribution alpha
        num_shards (int): for shards partition
        unbalance_sgm (float): sample number difference for unbalanced split
        seed (int): random seed
        save_dir (str): directory to save split results
        load_existing (bool): whether to try loading existing split
    Returns:
        client_datasets (list of Subset): Dataset for each client
        client_dict (dict): client_id -> sample indices
    """
    os.makedirs(save_dir, exist_ok=True)
    split_file = os.path.join(save_dir, f"split_{partition_mode}_clients{num_clients}_alpha{dir_alpha}.pkl")

    if load_existing and os.path.exists(split_file):
        print(f"Loading existing split: {split_file}")
        with open(split_file, "rb") as f:
            client_dict = pickle.load(f)
    else:
        print(f"Generating new split: {partition_mode}")
        sample0 = dataset[0]
        if isinstance(sample0, dict):
            targets = [dataset[i]['labels'] for i in range(len(dataset))]
        else:
            targets = [dataset[i][1] for i in range(len(dataset))]

        if partition_mode == "iid":
            fed_partitioner = CIFAR10Partitioner(
                targets,
                num_clients=num_clients,
                balance=balance,
                partition="iid",
                unbalance_sgm=unbalance_sgm,
                seed=seed
            )
        elif partition_mode == "dirichlet":
            fed_partitioner = CIFAR10Partitioner(
                targets,
                num_clients=num_clients,
                balance=balance,
                partition="dirichlet",
                dir_alpha=dir_alpha,
                unbalance_sgm=unbalance_sgm,
                seed=seed
            )
        elif partition_mode == "shards":
            if num_shards is None:
                raise ValueError("num_shards must be set when using shards partition mode")
            fed_partitioner = CIFAR10Partitioner(
                targets,
                num_clients=num_clients,
                balance=balance,
                partition="shards",
                num_shards=num_shards,
                seed=seed
            )
        else:
            raise ValueError(f"Unsupported partition_mode: {partition_mode}")

        client_dict = fed_partitioner.client_dict

        with open(split_file, "wb") as f:
            pickle.dump(client_dict, f)
        print(f"Split result saved to {split_file}")

    rng = np.random.RandomState(seed)
    client_train_datasets, client_test_datasets = [], []
    for cid in range(num_clients):
        indices = client_dict[cid]
        rng.shuffle(indices)

        split_point = int(len(indices) * (1 - local_test_ratio))
        train_idx, test_idx = indices[:split_point], indices[split_point:]

        client_train_datasets.append(Subset(dataset, train_idx))
        client_test_datasets.append(Subset(dataset, test_idx))

    return client_train_datasets, client_test_datasets, client_dict
# ========================= Dataset split function using fedlab =========================