import os
import numpy as np
from PIL import Image
import pandas as pd

import torch
from torch.utils.data import Dataset


class ChestXray14(Dataset):
    def __init__(self, datapath, mode, transform=None):
        self.datapath = datapath
        self.mode = mode
        self.transform = transform

        assert self.mode in ["train", "test"]
        csv_file = os.path.join("/home/szb/multilabel/", self.mode + "_dataset_8class.csv")
        self.file = pd.read_csv(csv_file)

        self.image_list = self.file["Image Index"].values
        self.targets = self.file.iloc[0:, 1:].values.astype(np.float32)

    def __getitem__(self, index: int):
        image_id, target = self.image_list[index], self.targets[index]
        image = self.read_image(image_id)

        if self.transform is not None:
            if isinstance(self.transform, tuple):
                image1 = self.transform[0](image)
                image2 = self.transform[1](image)
                return {"image_aug_1": image1,
                        "image_aug_2": image2,
                        "target": target,
                        "index": index,
                        "image_id": image_id}
            else:
                image = self.transform(image)
                return {"image": image,
                        "target": target,
                        "index": index,
                        "image_id": image_id}

    def __len__(self):
        return len(self.targets)

    def read_image(self, image_id):
        image_path = os.path.join("/home/szb/ChestXray14/images/image/", image_id)
        image = Image.open(image_path).convert("RGB")
        return image


class ODIR(Dataset):
    """ODIR multi-label dataset for FedMLP.
    Reads images from a flat directory + label txt file.
    """
    def __init__(self, root_dir, label_txt, mode, transform=None, corruption_std=0.0):
        self.root_dir = root_dir
        self.label_txt = label_txt
        self.mode = mode
        self.transform = transform
        self.corruption_std = corruption_std

        assert self.mode in ["train", "test_offsite", "test_onsite"]
        if self.mode == "train":
            subdir = "Training Set"
        elif self.mode == "test_offsite":
            subdir = "Off-site Test Set"
        else:
            subdir = "On-site Test Set"
        self.img_dir = os.path.join(root_dir, subdir)

        # Read label mapping: {filename: 8-dim binary vector}
        label_map = {}
        with open(label_txt, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 9:  # filename + 8 labels
                    continue
                fname = parts[0]
                vec = np.array([int(x) for x in parts[1:]], dtype=np.float32)
                label_map[fname] = vec

        # Match images in directory with labels
        self.image_list = []
        self.targets = []
        for fname in sorted(os.listdir(self.img_dir)):
            if fname in label_map:
                self.image_list.append(fname)
                self.targets.append(label_map[fname])

        self.targets = np.array(self.targets, dtype=np.float32)
        print(f"[ODIR] {mode}: {len(self.image_list)} samples, {self.targets.shape[1]} classes")

    def __getitem__(self, index: int):
        image_id = self.image_list[index]
        target = self.targets[index]
        image_path = os.path.join(self.img_dir, image_id)
        image = Image.open(image_path).convert("RGB")

        # Apply Gaussian noise corruption (same as ATP: std=0.1)
        if self.corruption_std > 0 and self.mode.startswith("test"):
            np_img = np.array(image).astype(np.float32) / 255.0
            noise = np.random.normal(0, self.corruption_std, np_img.shape)
            np_img = np.clip(np_img + noise, 0, 1)
            image = Image.fromarray((np_img * 255).astype(np.uint8))

        if self.transform is not None:
            if isinstance(self.transform, tuple):
                image1 = self.transform[0](image)
                image2 = self.transform[1](image)
                return {"image_aug_1": image1,
                        "image_aug_2": image2,
                        "target": target,
                        "index": index,
                        "image_id": image_id}
            else:
                image = self.transform(image)
                return {"image": image,
                        "target": target,
                        "index": index,
                        "image_id": image_id}

    def __len__(self):
        return len(self.targets)


class ICH(Dataset):
    def __init__(self, datapath, mode, transform=None):
        self.datapath = datapath
        self.mode = mode
        self.transform = transform

        assert self.mode in ["train", "test"]
        csv_file = os.path.join("/home/szb/ICH_stage2/ICH_stage2/", self.mode + "_dataset_ICH.csv")
        # csv_file = os.path.join("/home/szb/ICH_stage2/ICH_stage2/", self.mode + '_demo.csv')  # demo exp(5000 samples)
        self.file = pd.read_csv(csv_file)

        self.image_list = self.file["Image Index"].values
        self.targets = self.file.iloc[0:, 1:].values.astype(np.float32)

    def __getitem__(self, index: int):
        image_id, target = self.image_list[index], self.targets[index]
        image = self.read_image(image_id)
        if self.transform is not None:
            if isinstance(self.transform, tuple):
                image1 = self.transform[0](image)
                image2 = self.transform[1](image)
                return {"image_aug_1": image1,
                        "image_aug_2": image2,
                        "target": target,
                        "index": index,
                        "image_id": image_id}
            else:
                image = self.transform(image)
                return {"image": image,
                        "target": target,
                        "index": index,
                        "image_id": image_id}

    def __len__(self):
        return len(self.targets)

    def read_image(self, image_id):
        image_path = os.path.join("/home/szb/ICH_stage2/ICH_stage2/png185k_512/", image_id)
        image = Image.open(image_path).convert("RGB")
        return image
