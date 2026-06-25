# /src/dataset/odir_multilabel.py
import os
from typing import List, Tuple, Dict
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

NUM_CLASSES = 8
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

def _read_labels(txt_path: str) -> Dict[str, torch.Tensor]:
    """
    Read label txt file.
    Input format described in DATA.md.
    Returns: { filename: (8,) float32 tensor }
    """
    mapping = {}
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 1 + NUM_CLASSES:
                continue
            fname = parts[0]
            vec = torch.tensor([int(x) for x in parts[1:]], dtype=torch.float32)
            if vec.numel() == NUM_CLASSES:
                mapping[fname] = vec
    return mapping

class ODIRMultiLabelFlat(Dataset):
    """
    Build a multi-label dataset from “flat image directory + global label txt”.
    Only keeps files that exist in both the directory and the label txt.
    """
    def __init__(self, root_dir: str, label_txt: str, transform=None):
        assert os.path.isdir(root_dir), f"not dir: {root_dir}"
        assert os.path.isfile(label_txt), f"not file: {label_txt}"
        self.root_dir = root_dir
        self.lbl_map = _read_labels(label_txt)
        self.transform = transform or T.Compose([
            T.Resize((256, 256)),
            T.ToTensor()
        ])
        files = [f for f in os.listdir(root_dir) if f.lower().endswith(IMG_EXTS)]
        files.sort()
        self.samples: List[Tuple[str, torch.Tensor]] = []
        miss = 0
        for f in files:
            if f in self.lbl_map:
                self.samples.append((f, self.lbl_map[f]))
            else:
                miss += 1
        if miss > 0:
            print(f"[ODIRMultiLabelFlat] WARN: {miss} files in '{root_dir}' not found in txt.")
        print(f"[ODIRMultiLabelFlat] {root_dir}: kept {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, y = self.samples[idx]
        path = os.path.join(self.root_dir, fname)
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        return x, y
