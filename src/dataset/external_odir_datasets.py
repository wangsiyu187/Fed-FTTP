
import os
from typing import List, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import pandas as pd
from sklearn.metrics import roc_auc_score


# ODIR class order (consistent with the paper)
ODIR_CLASSES = ["N", "D", "G", "C", "A", "H", "M", "O"]
NUM_ODIR_CLASSES = len(ODIR_CLASSES)


def make_single_hot(idx: int) -> torch.Tensor:
    """Generate a single-hot label of length 8"""
    y = torch.zeros(NUM_ODIR_CLASSES, dtype=torch.float32)
    if 0 <= idx < NUM_ODIR_CLASSES:
        y[idx] = 1.0
    return y


# ===============================
# 1. DDR → ODIR(N / D)
# ===============================

class DDRAsODIR(Dataset):
    """
    DDR DR_grading task, labels 0~4:
      0: no DR  → N
      1~4: DR   → D

    txt file format:
        filename.jpg grade
    Corresponding image directory:
        Place corresponding jpg files under root_dir/train / valid / test
    """

    def __init__(
        self,
        img_root: str,
        split_txt: str,
        resize: int = 256,
        transform=None,
    ):
        """
        img_root: e.g. 'REPLACE_ME/DDR-dataset/DR_grading/train'
                  or valid / test
        split_txt: path to corresponding train.txt / valid.txt / test.txt
        """
        assert os.path.isdir(img_root), f"img_root not found: {img_root}"
        assert os.path.isfile(split_txt), f"split_txt not found: {split_txt}"

        self.img_root = img_root
        self.transform = transform or T.Compose([
            T.Resize((resize, resize)),
            T.ToTensor()
        ])

        self.samples: List[Tuple[str, torch.Tensor]] = []
        with open(split_txt, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 2:
                    continue
                fname, g_str = parts
                try:
                    grade = int(g_str)
                except ValueError:
                    continue

                # Map DDR grade to ODIR labels
                if grade == 0:
                    # Normal -> N
                    y = make_single_hot(0)  # index 0: N
                else:
                    # Has DR -> D
                    y = make_single_hot(1)  # index 1: D

                self.samples.append((fname, y))

        if len(self.samples) == 0:
            raise RuntimeError(f"No valid samples parsed from: {split_txt}")

        print(f"[DDRAsODIR] root={img_root}, txt={split_txt}, samples={len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, y = self.samples[idx]
        img_path = os.path.join(self.img_root, fname)
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = Image.open(img_path).convert("RGB")
        x = self.transform(img)
        return x, y


# ===============================
# 2. REFUGE → ODIR(N / G)
# ===============================

class REFUGEAsODIR(Dataset):
    """
    REFUGE-Multirater and Renji multi-class datasets.

      - imgName         : e.g. 'REFUGE/test_dataset/Test400/T0001.jpg'
      - multimaskName   : e.g. 'Dataset-new-2/Test-400/0544/0544.jpg'
      - Local directories: Training-400 / Validation-400 / Test-400 / possibly subdirectories

    """

    def __init__(
        self,
        refuge_root: str,
        csv_path: str,
        resize: int = 256,
        transform=None,
    ):
        assert os.path.isdir(refuge_root), f"refuge_root not found: {refuge_root}"
        assert os.path.isfile(csv_path), f"csv_path not found: {csv_path}"

        self.refuge_root = refuge_root
        self.transform = transform or T.Compose([
            T.Resize((resize, resize)),
            T.ToTensor()
        ])

        df = pd.read_csv(csv_path)
        if "label" not in df.columns:
            raise ValueError(
                f"{csv_path} must contain 'label' column. "
                f"Available columns: {list(df.columns)}"
            )

        base = os.path.basename(csv_path).lower()
        if "train" in base:
            subdir = "Training-400"
        elif "val" in base:
            subdir = "Validation-400"
        else:
            subdir = "Test-400"

        self.samples: List[Tuple[str, torch.Tensor]] = []
        missing = 0
        debug_print = 5

        for _, row in df.iterrows():
            candidates = []

            # ------- Construct candidate paths from imgName -------
            if "imgName" in df.columns:
                raw_img = str(row["imgName"]).strip()
                if raw_img:
                    bn = os.path.basename(raw_img)  # e.g. T0001.jpg
                    # 1) root / subdir / T0001.jpg
                    candidates.append(os.path.join(self.refuge_root, subdir, bn))
                    # 2) root / T0001.jpg
                    candidates.append(os.path.join(self.refuge_root, bn))

            # ------- Construct candidate paths from multimaskName -------
            if "multimaskName" in df.columns:
                raw_mm = str(row["multimaskName"]).strip()
                if raw_mm:
                    rel = raw_mm.lstrip("./")  # Dataset-new-2/Test-400/0544/0544.jpg
                    # 3) root / Dataset-new-2/Test-400/0544/0544.jpg
                    candidates.append(os.path.join(self.refuge_root, rel))
                    # 4) Strip the first directory prefix, e.g. Dataset-new-2/
                    if "/" in rel:
                        rel2 = rel.split("/", 1)[1]  # Test-400/0544/0544.jpg
                        candidates.append(os.path.join(self.refuge_root, rel2))

            # ------- Pick the first path that actually exists -------
            img_path = None
            for c in candidates:
                if os.path.isfile(c):
                    img_path = c
                    break

            if img_path is None:
                missing += 1
                continue

            label_val = int(row["label"])
            if label_val == 0:
                y = make_single_hot(0)  # N
            else:
                y = make_single_hot(2)  # G

            self.samples.append((img_path, y))

            if debug_print > 0:
                print("[REFUGEAsODIR] example path:", img_path)
                debug_print -= 1

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No valid samples parsed from: {csv_path}. "
                f"Check your paths and folder structure."
            )

        if missing > 0:
            print(f"[REFUGEAsODIR] Warning: {missing} samples skipped due to missing files.")

        print(f"[REFUGEAsODIR] root={refuge_root}, csv={csv_path}, "
              f"subdir_guess={subdir}, samples={len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, y = self.samples[idx]
        img = Image.open(img_path).convert("RGB")  # convert to 3-channel even if grayscale
        x = self.transform(img)
        return x, y


class PAPILAAsODIR(Dataset):
    def __init__(self, papila_root: str, resize: int = 256, transform=None):
        assert os.path.isdir(papila_root), f"papila_root not found: {papila_root}"

        self.papila_root = papila_root
        self.transform = transform or T.Compose([
            T.Resize((resize, resize)),
            T.ToTensor()
        ])

        # Debugging: print out the actual path we are checking
        fundus_images_dir = os.path.join(self.papila_root)
        print(f"[DEBUG] Checking FundusImages directory: {fundus_images_dir}")

        if not os.path.isdir(fundus_images_dir):
            raise RuntimeError(f"FundusImages directory not found in {papila_root}")

        self.samples: List[Tuple[str, torch.Tensor]] = []

        for fname in os.listdir(fundus_images_dir):
            if fname.endswith('.jpg'):  # Only process .jpg files
                img_path = os.path.join(fundus_images_dir, fname)
                label = 0 if 'OD' in fname else 1  # assume OD=normal, OS=abnormal
                y = make_single_hot(label)
                self.samples.append((img_path, y))

        if len(self.samples) == 0:
            raise RuntimeError(f"No valid samples found in {fundus_images_dir}.")
        print(f"[PAPILAAsODIR] samples={len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, y = self.samples[idx]
        img = Image.open(img_path).convert("RGB")  # convert to RGB format
        x = self.transform(img)
        return x, y


# ===============================
# 4. HRF → ODIR (N / D / G)
# ===============================

class HRFAsODIR(Dataset):
    """
    HRF dataset: multi-class -- N(0), D(1), G(2)
    Mapped to corresponding positions in ODIR 8-dim vector:
      N -> idx 0
      D -> idx 1
      G -> idx 2
    """

    def __init__(self, hrf_root: str, resize: int = 256, transform=None):
        assert os.path.isdir(hrf_root), f"HRF root not found: {hrf_root}"

        self.transform = transform or T.Compose([
            T.Resize((resize, resize)),
            T.ToTensor()
        ])

        self.samples = []
        img_dir = os.path.join(hrf_root, "images")
        assert os.path.isdir(img_dir), f"HRF images folder not found: {img_dir}"

        for fname in os.listdir(img_dir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".tif")):
                continue
            f_lower = fname.lower()

            if "_h" in f_lower or "healthy" in f_lower:
                y = make_single_hot(0)       # N
            elif "_dr" in f_lower:
                y = make_single_hot(1)       # D
            elif "_g" in f_lower:
                y = make_single_hot(2)       # G
            else:
                continue

            img_path = os.path.join(img_dir, fname)
            self.samples.append((img_path, y))

        if len(self.samples) == 0:
            raise RuntimeError("No HRF samples found!")

        print(f"[HRFAsODIR] samples={len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, y = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        x = self.transform(img)
        return x, y


# ===============================
# 5. Bajwa → ODIR (N / C / G / O)
# ===============================

BAJWA_CLASSES = ["N", "C", "G", "O"]
NUM_BAJWA_CLASSES = len(BAJWA_CLASSES)

def make_bajwa_onehot(idx: int) -> torch.Tensor:
    """Generate a one-hot label of length 4 for Bajwa (N, C, G, O)"""
    y = torch.zeros(NUM_BAJWA_CLASSES, dtype=torch.float32)
    if 0 <= idx < NUM_BAJWA_CLASSES:
        y[idx] = 1.0
    return y


class BajwaAsODIR(Dataset):
    """
    Bajwa hospital multi-eye disease dataset:
        1_normal          → N (0)
        2_cataract        → C (1)
        2_glaucoma        → G (2)
        3_retina_disease  → O (3)

    No longer uses ODIR 8-dim encoding,
    but directly uses 4-dim one-hot: [N, C, G, O].
    """

    def __init__(self, bajwa_root: str, resize: int = 256, transform=None):
        assert os.path.isdir(bajwa_root), f"Bajwa root not found: {bajwa_root}"

        self.transform = transform or T.Compose([
            T.Resize((resize, resize)),
            T.ToTensor()
        ])

        base = os.path.join(bajwa_root, "Eye_diseases_dataset")
        assert os.path.isdir(base), f"Eye_diseases_dataset not found: {base}"

        self.samples: List[Tuple[str, torch.Tensor]] = []

        # Map to 4-dim internal indices: 0=N, 1=C, 2=G, 3=O
        mapping = {
            "1_normal": 0,          # N
            "2_cataract": 1,        # C
            "2_glaucoma": 2,        # G
            "3_retina_disease": 3,  # O
        }

        for folder, idx4 in mapping.items():
            cls_dir = os.path.join(base, folder)
            if not os.path.isdir(cls_dir):
                print(f"[BajwaAsODIR] WARN: folder not found: {cls_dir}")
                continue

            for fname in os.listdir(cls_dir):
                if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue
                img_path = os.path.join(cls_dir, fname)
                y = make_bajwa_onehot(idx4)   # 4-dim one-hot
                self.samples.append((img_path, y))

        if len(self.samples) == 0:
            raise RuntimeError("No Bajwa samples found!")

        print(f"[BajwaAsODIR] samples={len(self.samples)} (labels dim = 4)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, y = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        x = self.transform(img)
        return x, y
