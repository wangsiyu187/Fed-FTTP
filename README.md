# 🩺 Fed-FTTP: Federated Fundus Test-Time Personalization

## 📝 Project Overview

This project implements **Fed-FTTP**, a federated learning framework for cross-site multi-label ocular disease recognition. It addresses three core challenges in medical federated learning: class imbalance via **Robust Asymmetric Loss (RAL)**, label shift via **Bias-Corrected Vector Scaling (BVS)**, and domain shift via **Test-Time Personalization (ATP)**. Built on PyTorch, it supports multiple FL algorithms, TTA methods, calibration techniques, and cross-dataset evaluation.

## 🚀 Key Features

- **Federated Learning Algorithms**: FedAvg, FedProx, FedBN, SCAFFOLD, PerFedAvg
- **Test-Time Adaptation (TTA)**: TENT, T3A, SHOT, MEMO, BatchNorm, ATP
- **Label Shift Estimation**: EM (Expectation-Maximization), BBSE (Black-Box Shift Estimation)
- **Calibration Methods**: BVS (Bias-Corrected Vector Scaling), BCTS, VS, TS
- **Loss Functions**: RAL (Robust Asymmetric Loss) with polynomial focusing and Hill-based truncation
- **Model Zoo**: ResNet18/50 (single & multi-label), ML-GCN (GCN-augmented), CNN
- **Multi-Label Metrics**: Cohen's Kappa, F1, AUC, bACC, Final (Kappa + F1 + AUC)
- **Cross-Dataset Evaluation**: OIA-ODIR → REFUGE, DDR, Bajwa, PAPILA, HRF

## 🛠️ Environment Requirements

- Python 3.8+
- PyTorch ≥ 1.9.0
- CUDA (optional, recommended for training)

```bash
pip install -r requirements.txt
```

| Package       | Version | Purpose                    |
|---------------|---------|----------------------------|
| torch         | ≥ 1.9.0 | Deep learning framework    |
| torchvision   | ≥ 0.10  | Vision models & transforms |
| numpy         | ≥ 1.19  | Numerical computation      |
| scipy         | ≥ 1.6   | Scientific computing       |
| scikit-learn  | ≥ 0.24  | Metrics (AUC, etc.)        |
| pandas        | ≥ 1.2   | Data loading & processing  |
| pillow        | ≥ 8.0   | Image I/O                  |
| tqdm          | ≥ 4.50  | Progress bars              |
| matplotlib    | ≥ 3.3   | Visualization              |
| pyyaml        | ≥ 5.4   | Configuration file parsing |

## 📦 Data Preparation

1. Download the [OIA-ODIR dataset](https://odir2019.grand-challenge.org/dataset/)
2. Organize images as follows:

```
data/
├── OIA-ODIR_dataset_multi/
│   └── RGB_preprocessed/
│       ├── Training Set/
│       ├── On-site Test Set/
│       └── Off-site Test Set/
└── oia_odir/
    ├── train_labels.txt
    └── mlgcn_adj_odir_multi.npy
```

3. Copy and edit the configuration:

```bash
cp src/config.yaml.template src/config.yaml
# Edit src/config.yaml with your data paths
```

See [DATA.md](DATA.md) for detailed instructions.

## ⚡ Quick Start

### 1. Prepare Data Partitions

```bash
bash experiments/odir/hybrid/data_prepare.sh
```

### 2. Pretrain a Model

```bash
bash experiments/odir/hybrid/pretrain_fedavg_resnet18.sh
```

### 3. Run ATP Training (Test-Time Personalization)

```bash
bash experiments/odir/hybrid/atp_train_resnet18.sh
```

### 4. Evaluate

```bash
bash experiments/odir/hybrid/atp_test_resnet18.sh
```

### 5. Cross-Dataset Evaluation (optional)

```bash
python src/scripts/eval_external.py        # Direct evaluation
python src/scripts/eval_external_ttp.py    # With TTP adaptation
```

## 🙏 Acknowledgements

- [PyTorch](https://pytorch.org/)
- [FedLab](https://fedlab.readthedocs.io/)
- [ML-GCN](https://github.com/Megvii-Nanjing/ML-GCN)
- [OIA-ODIR](https://odir2019.grand-challenge.org/)
- [TENT](https://github.com/DequanWang/tent)
- [SHOT](https://github.com/tim-learn/SHOT)

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
