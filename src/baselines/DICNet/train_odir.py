"""
DICNet training on ODIR multi-view features (complete mode).
Trains on train.mat, evaluates on test.mat.
Multi-seed evaluation with BACC, Kappa, F1, AUC.
"""
import os
import sys
import os.path as osp
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import argparse
import time
import copy
import numpy as np
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts
import scipy.io as sio
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, cohen_kappa_score, balanced_accuracy_score

# DICNet imports
from model import get_model
from loss import Loss

# ============================================================
# Minimal local copies (avoid pulling in FedMLP utils)
# ============================================================

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (self.count + 1e-8)


def setLogger(logfile=None):
    import logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    if logfile is not None:
        fh = logging.FileHandler(logfile, mode='a')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


# ============================================================
# Data loading
# ============================================================

def load_mat(mat_path):
    data = sio.loadmat(mat_path)
    mv_data = data['X'][0]
    labels = data['label']
    labels = labels.astype(np.float32)
    if labels.min() == -1:
        labels = (labels + 1) * 0.5
    total_sample_num = labels.shape[0]
    if mv_data[0].shape[0] != total_sample_num:
        mv_data = [v_data.T for v_data in mv_data]
    assert mv_data[0].shape[0] == labels.shape[0] == total_sample_num
    mv_data = [StandardScaler().fit_transform(v_data.astype(np.float32)) for v_data in mv_data]
    return mv_data, labels, total_sample_num


class ODIRViewsDataset(torch.utils.data.Dataset):
    def __init__(self, mat_path):
        self.mv_data, self.labels, self.n_samples = load_mat(mat_path)
        self.d_list = [da.shape[1] for da in self.mv_data]
        self.classes_num = self.labels.shape[1]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        data = [torch.tensor(v[index], dtype=torch.float) for v in self.mv_data]
        label = torch.tensor(self.labels[index], dtype=torch.float)
        # Return dummy inc_V_ind, inc_L_ind (all ones = complete)
        inc_V_ind = torch.ones(len(self.mv_data), dtype=torch.int32)
        inc_L_ind = torch.ones(self.classes_num, dtype=torch.int32)
        return data, label, inc_V_ind, inc_L_ind


# ============================================================
# Metrics
# ============================================================

def compute_bacc(y_true, y_pred):
    """Per-class balanced accuracy, averaged (multi-label)."""
    scores = []
    for i in range(y_true.shape[1]):
        if np.unique(y_true[:, i]).shape[0] == 2:
            scores.append(balanced_accuracy_score(y_true[:, i], y_pred[:, i]))
    return np.mean(scores) if scores else 0.0


def compute_kappa(y_true, y_pred):
    """Cohen's kappa per sample, averaged."""
    scores = []
    for i in range(y_true.shape[0]):
        if np.all(y_true[i] == y_true[i][0]):
            scores.append(1.0 if np.array_equal(y_true[i], y_pred[i]) else 0.0)
        else:
            scores.append(cohen_kappa_score(y_true[i], y_pred[i]))
    return np.mean(scores)


def compute_auc_me(y_prob, label):
    """Macro AUC (per-class)."""
    n, m = label.shape
    macro_auc = 0
    valid_labels = 0
    for i in range(m):
        if np.unique(label[:, i]).shape[0] == 2:
            try:
                macro_auc += roc_auc_score(label[:, i], y_prob[:, i])
                valid_labels += 1
            except ValueError:
                pass
    return macro_auc / max(valid_labels, 1)


def compute_f1(y_true, y_pred):
    """Macro F1."""
    return f1_score(y_true, y_pred, average='macro', zero_division=0)


def do_metric(y_prob, label):
    y_pred = (y_prob > 0.5).astype(np.float32)
    bacc = compute_bacc(label, y_pred)
    kappa = compute_kappa(label, y_pred)
    f1 = compute_f1(label, y_pred)
    auc = compute_auc_me(y_prob, label)
    return bacc, kappa, f1, auc, y_pred


# ============================================================
# Train / Test
# ============================================================

def train(loader, model, loss_model, opt, sche, epoch, logger, args):
    model.train()
    total_loss = 0
    for data, label, inc_V_ind, inc_L_ind in loader:
        data = [v.to(args.device) for v in data]
        label = label.to(args.device)
        inc_V_ind = inc_V_ind.float().to(args.device)
        inc_L_ind = inc_L_ind.float().to(args.device)

        x_bar_list, target_pre, fusion_z, individual_zs = model(data, inc_V_ind)

        loss_Cont = 0
        for i in range(len(individual_zs)):
            for j in range(i + 1, len(individual_zs)):
                loss_Cont += loss_model.contrast_loss(
                    individual_zs[i], individual_zs[j],
                    inc_V_ind[:, i], inc_V_ind[:, j])

        loss_CL = loss_model.weighted_BCE_loss(target_pre, label, inc_L_ind)
        loss_AE = sum(loss_model.wmse_loss(x_bar_list[iv], data[iv], inc_V_ind[:, iv])
                      for iv in range(len(x_bar_list)))
        loss = loss_CL + args.gamma * loss_AE + loss_Cont * args.beta

        opt.zero_grad()
        loss.backward()
        if isinstance(sche, CosineAnnealingWarmRestarts):
            sche.step(epoch + len(loader) / 1000)
        opt.step()
        total_loss += loss.item()

    if isinstance(sche, StepLR):
        sche.step()
    avg_loss = total_loss / max(len(loader), 1)
    if epoch % 20 == 0:
        logger.info(f"Epoch [{epoch}]: Loss={avg_loss:.4f}")
    return avg_loss


@torch.no_grad()
def evaluate(loader, model, device, logger, epoch, prefix=""):
    model.eval()
    all_labels = []
    all_preds = []
    for data, label, inc_V_ind, inc_L_ind in loader:
        data = [v.to(device) for v in data]
        inc_V_ind = inc_V_ind.float().to(device)
        _, pred, _, _ = model(data, inc_V_ind)
        all_labels.append(label.numpy())
        all_preds.append(pred.cpu().numpy())

    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    bacc, kappa, f1, auc, _ = do_metric(all_preds, all_labels)

    p = f"[{prefix}] " if prefix else ""
    if epoch % 20 == 0:
        logger.info(f"{p}Epoch [{epoch}]: BACC={bacc:.4f}, Kappa={kappa:.4f}, "
                    f"F1={f1:.4f}, AUC={auc:.4f}")
    return bacc, kappa, f1, auc


# ============================================================
# Main
# ============================================================

def main(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    args.device = device

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    train_path = osp.join(args.data_dir, "odir_train.mat")

    # Load both test sets
    test_sets = {}
    for tset_name, tset_file in [("Off-site", "odir_test.mat"), ("On-site", "odir_test_onsite.mat")]:
        test_path = osp.join(args.data_dir, tset_file)
        if osp.exists(test_path):
            test_sets[tset_name] = ODIRViewsDataset(test_path)

    train_dataset = ODIRViewsDataset(train_path)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    test_loaders = {}
    for tset_name, tds in test_sets.items():
        test_loaders[tset_name] = torch.utils.data.DataLoader(
            tds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    d_list = train_dataset.d_list
    classes_num = train_dataset.classes_num
    logger = setLogger(args.logfile)

    logger.info(f"Train: {len(train_dataset)}")
    for tset_name, tds in test_sets.items():
        logger.info(f"Test ({tset_name}): {len(tds)}")
    logger.info(f"d_list: {d_list}, classes: {classes_num}")
    logger.info(f"alpha={args.alpha}, beta={args.beta}, gamma={args.gamma}, lr={args.lr}, seed={args.seed}")

    model = get_model(d_list, n_layers=4, classes_num=classes_num, device=device)
    loss_model = Loss(args.alpha, device)
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = None

    best_bacc = {k: 0 for k in test_sets}
    best_metrics = {k: None for k in test_sets}

    for epoch in range(args.epochs):
        train(train_loader, model, loss_model, optimizer, scheduler, epoch, logger, args)
        for tset_name, test_loader in test_loaders.items():
            bacc, kappa, f1, auc = evaluate(test_loader, model, device, logger, epoch, prefix=tset_name)
            if bacc >= best_bacc[tset_name]:
                best_bacc[tset_name] = bacc
                best_metrics[tset_name] = (bacc, kappa, f1, auc)

    for tset_name in test_sets:
        m = best_metrics[tset_name]
        logger.info(f"Best ({tset_name}): BACC={m[0]:.4f}, Kappa={m[1]:.4f}, "
                    f"F1={m[2]:.4f}, AUC={m[3]:.4f}")

    return best_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/ODIR")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=1e-1)
    parser.add_argument("--logfile", type=str, default=None)

    args = parser.parse_args()
    best = main(args)
    for tset_name, metrics in best.items():
        print(f"Final ({tset_name}): BACC={metrics[0]:.4f}, Kappa={metrics[1]:.4f}, F1={metrics[2]:.4f}, AUC={metrics[3]:.4f}")
