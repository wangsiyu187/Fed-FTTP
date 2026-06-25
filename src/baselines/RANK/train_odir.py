"""
RANK training on ODIR ATP-aligned features (complete mode).
Same data as DICNet: ODIR_ATP/odir_train.mat + odir_test.mat
Same corruption: Gaussian noise (std=0.1) on test set (via ATP pickle features)
"""
import os, sys, argparse, time, copy
import numpy as np
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts
import scipy.io as sio
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, cohen_kappa_score, balanced_accuracy_score

from model_wf import get_model
from myloss import Loss


class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.val = self.avg = self.sum = 0; self.count = 0
    def update(self, val, n=1): self.val = val; self.sum += val * n; self.count += n; self.avg = self.sum / (self.count + 1e-8)


def setLogger(logfile=None):
    import logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    if logfile:
        fh = logging.FileHandler(logfile, mode='a'); fh.setFormatter(formatter); logger.addHandler(fh)
    ch = logging.StreamHandler(); ch.setFormatter(formatter); logger.addHandler(ch)
    return logger


def load_mat(mat_path):
    data = sio.loadmat(mat_path)
    mv_data = data['X'][0]
    labels = data['label'].astype(np.float32)
    if labels.min() == -1: labels = (labels + 1) * 0.5
    if mv_data[0].shape[0] != labels.shape[0]: mv_data = [v.T for v in mv_data]
    mv_data = [StandardScaler().fit_transform(v.astype(np.float32)) for v in mv_data]
    return mv_data, labels


class ODIRViewsDataset(torch.utils.data.Dataset):
    def __init__(self, mat_path):
        self.mv_data, self.labels = load_mat(mat_path)
        self.n_samples = self.labels.shape[0]
        self.d_list = [da.shape[1] for da in self.mv_data]
        self.classes_num = self.labels.shape[1]
        self.cur_mv_data = self.mv_data
        self.cur_labels = self.labels
        self.cur_inc_V_ind = np.ones((self.n_samples, len(self.mv_data)), dtype=np.int32)
        self.cur_inc_L_ind = np.ones((self.n_samples, self.classes_num), dtype=np.int32)

    def __len__(self): return self.n_samples
    def __getitem__(self, idx):
        data = [torch.tensor(v[idx], dtype=torch.float) for v in self.cur_mv_data]
        label = torch.tensor(self.cur_labels[idx], dtype=torch.float)
        inc_V = torch.tensor(self.cur_inc_V_ind[idx], dtype=torch.int32)
        inc_L = torch.tensor(self.cur_inc_L_ind[idx], dtype=torch.int32)
        return data, label, inc_V, inc_L


# -------- Metrics (same as DICNet) --------
def compute_bacc(y_true, y_pred):
    scores = []
    for i in range(y_true.shape[1]):
        if np.unique(y_true[:, i]).shape[0] == 2:
            scores.append(balanced_accuracy_score(y_true[:, i], y_pred[:, i]))
    return np.mean(scores) if scores else 0.0

def compute_kappa(y_true, y_pred):
    scores = []
    for i in range(y_true.shape[0]):
        if np.all(y_true[i] == y_true[i][0]):
            scores.append(1.0 if np.array_equal(y_true[i], y_pred[i]) else 0.0)
        else:
            scores.append(cohen_kappa_score(y_true[i], y_pred[i]))
    return np.mean(scores)

def compute_auc_me(y_prob, label):
    n, m = label.shape
    macro_auc = 0; valid = 0
    for i in range(m):
        if np.unique(label[:, i]).shape[0] == 2:
            try: macro_auc += roc_auc_score(label[:, i], y_prob[:, i]); valid += 1
            except ValueError: pass
    return macro_auc / max(valid, 1)

def compute_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro', zero_division=0)

def do_metric(y_prob, label):
    y_pred = (y_prob > 0.5).astype(np.float32)
    return compute_bacc(label, y_pred), compute_kappa(label, y_pred), compute_f1(label, y_pred), compute_auc_me(y_prob, label)


# -------- Train / Eval --------
def train(loader, model, loss_model, opt, sche, epoch, dep_graph, logger, args):
    model.train()
    total_loss = 0
    global all_z, all_label, all_inc_V, all_inc_L
    for i, (data, label, inc_V_ind, inc_L_ind) in enumerate(loader):
        data = [v.to(args.device) for v in data]
        label = label.to(args.device)
        inc_V_ind = inc_V_ind.float().to(args.device)
        inc_L_ind = inc_L_ind.float().to(args.device)

        x_bar_list, target_pre, fusion_z, individual_zs, individual_preds, dis_score = model(data, inc_V_ind)
        z_nvd = torch.stack(individual_zs, dim=1)

        if epoch == 0 and i == 0:
            all_z = z_nvd.clone().detach()
            all_label = label.clone()
            all_inc_V = inc_V_ind.clone()
            all_inc_L = inc_L_ind.clone()
        elif epoch == 0:
            all_z = torch.cat((all_z, z_nvd.clone().detach()), dim=0)
            all_label = torch.cat((all_label, label), dim=0)
            all_inc_V = torch.cat((all_inc_V, inc_V_ind), dim=0)
            all_inc_L = torch.cat((all_inc_L, inc_L_ind), dim=0)

        # True view confidence score
        tru_score = torch.abs((label.unsqueeze(1).mul(torch.log(individual_preds + 1e-10))
                               + (1 - label.unsqueeze(1)).mul(torch.log(1 - individual_preds + 1e-10)))
                              .mul(inc_L_ind.unsqueeze(1))).sum(-1) / inc_L_ind.unsqueeze(1).sum(-1)
        tru_score[(1 - inc_V_ind).bool()] = 1e9
        tru_score = F.softmax(-tru_score, dim=-1)
        loss_dis = torch.abs(tru_score.mul(torch.log(dis_score + 1e-10)).sum()) / dis_score.shape[0]

        loss_Gp = loss_model.grather_positive_pairs(individual_zs, inc_V_ind)
        loss_CL = loss_model.New_CE9(target_pre, label, inc_L_ind, dep_graph)

        if epoch > 0:
            loss_LG = loss_model.all_label_guide(z_nvd, label, inc_V_ind, inc_L_ind,
                                                  all_z, all_label, all_inc_V, all_inc_L, i, args.batch_size)
        else:
            loss_LG = 0

        loss_AE = sum(loss_model.wmse_loss(x_bar_list[iv], data[iv], inc_V_ind[:, iv]) for iv in range(len(x_bar_list)))
        loss = loss_CL + args.gamma * loss_AE + loss_Gp * (1 - args.beta ** epoch) + args.alpha * loss_LG + loss_dis

        opt.zero_grad()
        loss.backward()
        if isinstance(sche, CosineAnnealingWarmRestarts):
            sche.step(epoch + i / len(loader))
        opt.step()
        total_loss += loss.item()

    if isinstance(sche, StepLR): sche.step()
    if epoch % 20 == 0:
        logger.info(f"Epoch [{epoch}]: Loss={total_loss / max(len(loader), 1):.4f}")
    return total_loss


@torch.no_grad()
def evaluate(loader, model, device, logger, epoch, prefix=""):
    model.eval()
    all_labels, all_preds = [], []
    for data, label, inc_V_ind, inc_L_ind in loader:
        data = [v.to(device) for v in data]
        inc_V_ind = inc_V_ind.float().to(device)
        _, pred, _, _, _, _ = model(data, inc_V_ind)
        all_labels.append(label.numpy())
        all_preds.append(pred.cpu().numpy())
    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    bacc, kappa, f1, auc = do_metric(all_preds, all_labels)
    if epoch % 20 == 0:
        logger.info(f"[{prefix}] Epoch [{epoch}]: BACC={bacc:.4f}, Kappa={kappa:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
    return bacc, kappa, f1, auc


import torch.nn.functional as F


def main(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    args.device = device
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)

    train_ds = ODIRViewsDataset(os.path.join(args.data_dir, "odir_train.mat"))
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)

    test_sets = {}
    for tset_name, tset_file in [("Off-site", "odir_test.mat"), ("On-site", "odir_test_onsite.mat")]:
        tp = os.path.join(args.data_dir, tset_file)
        if os.path.exists(tp):
            test_sets[tset_name] = ODIRViewsDataset(tp)

    test_loaders = {k: torch.utils.data.DataLoader(v, batch_size=args.batch_size, shuffle=False, num_workers=4)
                    for k, v in test_sets.items()}

    logger = setLogger(args.logfile)
    logger.info(f"Train: {len(train_ds)}")
    for k, v in test_sets.items(): logger.info(f"Test ({k}): {len(v)}")
    logger.info(f"d_list: {train_ds.d_list}, classes: {train_ds.classes_num}")

    # Build dep_graph from training labels
    labels_t = torch.tensor(train_ds.cur_labels).float().to(device)
    dep_graph = torch.matmul(labels_t.T, labels_t)
    dep_graph = dep_graph / (torch.diag(dep_graph).unsqueeze(1) + 1e-10)
    dep_graph[dep_graph <= args.sigma] = 0.
    dep_graph.fill_diagonal_(1.)

    global all_z, all_label, all_inc_V, all_inc_L
    all_z = torch.tensor([]).to(device)
    all_label = torch.tensor([]).to(device)
    all_inc_V = torch.tensor([]).to(device)
    all_inc_L = torch.tensor([]).to(device)

    model = get_model(n_stacks=4, n_input=train_ds.d_list, n_z=args.n_z, Nlabel=train_ds.classes_num, device=device)
    loss_model = Loss(args.alpha, train_ds.classes_num, device)
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = None

    best_bacc = {k: 0 for k in test_sets}
    best_metrics = {k: None for k in test_sets}

    for epoch in range(args.epochs):
        train(train_loader, model, loss_model, optimizer, scheduler, epoch, dep_graph, logger, args)
        for tset_name, test_loader in test_loaders.items():
            bacc, kappa, f1, auc = evaluate(test_loader, model, device, logger, epoch, prefix=tset_name)
            if bacc >= best_bacc[tset_name]:
                best_bacc[tset_name] = bacc
                best_metrics[tset_name] = (bacc, kappa, f1, auc)

    for k in test_sets:
        m = best_metrics[k]
        logger.info(f"Best ({k}): BACC={m[0]:.4f}, Kappa={m[1]:.4f}, F1={m[2]:.4f}, AUC={m[3]:.4f}")
    return best_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/ODIR_ATP")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.97)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--sigma", type=float, default=0.)
    parser.add_argument("--n_z", type=int, default=512)
    parser.add_argument("--logfile", type=str, default=None)
    args = parser.parse_args()
    best = main(args)
    for k, m in best.items():
        print(f"Final ({k}): BACC={m[0]:.4f}, Kappa={m[1]:.4f}, F1={m[2]:.4f}, AUC={m[3]:.4f}")
