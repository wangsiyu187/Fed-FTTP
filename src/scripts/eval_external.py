
import os
import torch
from torch.utils.data import DataLoader

from options import args_parser
from model import create_model, create_loss

from dataset.external_odir_datasets import (
    DDRAsODIR,
    REFUGEAsODIR,
    PAPILAAsODIR,
    HRFAsODIR,
    BajwaAsODIR,
    ODIR_CLASSES,  
)

from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_auc_score
import numpy as np
@torch.no_grad()
def eval_binary_subset(model, loader, criterion, device, pos_idx, neg_idx):
    """
    Evaluate binary classification using only (neg_idx, pos_idx) channels from multi-label output.
    pos_idx: disease class (e.g., D or G)
    neg_idx: normal class N
    """
    model.eval()
    total_loss = 0.0
    total_n = 0

    tp = tn = fp = fn = 0
    all_probs = []
    all_labels = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_n += bs

        # Multi-label -> probability
        probs = torch.sigmoid(logits)

        # Only take N / disease dimensions
        p_pos = probs[:, pos_idx]   # disease
        p_neg = probs[:, neg_idx]   # normal

        # Prediction: pick the larger (equivalent to threshold 0.5 + paired comparison)
        pred = (p_pos >= p_neg).long()   # 1: disease, 0: normal

        # Ground truth: which of N / disease channels is 1
        y_pos = y[:, pos_idx]
        y_neg = y[:, neg_idx]
        true = (y_pos > y_neg).long()    # 1: disease, 0: normal

        tp += ((pred == 1) & (true == 1)).sum().item()
        tn += ((pred == 0) & (true == 0)).sum().item()
        fp += ((pred == 1) & (true == 0)).sum().item()
        fn += ((pred == 0) & (true == 1)).sum().item()

        # Store probabilities and labels for AUC calculation
        all_probs.append(probs[:, pos_idx].cpu().numpy())
        all_labels.append(true.cpu().numpy())

    loss = total_loss / max(total_n, 1)

    total = tp + tn + fp + fn
    if total == 0:
        acc = sens = spec = f1 = kappa = auc = final = 0.0
    else:
        acc = (tp + tn) / total
        sens = tp / max(tp + fn, 1)   # recall for disease
        spec = tn / max(tn + fp, 1)

        # F1 (for disease)
        precision = tp / max(tp + fp, 1)
        recall = sens
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        # Cohen's kappa
        p_o = acc
        p_yes_true = (tp + fn) / total
        p_yes_pred = (tp + fp) / total
        p_no_true = (tn + fp) / total
        p_no_pred = (tn + fn) / total
        p_e = p_yes_true * p_yes_pred + p_no_true * p_no_pred
        if 1 - p_e == 0:
            kappa = 0.0
        else:
            kappa = (p_o - p_e) / (1 - p_e)

        # Calculate AUC
        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)
        auc = roc_auc_score(all_labels, all_probs)

        # Calculate Final score: average of Kappa, F1, and AUC
        final = (kappa + f1 + auc) / 3.0

    return {
        "loss": loss,
        "acc": acc,
        "sens": sens,
        "spec": spec,
        "f1": f1,
        "kappa": kappa,
        "auc": auc,
        "final": final,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }

@torch.no_grad()
def eval_multiclass(model, loader, criterion, device):
    """
    Multi-class evaluation (e.g., HRF: N/D/G; Bajwa: N/C/G/O),
    output overall metrics: acc / sens / spec / f1 / kappa / auc / final / tp / tn / fp / fn,
    plus per-class AUC by ODIR categories.
    """
    model.eval()

    total_loss = 0.0
    total_n = 0

    probs_list = []
    labels_list = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_n += bs

        probs = torch.sigmoid(logits)

        probs_list.append(probs.detach().cpu())
        labels_list.append(y.detach().cpu())

    if total_n == 0:
        # Fallback for no data
        return {
            "loss": 0.0,
            "acc": 0.0,
            "sens": 0.0,
            "spec": 0.0,
            "f1": 0.0,
            "kappa": 0.0,
            "auc": 0.0,
            "final": 0.0,
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "per_class_auc": {},
        }

    probs_all = torch.cat(probs_list, dim=0)   # [N, 8]
    labels_all = torch.cat(labels_list, dim=0) # [N, 8]

    # Keep only classes that actually appear in this dataset (pos_samples > 0)
    cls_mask = (labels_all.sum(dim=0) > 0)           # [8]
    used_indices = [i for i, u in enumerate(cls_mask) if u]

    probs_used = probs_all[:, cls_mask].float()      # [N, C_used]
    labels_used = labels_all[:, cls_mask].float()    # [N, C_used]

    # ========= 1) Compute overall confusion / metrics (multi-label/multi-class micro) =========
    eps = 1e-8
    thr = 0.5

    pred = (probs_used >= thr).float()
    y = labels_used

    # per-class confusion
    TP_c = (pred * y).sum(dim=0)
    TN_c = ((1 - pred) * (1 - y)).sum(dim=0)
    FP_c = (pred * (1 - y)).sum(dim=0)
    FN_c = ((1 - pred) * y).sum(dim=0)

    # micro aggregation
    TP = TP_c.sum().item()
    TN = TN_c.sum().item()
    FP = FP_c.sum().item()
    FN = FN_c.sum().item()

    total = TP + TN + FP + FN + eps
    acc = (TP + TN) / total
    sens = TP / (TP + FN + eps)   # per-class recall / sensitivity
    spec = TN / (TN + FP + eps)

    f1 = 2 * TP / (2 * TP + FP + FN + eps)

    # Cohen's kappa (same as eval_binary_subset)
    p_o = acc
    p_yes_true = (TP + FN) / total
    p_yes_pred = (TP + FP) / total
    p_no_true = (TN + FP) / total
    p_no_pred = (TN + FN) / total
    p_e = p_yes_true * p_yes_pred + p_no_true * p_no_pred
    kappa = (p_o - p_e) / (1 - p_e + eps)

    # ========= 2) AUC（macro over used classes）=========
    probs_np = probs_used.numpy()
    labels_np = labels_used.numpy()

    try:
        auc_macro = roc_auc_score(labels_np, probs_np, average='macro')
    except Exception:
        auc_macro = 0.0

    # ========= 3) per-class AUC, named by ODIR_CLASSES =========
    per_class_auc = {}
    for j, cls_idx in enumerate(used_indices):
        y_true = labels_np[:, j]
        y_pred = probs_np[:, j]
        try:
            auc_j = roc_auc_score(y_true, y_pred)
        except Exception:
            auc_j = 0.0

        per_class_auc[ODIR_CLASSES[cls_idx]] = {
            "auc": float(auc_j),
            "pos_samples": int(y_true.sum()),
        }

    final = (kappa + f1 + auc_macro) / 3.0

    return {
        "loss": total_loss / max(total_n, 1),
        "acc": float(acc),
        "sens": float(sens),
        "spec": float(spec),
        "f1": float(f1),
        "kappa": float(kappa),
        "auc": float(auc_macro),
        "final": float(final),
        "tp": int(TP),
        "tn": int(TN),
        "fp": int(FP),
        "fn": int(FN),
        "per_class_auc": per_class_auc,
    }



def main():
    args = args_parser()

    # Fix to the configuration used during OIA-ODIR training
    args.dataset ='odir_multi'
    args.model = 'resnet18_multi'
    args.loss = 'bce'
    args.shape_in = (3, 256, 256)
    args.shape_out = 8   # adjust based on dataset

    # Best pretrained model path (consistent with pretrain_fedavg_resnet18.sh)
    args.load_model_path = (
        # "REPLACE_ME/ATP_multi/ATP/weights/odir_multi_bajwa/label/pretrain_fedavg_bce_resnet18_multi_bajwa__pseed_0_seed_0.pkl"
    #     "weights/odir_multi/label/"
        #  "pretrain_fedavg_bce_resnet18_multi__pseed_0_seed_0.pkl"
    "./weights/pretrain_fedavg_bce_resnet18_multi.pkl"
    )

    use_cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if use_cuda else "cpu")

    print(f"=> Loading model from: {args.load_model_path}")
    model = create_model(args)
    criterion = create_loss(args.loss)

    # # -------- DDR: N(0) vs D(1) --------
    # ddr_root = "REPLACE_ME/DDR-dataset/DR_grading"
    # ddr_test = DDRAsODIR(
    #     img_root=os.path.join(ddr_root, "test"),
    #     split_txt=os.path.join(ddr_root, "test.txt"),
    #     resize=256,
    # )
    # ddr_loader = DataLoader(ddr_test, batch_size=64, shuffle=False, num_workers=4)

    # print(f"[DDRAsODIR] root={ddr_root}/test, txt={ddr_root}/test.txt, samples={len(ddr_test)}")
    # ddr_metrics = eval_binary_subset(
    #     model=model,
    #     loader=ddr_loader,
    #     criterion=criterion,
    #     device=args.device,
    #     pos_idx=1,   # D
    #     neg_idx=0,   # N
    # )
    # print("[DDR N/D] "
    #       f"loss={ddr_metrics['loss']:.4f}, "
    #       f"acc={ddr_metrics['acc']:.4f}, "
    #       f"sens={ddr_metrics['sens']:.4f}, "
    #       f"spec={ddr_metrics['spec']:.4f}, "
    #       f"f1={ddr_metrics['f1']:.4f}, "
    #       f"kappa={ddr_metrics['kappa']:.4f}, "
    #       f"auc={ddr_metrics['auc']:.4f}, "
    #       f"final={ddr_metrics['final']:.4f}")
    # print(f"          TP={ddr_metrics['tp']}, TN={ddr_metrics['tn']}, "
    #       f"FP={ddr_metrics['fp']}, FN={ddr_metrics['fn']}")

    # -------- REFUGE: N(0) vs G(2) --------
    refuge_root = "REPLACE_ME/REFUGE-Multirater"
    refuge_test = REFUGEAsODIR(
        refuge_root=refuge_root,
        csv_path=os.path.join(refuge_root, "REFUGE1Test.csv"),
        resize=256,
    )
    refuge_loader = DataLoader(refuge_test, batch_size=64, shuffle=False, num_workers=4)

    refuge_metrics = eval_binary_subset(
        model=model,
        loader=refuge_loader,
        criterion=criterion,
        device=args.device,
        pos_idx=2,   # G
        neg_idx=0,   # N
    )
    print("[REFUGE N/G] "
          f"loss={refuge_metrics['loss']:.4f}, "
          f"acc={refuge_metrics['acc']:.4f}, "
          f"sens={refuge_metrics['sens']:.4f}, "
          f"spec={refuge_metrics['spec']:.4f}, "
          f"f1={refuge_metrics['f1']:.4f}, "
          f"kappa={refuge_metrics['kappa']:.4f}, "
          f"auc={refuge_metrics['auc']:.4f}, "
          f"final={refuge_metrics['final']:.4f}")
    print(f"            TP={refuge_metrics['tp']}, TN={refuge_metrics['tn']}, "
          f"FP={refuge_metrics['fp']}, FN={refuge_metrics['fn']}")

    # # -------- PAPILA: N(0) vs A(1) --------
    # papila_root = "REPLACE_ME/PapilaDB/FundusImages"
    # papila_test = PAPILAAsODIR(
    #     papila_root=papila_root,
    #     resize=256,
    # )
    # papila_loader = DataLoader(papila_test, batch_size=64, shuffle=False, num_workers=4)

    # papila_metrics = eval_binary_subset(
    #     model=model,
    #     loader=papila_loader,
    #     criterion=criterion,
    #     device=args.device,
    #     pos_idx=1,   # A (Abnormal)
    #     neg_idx=0,   # N (Normal)
    # )
    # print("[PAPILA N/A] "
    #       f"loss={papila_metrics['loss']:.4f}, "
    #       f"acc={papila_metrics['acc']:.4f}, "
    #       f"sens={papila_metrics['sens']:.4f}, "
    #       f"spec={papila_metrics['spec']:.4f}, "
    #       f"f1={papila_metrics['f1']:.4f}, "
    #       f"kappa={papila_metrics['kappa']:.4f}, "
    #       f"auc={papila_metrics['auc']:.4f}, "
    #       f"final={papila_metrics['final']:.4f}")
    # print(f"            TP={papila_metrics['tp']}, TN={papila_metrics['tn']}, "
    #       f"FP={papila_metrics['fp']}, FN={papila_metrics['fn']}")

    # # -------- HRF: N(0), D(1), G(2) --------
    # hrf_root = "REPLACE_ME/HRF"
    # hrf_test = HRFAsODIR(hrf_root=hrf_root, resize=256)
    # hrf_loader = DataLoader(hrf_test, batch_size=64, shuffle=False, num_workers=4)

    # print("\n[HRF N/D/G]")
    # hrf_metrics = eval_multiclass(model, hrf_loader, criterion, args.device)
    # print(
    #     "[HRF N/D/G] "
    #     f"loss={hrf_metrics['loss']:.4f}, "
    #     f"acc={hrf_metrics['acc']:.4f}, "
    #     f"sens={hrf_metrics['sens']:.4f}, "
    #     f"spec={hrf_metrics['spec']:.4f}, "
    #     f"f1={hrf_metrics['f1']:.4f}, "
    #     f"kappa={hrf_metrics['kappa']:.4f}, "
    #     f"auc={hrf_metrics['auc']:.4f}, "
    #     f"final={hrf_metrics['final']:.4f}"
    # )
    # print(
    #     f"          TP={hrf_metrics['tp']}, TN={hrf_metrics['tn']}, "
    #     f"FP={hrf_metrics['fp']}, FN={hrf_metrics['fn']}"
    # )
    # print("          per-class AUC:", hrf_metrics["per_class_auc"])


    # # -------- Bajwa: N(0), C(3), G(2), O(7) --------
    # bajwa_root = "REPLACE_ME/Bajwa"
    # bajwa_test = BajwaAsODIR(bajwa_root=bajwa_root, resize=256)
    # bajwa_loader = DataLoader(bajwa_test, batch_size=64, shuffle=False, num_workers=4)

    # print("\n[Bajwa N/C/G/O]")
    # bajwa_metrics = eval_multiclass(model, bajwa_loader, criterion, args.device)
    # print(
    #     "[Bajwa N/C/G/O] "
    #     f"loss={bajwa_metrics['loss']:.4f}, "
    #     f"acc={bajwa_metrics['acc']:.4f}, "
    #     f"sens={bajwa_metrics['sens']:.4f}, "
    #     f"spec={bajwa_metrics['spec']:.4f}, "
    #     f"f1={bajwa_metrics['f1']:.4f}, "
    #     f"kappa={bajwa_metrics  ['kappa']:.4f}, "
    #     f"auc={bajwa_metrics['auc']:.4f}, "
    #     f"final={bajwa_metrics['final']:.4f}"
    # )
    # print(
    #     f"          TP={bajwa_metrics['tp']}, TN={bajwa_metrics['tn']}, "
    #     f"FP={bajwa_metrics['fp']}, FN={bajwa_metrics['fn']}"
    # )
    # print("          per-class AUC:", bajwa_metrics["per_class_auc"])




if __name__ == "__main__":
    main()

