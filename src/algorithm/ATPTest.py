import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from copy import deepcopy

from model import create_model, create_loss, create_metric, create_optimizer
from model.MyBatchNorm2d import MyBatchNorm2d
from utils import pickle_load

from .Base import BaseServer, BaseClient
# from .TTABase import TTABaseServer  # optional dependency

import pickle
from typing import Optional
import numpy as np
import os
from algorithm.LabelShiftEM import (
    BinaryVectorScaling, BCTS, VS,
    em_labelshift_multiclass, em_labelshift_binary,
    prob_to_logit
)

# ============= Utilities ============= #

def robust_load_state_dict(path):
    """
    Supports both torch.save() and pickle.dump(state_dict) generated files.
    Returns: state_dict (dict[str, Tensor])
    """
    sd = None
    try:
        obj = torch.load(path, map_location="cpu")
    except Exception:
        obj = None

    if obj is None:
        with open(path, "rb") as f:
            obj = pickle.load(f)

    def find_state_dict(o):
        if isinstance(o, dict) and all(isinstance(v, torch.Tensor) for v in o.values()):
            return o
        if isinstance(o, dict):
            for k in ["state_dict", "model", "model_state", "net", "weights"]:
                if k in o:
                    return find_state_dict(o[k])
        if isinstance(o, list):
            for elem in reversed(o):
                sd2 = find_state_dict(elem)
                if sd2 is not None:
                    return sd2
        return None

    sd = find_state_dict(obj)
    if sd is None:
        raise RuntimeError(f"cannot find state_dict in file: {path}")
    return sd


def _unwrap_logits(out):
    """out may be a Tensor or a tuple/list like (logits, class_feats). Always extract logits."""
    return out[0] if isinstance(out, (tuple, list)) else out


def wavg_state(state1, state2, lamda):
    state = deepcopy(state1)
    for k in state1.keys():
        state[k] = lamda * state1[k] + (1 - lamda) * state2[k]
    return state


def compute_calib_metrics(probs, labels, n_bins=15):
    """
    Compute calibration metrics for multi-label classification.
    probs: [N, C] predicted probabilities (clamped to [eps, 1-eps])
    labels: [N, C] binary ground truth
    Returns: dict with ECE, MCE, Brier, NLL
    """
    N, C = probs.shape
    device = probs.device

    # ECE / MCE
    ece_total = 0.0
    mce = 0.0
    boundaries = torch.linspace(0.0, 1.0, n_bins + 1, device=device)
    for c in range(C):
        p_c = probs[:, c]
        y_c = labels[:, c]
        ece_c = 0.0
        for i in range(n_bins):
            lo, hi = boundaries[i], boundaries[i + 1]
            in_bin = (p_c > lo) & (p_c <= hi)
            if i == 0:
                in_bin = in_bin | (p_c == 0.0)
            n_b = in_bin.sum().item()
            if n_b == 0:
                continue
            conf = p_c[in_bin].mean().item()
            acc = y_c[in_bin].float().mean().item()
            gap = abs(acc - conf)
            ece_c += (n_b / N) * gap
            if gap > mce:
                mce = gap
        ece_total += ece_c

    ece = ece_total / C
    brier = float(((probs - labels) ** 2).mean().item())
    nll = float(F.binary_cross_entropy(probs, labels, reduction="mean").item())

    return {"ECE": ece, "MCE": mce, "Brier": brier, "NLL": nll}


# ============= Server ============= #

class ATPTestServer(BaseServer):
    def __init__(self, train_datasets, test_datasets, args):
        BaseServer.__init__(self, train_datasets, test_datasets, args)
        self.args = args
        self.debug = bool(getattr(args, "debug", 1))

        self.train_clients = {cid: ATPTestClient(cid, datasets, args) for cid, datasets in train_datasets.items()}
        self.test_clients  = {cid: ATPTestClient(cid, datasets, args) for cid, datasets in test_datasets.items()}

        # load model
        self.model = create_model(args)

        # checkpoint
        if getattr(self.args, "load_model_path", None):
            try:
                sd = robust_load_state_dict(self.args.load_model_path)
                missing, unexpected = self.model.load_state_dict(sd, strict=False)
                print(f"[CHKPT] loaded: {self.args.load_model_path}")
                print(f"[CHKPT] missing={len(missing)}, unexpected={len(unexpected)}")
                with torch.no_grad():
                    s, n = 0.0, 0
                    for p in self.model.parameters():
                        s += p.abs().sum().item()
                        n += p.numel()
                    print(f"[CHKPT] model digest: sum={s:.6f}, mean={s/n:.6f}")
            except Exception as e:
                print(f"[CHKPT][FATAL] cannot load {self.args.load_model_path}: {e}")

        self.model.change_bn(mode='grad')
        self.model.eval()

        # load adapt lrs
        self.adaptation_rates = self.load_adapt_lrs(args)

        # restore calibrator, prior, thresholds
        calib, p_y_src, taus = self._restore_calibrator_prior_thresholds(self.args)
        self._calibrator = calib
        self._p_y_src    = p_y_src
        if getattr(self.args, 'eval_thresholds', 'auto') == '0.5':
            self.args.eval_thresholds = None  # force 0.5 for all classes
        elif taus is not None:
            self.args.eval_thresholds = taus.detach().cpu().tolist()
            if self.debug:
                print(f"[TAU] min={taus.min().item():.4f}  mean={taus.mean().item():.4f}  max={taus.max().item():.4f}")

    # --------- run() --------- #
    def run(self, args):
        """
        Called by main.py. Routes based on labelshift and test mode.
        """
        if getattr(args, 'labelshift', 'none') == 'em':
            if args.test in ['batch', 'online', 'online_avg', 'online_exp',
                             'online_small', 'online_raw', 'large_batch', 'online_ha']:
                self.eval_with_labelshift_em_clientwise(args)   # TTA + EM (per client)
            else:
                self.eval_with_labelshift_em(args)              # EM only (no TTA)
        else:
            self.adapt_and_eval(args, 'test')                   # TTA only

    # --------- load adapt lrs (robust) --------- #
    def load_adapt_lrs(self, args):
        path = args.load_adapt_path
        idx  = int(getattr(args, "load_adapt_idx", 0))
        rnd  = int(getattr(args, "load_adapt_round", -1))
        dev  = args.device

        def _zeros_like_model():
            try:
                L = len(self.model.trainable_parameters())
            except Exception:
                L = len([p for p in self.model.parameters() if p.requires_grad])
            return torch.zeros(L, device=dev, dtype=torch.float32)

        if path == 'manual':
            print("[ADAPT] Using manual pattern (fallback to zeros for test).")
            return _zeros_like_model()

        if path == 'zero':
            rate = _zeros_like_model()
            print(f"[DEBUG] adapt_lrs (zero) len={rate.numel()}")
            return rate

        try:
            payloads = pickle_load(path, multiple=True)
            seq = payloads if isinstance(payloads, list) else [payloads]
            if len(seq) == 0:
                print(f"[ADAPT][WARN] empty history in {path}; fallback zeros.")
                return _zeros_like_model()

            sel = seq[idx if idx != -1 else -1]
            hist = sel.get("history", sel) if isinstance(sel, dict) else sel
            if "adapt_lrs" not in hist or len(hist["adapt_lrs"]) == 0:
                print(f"[ADAPT][WARN] 'adapt_lrs' not found in history; fallback zeros.")
                return _zeros_like_model()

            vecs = hist["adapt_lrs"]
            vec  = vecs[rnd if rnd != -1 else -1]
            rate = torch.tensor(vec, dtype=torch.float32, device=dev)

            need = len(self.model.trainable_parameters())
            if rate.numel() != need:
                print(f"[ADAPT][WARN] adapt_lrs length mismatch: file={rate.numel()} vs model={need}. "
                      f"{'truncate' if rate.numel()>need else 'pad zeros'}")
                if rate.numel() > need:
                    rate = rate[:need]
                else:
                    z = torch.zeros(need, device=dev, dtype=rate.dtype)
                    z[:rate.numel()] = rate
                    rate = z

            print(f"[DEBUG] adapt_lrs mean={rate.mean().item():.6f}, max={rate.max().item():.6f}")
            return rate

        except Exception as e:
            print(f"[ADAPT][FATAL] cannot load adapt_lrs from {path}: {e}")
            return _zeros_like_model()

    # --------- restore calibrator + prior + thresholds --------- #
    def _restore_calibrator_prior_thresholds(self, args):
        path = args.load_adapt_path
        try:
            payloads = pickle_load(path, multiple=True)
        except Exception as e:
            print(f"[CAL][FATAL] cannot load {path}: {e}")
            return None, None, None

        calib_rec, prior_rec, taus_rec = None, None, None

        seq = payloads if isinstance(payloads, list) else [payloads]
        for obj in reversed(seq):
            hist = obj.get("history", obj) if isinstance(obj, dict) else obj
            if not isinstance(hist, dict):
                continue
            if calib_rec is None and "calibration" in hist:
                calib_rec = hist["calibration"]
            if prior_rec is None and "source_prior" in hist:
                prior_rec = hist["source_prior"]
            if taus_rec is None and "src_thresholds" in hist:
                taus_rec = hist["src_thresholds"]
            if calib_rec is not None and prior_rec is not None and taus_rec is not None:
                break

        assert calib_rec is not None and prior_rec is not None, "No calibration/source_prior in history!"

        device = next(self.model.parameters()).device
        C = len(prior_rec)
        prior = torch.tensor(prior_rec, dtype=torch.float32, device=device)

        typ = calib_rec.get('type', 'none')
        params = calib_rec.get('params', {})

        if typ == 'bvs':
            calib = BinaryVectorScaling(C).to(device)
            calib.a.data.copy_(torch.tensor(params['a'], device=device))
            calib.b.data.copy_(torch.tensor(params['b'], device=device))
        elif typ == 'bcts':
            calib = BCTS(C).to(device)
            calib.T.data.copy_(torch.tensor(params['T'], device=device))
            calib.bias.data.copy_(torch.tensor(params['bias'], device=device))
        elif typ == 'vs':
            calib = VS(C).to(device)
            calib.W.data.copy_(torch.tensor(params['W'], device=device))
            calib.b.data.copy_(torch.tensor(params['b'], device=device))
        else:
            calib = None

        if calib is not None:
            for p in calib.parameters():
                p.requires_grad_(False)

        if self.debug:
            print(f"[CAL] from: {path} | type={typ} | prior_sum={float(prior.sum())}")
            if typ == 'bvs':
                a = torch.tensor(params['a']).float()
                b = torch.tensor(params['b']).float()
                print(f"[CAL] BVS a.mean={a.mean():.6f}, b.mean={b.mean():.6f}")

        taus = None
        if taus_rec is not None:
            taus = torch.tensor(taus_rec, dtype=torch.float32, device=device)

        return calib, prior, taus

    # --------- helpers --------- #
    @torch.no_grad()
    def _calibrate_probs(self, logits, calib, is_multilabel=True, eps=1e-6):
        if calib is None:
            p = torch.sigmoid(logits) if is_multilabel else torch.softmax(logits, dim=-1)
        else:
            p = calib.predict_proba(logits)
        return p.clamp(min=eps, max=1 - eps)

    def _safe_logit(self, p, eps=1e-6):
        p = p.clamp(min=eps, max=1 - eps)
        z = torch.log(p) - torch.log1p(-p)
        if torch.isnan(z).any() or torch.isinf(z).any():
            z = torch.nan_to_num(z, nan=0.0, posinf=10.0, neginf=-10.0)
        return z

    def _row_normalize_probs(self, p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        p = p.clamp_min(eps)
        s = p.sum(dim=1, keepdim=True)
        mask = (s > eps).float()
        p_norm = p / torch.where(s > eps, s, torch.ones_like(s))
        if (mask == 0).any():
            K = p.shape[1]
            uniform = torch.full_like(p, 1.0 / K)
            p_norm = mask * p_norm + (1 - mask) * uniform
        return p_norm

    def _prep_em_input(self, p: torch.Tensor, temp: float, conf_pow: float,
                       conf_thresh: float, eps: float) -> torch.Tensor:
        p = p.clamp_min(eps)
        if temp != 1.0:
            p = F.softmax(torch.log(p) / temp, dim=-1)
        if conf_pow != 1.0:
            p = (p ** conf_pow)
            p = self._row_normalize_probs(p, eps=eps)
        if conf_thresh > 0.0:
            m = p.max(dim=1).values
            keep = (m >= conf_thresh).float().unsqueeze(1)
            uniform = torch.full_like(p, 1.0 / p.size(1))
            p = keep * p + (1 - keep) * uniform
        return p.clamp_min(eps)

    def _shrink_prior_and_recompose(self, p_in: torch.Tensor, qy_hat: torch.Tensor, p_y_src: torch.Tensor,
                                    mix: float, ratio_cap: Optional[float], eps: float):
        qy = qy_hat / qy_hat.sum()
        if mix > 0:
            qy = (1 - mix) * qy + mix * (p_y_src / p_y_src.sum())
            qy = qy / qy.sum()

        adj = (qy / (p_y_src + eps))
        if ratio_cap is not None and ratio_cap > 0:
            adj = adj.clamp(1.0 / ratio_cap, ratio_cap)

        qyx = self._row_normalize_probs(p_in * adj.view(1, -1), eps=eps)
        return qy, qyx

#---------------------------------------------------------------

    # --------- Compute multi-metrics from probabilities: bACC / Kappa / F1 / AUC / Final --------- #
    def _all_metrics_from_probs(self, probs: torch.Tensor, Y: torch.Tensor,
                                taus=None, eps: float = 1e-8):
        """
        probs: [N, C], multi-label probabilities (sigmoid or calibrated)
        Y    : [N] or [N, C]
        taus : None or [C] threshold vector (per-class tau from source calibration)
        Returns: dict { 'bacc', 'kappa', 'f1', 'auc', 'final' }
        """
        device = probs.device
        N, C = probs.shape

        # Convert to multi-hot [N,C]
        if Y.dim() == 1:
            Y = F.one_hot(Y.long(), num_classes=C).float().to(device)
        elif Y.dim() == 2 and Y.size(1) == C:
            Y = Y.float().to(device)
        else:
            raise ValueError(f"Unexpected Y shape: {tuple(Y.shape)}; expect [N] or [N,{C}]")

        # Threshold: per-class tau or 0.5
        if taus is None:
            thr = 0.5
        else:
            if isinstance(taus, torch.Tensor):
                taus_t = taus.to(device=device, dtype=probs.dtype)
            else:
                taus_t = torch.tensor(taus, device=device, dtype=probs.dtype)
            thr = taus_t.view(1, C)   # [1,C]

        pred = (probs >= thr).float()

        # --- per-class confusion matrix for bACC ---
        TP_c = (pred * Y).sum(dim=0)
        TN_c = ((1 - pred) * (1 - Y)).sum(dim=0)
        FP_c = (pred * (1 - Y)).sum(dim=0)
        FN_c = ((1 - pred) * Y).sum(dim=0)

        TPR_c = TP_c / (TP_c + FN_c + eps)
        TNR_c = TN_c / (TN_c + FP_c + eps)
        bacc = 0.5 * (TPR_c + TNR_c)
        bacc_mean = bacc.mean().item()

        # --- micro confusion matrix for Kappa / F1 ---
        TP = TP_c.sum()
        TN = TN_c.sum()
        FP = FP_c.sum()
        FN = FN_c.sum()

        total = TP + TN + FP + FN + eps
        po = (TP + TN) / total
        pe = ((TP + FP) * (TP + FN) + (FN + TN) * (FP + TN)) / (total * total)
        kappa = ((po - pe) / (1 - pe + eps)).item()

        f1 = (2 * TP / (2 * TP + FP + FN + eps)).item()

        # --- AUC（macro over classes）---
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(
                Y.detach().cpu().numpy(),
                probs.detach().cpu().numpy(),
                average='macro'
            )
            auc = float(auc)
        except Exception:
            auc = float('nan')

        final = (kappa + f1 + auc) / 3.0

        return {
            'bacc':  bacc_mean,
            'kappa': kappa,
            'f1':    f1,
            'auc':   auc,
            'final': final,
        }


#---------------------------------------------------------------

    # --------- EM + clientwise TTA --------- #
    def eval_with_labelshift_em_clientwise(self, args):
        device = next(self.model.parameters()).device
        self.model.eval()

        calib = self._calibrator
        p_y_src = self._p_y_src

        global_state = deepcopy(self.model.updated_state_dict())
        eps = float(getattr(args, "em_min_prob", 1e-6))

        taus = getattr(self.args, 'eval_thresholds', None)

        weights = []

        # Aggregate weighted sum of all metrics per scenario
        sums_after      = {k: 0.0 for k in ['bacc', 'kappa', 'f1', 'auc', 'final']}
        sums_calib      = {k: 0.0 for k in ['bacc', 'kappa', 'f1', 'auc', 'final']}
        sums_em_only    = {k: 0.0 for k in ['bacc', 'kappa', 'f1', 'auc', 'final']}
        sums_em_calib   = {k: 0.0 for k in ['bacc', 'kappa', 'f1', 'auc', 'final']}

        # Calibration metrics accumulators (after-TTA, +Calib, +Calib+EM)
        sums_calib_after = {k: 0.0 for k in ['ECE', 'MCE', 'Brier', 'NLL']}
        sums_calib_calib = {k: 0.0 for k in ['ECE', 'MCE', 'Brier', 'NLL']}

        losses_em_only, losses_em_calib = [], []

        # Reliability diagram data collectors
        reli_probs_after, reli_probs_calib, reli_labels = [], [], []

        # Timing accumulators (seconds)
        time_ttp_total = 0.0
        time_calib_total = 0.0
        time_em_total = 0.0
        time_forward_total = 0.0
        forward_total_samples = 0
        total_latency_samples = 0

        # Parameter counts: total vs BN-only (what TTP actually adapts)
        n_total_params = sum(p.numel() for p in self.model.parameters())
        n_bn_params = sum(
            p.numel() for m in self.model.modules()
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, MyBatchNorm2d))
            for p in m.parameters()
        )
        n_adapt_params = len(self.adaptation_rates)
        n_nonzero_lr = int((self.adaptation_rates.abs() > 1e-8).sum().item())

        for cid, client in self.test_clients.items():
            t0 = time.perf_counter()
            with torch.enable_grad():
                avg_loss, avg_metric, n, logits_cpu, Y_cpu = client.local_eval(
                    self.model, self.adaptation_rates, args, dataset='test', return_logits=True
                )
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            time_ttp_total += (t1 - t0)
            total_latency_samples += n

            # Pure forward-pass timing (no grad) for comparison
            if time_forward_total == 0.0:  # measure once on first client
                dl = client.dataloaders['test']
                model_state = deepcopy(self.model.state_dict())
                self.model.eval()
                t_f0 = time.perf_counter()
                with torch.no_grad():
                    for *X_f, _ in dl:
                        X_f = [x.to(device) for x in X_f]
                        _ = self.model(*X_f)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                time_forward_total = time.perf_counter() - t_f0
                forward_total_samples = sum(
                    x[0].size(0) for *x, _ in dl
                )
                self.model.load_state_dict(model_state)

            if n == 0:
                continue

            weights.append(n)
            with torch.no_grad():
                logits = logits_cpu.to(device)
                Y = Y_cpu.to(device)
                if Y.dim() == 1:
                    Y = F.one_hot(Y.long(), num_classes=logits.shape[1]).float()
                else:
                    Y = Y.float()

                # -------- after-TTA (raw logits -> sigmoid probabilities) --------
                p_after = torch.sigmoid(logits).clamp(eps, 1 - eps)
                m_after = self._all_metrics_from_probs(p_after, Y, taus=taus, eps=eps)
                for k in sums_after.keys():
                    sums_after[k] += m_after[k] * n

                # -------- +Calib --------
                t_cal0 = time.perf_counter()
                p_calib = self._calibrate_probs(logits, calib, is_multilabel=True, eps=eps)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                time_calib_total += (time.perf_counter() - t_cal0)
                m_calib = self._all_metrics_from_probs(p_calib, Y, taus=taus, eps=eps)
                for k in sums_calib.keys():
                    sums_calib[k] += m_calib[k] * n

                # Calibration metrics: after-TTA vs +Calib
                cal_after = compute_calib_metrics(p_after, Y)
                cal_calib = compute_calib_metrics(p_calib, Y)
                for k in sums_calib_after.keys():
                    sums_calib_after[k] += cal_after[k] * n
                    sums_calib_calib[k] += cal_calib[k] * n

                # Save (prob, label) for reliability diagram
                reli_probs_after.append(p_after.cpu().numpy())
                reli_probs_calib.append(p_calib.cpu().numpy())
                reli_labels.append(Y.cpu().numpy())

                # -------- +EM-only (per-column binary EM) --------
                t_em0 = time.perf_counter()
                p_raw = torch.sigmoid(logits).clamp(eps, 1 - eps)
                qyx_cols = []
                for c in range(p_y_src.numel()):
                    q1, q1_x = em_labelshift_binary(
                        p_raw[:, c], p_y_src[c],
                        max_iter=args.em_max_iter, tol=args.em_tol, min_prob=eps
                    )
                    qyx_cols.append(q1_x.unsqueeze(1))
                qyx_only = torch.cat(qyx_cols, dim=1).clamp(eps, 1 - eps)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                time_em_total += (time.perf_counter() - t_em0)

                loss_em_only = F.binary_cross_entropy(qyx_only, Y)
                losses_em_only.append(loss_em_only.item() * n)

                m_em_only = self._all_metrics_from_probs(qyx_only, Y, taus=taus, eps=eps)
                for k in sums_em_only.keys():
                    sums_em_only[k] += m_em_only[k] * n

                # -------- +Calib+EM --------
                qyx_cols2 = []
                for c in range(p_y_src.numel()):
                    q1, q1_x = em_labelshift_binary(
                        p_calib[:, c], p_y_src[c],
                        max_iter=args.em_max_iter, tol=args.em_tol, min_prob=eps
                    )
                    qyx_cols2.append(q1_x.unsqueeze(1))
                qyx = torch.cat(qyx_cols2, dim=1).clamp(eps, 1 - eps)

                loss_em = F.binary_cross_entropy(qyx, Y)
                losses_em_calib.append(loss_em.item() * n)

                m_em = self._all_metrics_from_probs(qyx, Y, taus=taus, eps=eps)
                for k in sums_em_calib.keys():
                    sums_em_calib[k] += m_em[k] * n

                # Positive ratio diagnostic (keep original print logic)
                if self.debug and cid == list(self.test_clients.keys())[0]:
                    taus_t = torch.tensor(taus, device=device, dtype=torch.float32) if taus is not None else None
                    if taus_t is not None:
                        pos_after = (p_after >= taus_t.view(1, -1)).float().mean(dim=0)
                        pos_calib = (p_calib >= taus_t.view(1, -1)).float().mean(dim=0)
                        pos_em    = (qyx_only >= taus_t.view(1, -1)).float().mean(dim=0)
                        pos_emc   = (qyx >= taus_t.view(1, -1)).float().mean(dim=0)
                        def _summ(name, v):
                            return f"{name}: min={v.min():.3f} mean={v.mean():.3f} max={v.max():.3f}"
                        print("[DIAG][posrate] " +
                              _summ("after", pos_after) + " | " +
                              _summ("calib", pos_calib) + " | " +
                              _summ("em", pos_em) + " | " +
                              _summ("em+calib", pos_emc))

            # reset for next client
            self.model.load_state_dict(global_state, strict=False)

            if self.debug and cid == list(self.test_clients.keys())[0]:
                print(f"[DIAG] after-TTA={m_after['bacc']:.4f} "
                      f"| +Calib={m_calib['bacc']:.4f} "
                      f"| +EM-only={m_em_only['bacc']:.4f} "
                      f"| +Calib+EM={m_em['bacc']:.4f}")
                

        denom = sum(weights) if len(weights) else 1.0

        def _avg(sums_dict):
            return {k: v / denom for k, v in sums_dict.items()}

        avg_after    = _avg(sums_after)
        avg_calib    = _avg(sums_calib)
        avg_em_only  = _avg(sums_em_only)
        avg_em_calib = _avg(sums_em_calib)

        print(f"\n\t[SUMMARY][TTA + Calib/EM]  (mode={args.test})")
        print(f"\t  after-TTA : "
              f"bACC={avg_after['bacc']:.4f}  Kappa={avg_after['kappa']:.4f}  "
              f"F1={avg_after['f1']:.4f}  AUC={avg_after['auc']:.4f}  Final={avg_after['final']:.4f}")
        print(f"\t  after-TTA_calib: "
              f"ECE={sums_calib_after['ECE']/denom:.4f}  MCE={sums_calib_after['MCE']/denom:.4f}  "
              f"Brier={sums_calib_after['Brier']/denom:.4f}  NLL={sums_calib_after['NLL']/denom:.4f}")
        print(f"\t  +Calib    : "
              f"bACC={avg_calib['bacc']:.4f}  Kappa={avg_calib['kappa']:.4f}  "
              f"F1={avg_calib['f1']:.4f}  AUC={avg_calib['auc']:.4f}  Final={avg_calib['final']:.4f}")
        print(f"\t  +Calib_calib: "
              f"ECE={sums_calib_calib['ECE']/denom:.4f}  MCE={sums_calib_calib['MCE']/denom:.4f}  "
              f"Brier={sums_calib_calib['Brier']/denom:.4f}  NLL={sums_calib_calib['NLL']/denom:.4f}")
        print(f"\t  +EM-only  : Loss={sum(losses_em_only)/denom:.4f}  "
              f"bACC={avg_em_only['bacc']:.4f}  Kappa={avg_em_only['kappa']:.4f}  "
              f"F1={avg_em_only['f1']:.4f}  AUC={avg_em_only['auc']:.4f}  Final={avg_em_only['final']:.4f}")
        print(f"\t  +Calib+EM : Loss={sum(losses_em_calib)/denom:.4f}  "
              f"bACC={avg_em_calib['bacc']:.4f}  Kappa={avg_em_calib['kappa']:.4f}  "
              f"F1={avg_em_calib['f1']:.4f}  AUC={avg_em_calib['auc']:.4f}  Final={avg_em_calib['final']:.4f}")

        # Timing summary
        if total_latency_samples > 0:
            ms_ttp = time_ttp_total / total_latency_samples * 1000
            ms_calib = time_calib_total / total_latency_samples * 1000
            ms_em = time_em_total / total_latency_samples * 1000
            ms_fwd = time_forward_total / forward_total_samples * 1000 if forward_total_samples > 0 else 0
            n_clients = len(self.test_clients)
            print(f"\n\t[LATENCY]  mode={args.test}  "
                  f"clients={n_clients}  samples={total_latency_samples}  "
                  f"batch_size={args.batch_size}")
            print(f"\t  Total params: {n_total_params:,}  "
                  f"BN params: {n_bn_params:,}  "
                  f"({n_bn_params/n_total_params*100:.1f}% of total)  "
                  f"non-zero adapt_lrs: {n_nonzero_lr}/{n_adapt_params}")
            print(f"\t  Forward-only (no TTP):  {time_forward_total:.4f}s total  "
                  f"{ms_fwd:.4f} ms/sample")
            print(f"\t  TTP (fwd+bwd+update):   {time_ttp_total:.4f}s total  "
                  f"{ms_ttp:.4f} ms/sample  "
                  f"(+{(ms_ttp - ms_fwd)/ms_fwd*100:.1f}% vs forward-only)" if ms_fwd > 0 else "")
            print(f"\t  BVS calibration:        {time_calib_total:.4f}s total  "
                  f"{ms_calib:.4f} ms/sample")
            print(f"\t  EM label shift:          {time_em_total:.4f}s total  "
                  f"{ms_em:.4f} ms/sample")
            print(f"\t  Total TTP+BVS+EM:        {time_ttp_total+time_calib_total+time_em_total:.4f}s total  "
                  f"{(time_ttp_total+time_calib_total+time_em_total)/total_latency_samples*1000:.4f} ms/sample")

        self.history.append({
            f'after_tta_metrics_{args.test}':  avg_after,
            f'calib_metrics_{args.test}':      avg_calib,
            f'em_only_clientwise_tta_{args.test}_wavg_loss':   sum(losses_em_only) / denom,
            f'em_only_clientwise_tta_{args.test}_wavg_metrics': avg_em_only,
            f'em_clientwise_tta_{args.test}_wavg_loss':        sum(losses_em_calib) / denom,
            f'em_clientwise_tta_{args.test}_wavg_metrics':     avg_em_calib,
        })

        # Save reliability diagram source data
        if reli_probs_after:
            all_after = np.concatenate(reli_probs_after, axis=0)
            all_calib = np.concatenate(reli_probs_calib, axis=0)
            all_labels_np = np.concatenate(reli_labels, axis=0)
            reli_path = getattr(args, 'history_path', 'none')
            if reli_path != 'none':
                reli_npz = reli_path.replace('.pkl', '_reliability.npz')
                np.savez_compressed(reli_npz,
                    probs_after=all_after, probs_calib=all_calib, labels=all_labels_np)
                print(f"[RELI] Saved reliability data to {reli_npz}")


    # --------- EM only (no TTA) --------- #
    @torch.no_grad()
    def eval_with_labelshift_em(self, args):
        device = next(self.model.parameters()).device
        self.model.eval()

        calib = self._calibrator
        p_y_src = self._p_y_src
        eps = float(getattr(args, "em_min_prob", 1e-6))
        taus = getattr(args, 'eval_thresholds', None)

        weights = []
        losses_em = []
        all_logits_list, all_labels_list = [], []

        for cid, client in self.test_clients.items():
            dl = client.dataloaders['test']
            n_client = 0
            loss_em_sum = 0.0

            for *X, Y in dl:
                X = [x.to(device) for x in X]
                Y = Y.to(device)
                out = self.model(*X)
                logits = _unwrap_logits(out)
                if Y.dim() == 1:
                    Y = F.one_hot(Y.long(), num_classes=logits.shape[1]).float()
                else:
                    Y = Y.float()

                all_logits_list.append(logits.cpu())
                all_labels_list.append(Y.cpu())

                # EM-only (per class)
                p_raw = torch.sigmoid(logits).clamp(eps, 1 - eps)
                qyx_cols = []
                for c in range(p_y_src.numel()):
                    q1, q1_x = em_labelshift_binary(
                        p_raw[:, c], p_y_src[c],
                        max_iter=args.em_max_iter, tol=args.em_tol, min_prob=eps
                    )
                    qyx_cols.append(q1_x.unsqueeze(1))
                qyx_only = torch.cat(qyx_cols, dim=1).clamp(eps, 1 - eps)
                loss_em_sum += F.binary_cross_entropy(qyx_only, Y).item() * Y.size(0)
                n_client += Y.size(0)

            if n_client > 0:
                weights.append(n_client)
                losses_em.append(loss_em_sum)

        denom = sum(weights) if len(weights) else 1.0

        # Compute full per-metric breakdown
        logits_all = torch.cat(all_logits_list, dim=0).to(device)
        labels_all = torch.cat(all_labels_list, dim=0).to(device)

        # Raw sigmoid (no calibration, no TTP)
        p_raw = torch.sigmoid(logits_all).clamp(eps, 1 - eps)
        m_raw = self._all_metrics_from_probs(p_raw, labels_all, taus=taus, eps=eps)

        # +BVS calibration (no TTP)
        if calib is not None:
            p_calib = self._calibrate_probs(logits_all, calib, is_multilabel=True, eps=eps)
            m_calib = self._all_metrics_from_probs(p_calib, labels_all, taus=taus, eps=eps)
        else:
            m_calib = m_raw

        print(f"\n\t[SUMMARY][EM only (no TTP)]")
        print(f"\t  Raw       : bACC={m_raw['bacc']:.4f}  Kappa={m_raw['kappa']:.4f}  "
              f"F1={m_raw['f1']:.4f}  AUC={m_raw['auc']:.4f}  Final={m_raw['final']:.4f}")
        if calib is not None:
            print(f"\t  +Calib    : bACC={m_calib['bacc']:.4f}  Kappa={m_calib['kappa']:.4f}  "
                  f"F1={m_calib['f1']:.4f}  AUC={m_calib['auc']:.4f}  Final={m_calib['final']:.4f}")
        print(f"\t  +EM-only  : Loss={ sum(losses_em) /denom :.4f}")

        self.history.append({
            f'em_only_wavg_loss':    sum(losses_em) / denom,
            f'em_only_raw_metrics':  m_raw,
            f'em_only_calib_metrics': m_calib,
        })

    # --------- Plain TTA-only --------- #
    def adapt_and_eval(self, args, mode='test'):
        global_state = deepcopy(self.model.updated_state_dict())

        weights, losses, metrics = [], [], []
        clients = self.train_clients if mode == 'valid' else self.test_clients

        for cid, client in tqdm(clients.items()):
            loss, metric, num_data = client.local_eval(self.model, self.adaptation_rates, args, 'test')
            weights.append(num_data)
            losses.append(loss)
            metrics.append(metric)
            self.model.load_state_dict(global_state, strict=False)

        agg_loss = sum([w * l for w, l in zip(weights, losses)]) / max(sum(weights), 1)
        agg_metric = sum([w * m for w, m in zip(weights, metrics)]) / max(sum(weights), 1)
        tqdm.write('\t Eval[SUMMARY][TTA-only]:  Loss: %.4f \t Metric: %.4f' % (agg_loss, agg_metric))

        self.history.append({
            mode + '_losses': losses,
            mode + '_metrics': metrics,
            mode + '_wavg_loss': agg_loss,
            mode + '_wavg_metric': agg_metric,
        })


# ============= Client ============= #

class ATPTestClient(BaseClient):
    def adapt_one_step(self, model, adapt_lrs, X, Y, unspv_loss_func, args):
        model.eval()
        out = model(*X)
        logits = _unwrap_logits(out)

        # Unsupervised entropy minimization (multi-label)
        loss = unspv_loss_func(logits, Y)
        loss.backward()

        model.set_running_stat_grads()
        unspv_grad = [p.grad.clone() for p in model.trainable_parameters()]

        with torch.no_grad():
            for i, (p, g) in enumerate(zip(model.trainable_parameters(), unspv_grad)):
                p -= adapt_lrs[i] * g

        model.zero_grad()
        model.clip_bn_running_vars()
        return unspv_grad

    def local_eval(self, model, adapt_lrs, args, dataset='test', return_logits=False):
        unspv_loss_func = create_loss('ent')
        spv_loss_func   = create_loss('ral')

        # Critical: metric must consume thresholds
        metric_func = create_metric('bacc', thresholds=getattr(args, 'eval_thresholds', None))

        current_lrs = adapt_lrs.clone()
        total_examples, total_loss, total_metric = 0, 0, 0

        dataloader = self.dataloaders[dataset]
        num_data = self.num_data[dataset]

        if getattr(args, "verbose", 0) or getattr(args, "debug", 1):
            print(f"[DEBUG] Client {self.cid}, Dataset '{dataset}', num_data: {num_data}, len(dataloader): {len(dataloader)}")

        if len(dataloader) == 0 or num_data == 0:
            print(f"[WARNING] Client {self.cid} has no data for dataset '{dataset}'. Skipping local_eval.")
            if return_logits:
                return 0.0, 0.0, 0, torch.empty(0), torch.empty(0)
            else:
                return 0.0, 0.0, 0

        state = deepcopy(model.state_dict())
        cached_logits, cached_labels = [], []

        acc_state = None  # for online_avg

        for i, (*X, Y) in enumerate(dataloader):
            if args.test in ['batch', 'large_batch']:
                model.load_state_dict(state)
                if args.test == 'large_batch':
                    current_lrs = adapt_lrs
            elif args.test == 'online_raw':
                pass
            elif args.test == 'online_small':
                current_lrs = adapt_lrs / 10
            elif args.test == 'online_exp':
                current_lrs = adapt_lrs * (0.5 ** i) * 0.6667
            elif args.test == 'online':
                state_now = model.state_dict()
                state_start = wavg_state(state, state_now, 0.5)
                model.load_state_dict(state_start)
                current_lrs = adapt_lrs * 0.5
            elif args.test == 'online_ha':
                state_now = model.state_dict()
                state_start = wavg_state(state, state_now, 1 / (i + 1))
                model.load_state_dict(state_start)
                current_lrs = adapt_lrs / (i + 1)
            elif args.test == 'online_avg':
                if i == 0:
                    acc_state = deepcopy(state)
                else:
                    acc_state = deepcopy(model.state_dict())
                model.load_state_dict(state)
            else:
                model.load_state_dict(state)

            # to device
            X = [x.to(self.device) for x in X]
            Y = Y.to(self.device)

            # 1) TTA step
            self.adapt_one_step(model, current_lrs, X, Y, unspv_loss_func, args)

            # online_avg: cumulative smoothing
            if args.test == 'online_avg':
                state_now = model.state_dict()
                state_new = wavg_state(acc_state, state_now, i / (i + 1))
                model.load_state_dict(state_new)

            # 2) supervised eval
            model.eval()
            out = model(*X)
            logits = _unwrap_logits(out)
            with torch.no_grad():
                spv_loss = spv_loss_func(logits, Y)

                num_examples = len(X[0])
                total_examples += num_examples
                total_loss += spv_loss.item() * num_examples

                metric = metric_func(logits, Y)  # computes bACC internally using tau
                total_metric += metric.item() * num_examples

                if return_logits:
                    cached_logits.append(logits.detach().cpu())
                    cached_labels.append(Y.detach().cpu())

                # Only check bACC alignment manually on first batch
                if getattr(args, "debug", 1) and i == 0 and getattr(args, "eval_thresholds", None) is not None:
                    taus = torch.tensor(getattr(args, 'eval_thresholds'), device=logits.device, dtype=torch.float32)
                    probs = torch.sigmoid(logits)
                    pred = (probs >= taus).float()
                    y = Y.float()
                    tp = (pred * y).sum(dim=0)
                    fn = ((1 - pred) * y).sum(dim=0)
                    tn = ((1 - pred) * (1 - y)).sum(dim=0)
                    fp = (pred * (1 - y)).sum(dim=0)
                    tpr = tp / (tp + fn + 1e-8)
                    tnr = tn / (tn + fp + 1e-8)
                    bacc_manual = (tpr + tnr) / 2
                    print(f"[CHECK][client {self.cid}] metric_func={metric.item():.4f} "
                          f"manual_bacc_mean={bacc_manual.mean().item():.4f} "
                          f"(min={bacc_manual.min().item():.4f}, max={bacc_manual.max().item():.4f})")

        avg_loss = total_loss / max(total_examples, 1)
        avg_metric = total_metric / max(total_examples, 1)

        if return_logits:
            logits_cpu = torch.cat(cached_logits, dim=0) if cached_logits else torch.empty(0)
            labels_cpu = torch.cat(cached_labels, dim=0) if cached_labels else torch.empty(0)
            return avg_loss, avg_metric, num_data, logits_cpu, labels_cpu
        else:
            return avg_loss, avg_metric, num_data
