import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from copy import deepcopy

from model import create_model, create_loss, create_metric, create_optimizer
from model.MyBatchNorm2d import MyBatchNorm2d

from .Base import BaseServer, BaseClient
from .TTPBase import TTPBaseServer

#=======
from utils import pickle_save
from algorithm.LabelShiftEM import BinaryVectorScaling, BCTS, VS

#=======

# ---------- Utility: convert labels to [N,C] multi-hot ----------
def _as_multihot(y: torch.Tensor, C: int) -> torch.Tensor:
    """
    y: [N] (single-label, range 0..C-1) or [N,C] (multi-label 0/1)
    Returns: [N,C] (float)
    """
    if y.dim() == 1:
        return F.one_hot(y.long(), num_classes=C).float()
    if y.dim() == 2 and y.size(1) == C:
        return y.float()
    raise ValueError(f"Unexpected y shape: {tuple(y.shape)}; expect [N] or [N,{C}]")

#------------------------------------------------------------------

class ATPServer(TTPBaseServer):
    """
    A class for debugging
    """

    def __init__(self, train_datasets, test_datasets, args):
        TTPBaseServer.__init__(self, train_datasets, test_datasets, args)

        self.train_clients = {cid: ATPClient(cid, datasets, args) for cid, datasets in train_datasets.items()}
        self.test_clients = {cid: ATPClient(cid, datasets, args) for cid, datasets in test_datasets.items()}

        # load a pre-trained model
        self.model = create_model(args)

        self.model.change_bn(mode='grad')  # replace the nn.BatchNorm2d to our BatchNorm,
        # which has identical behavior, but support taking gradient
        self.model.eval()

        num_ar = len([name for name, params in self.model.named_parameters() if params.requires_grad])
        print('Dimension of Adapt Rate:', num_ar)

        args.idx_params = [i for i, (name, params) in enumerate(self.model.named_parameters()) if 'running' not in name]
        args.idx_stats = [i for i, (name, params) in enumerate(self.model.named_parameters()) if 'running' in name]

        print('  - Params:', len(args.idx_params))
        print('  - Stats:', len(args.idx_stats))

        if args.verbose:
            print([name for name, params in self.model.named_parameters() if params.requires_grad])

        self.adapt_lrs = torch.zeros(len(self.model.trainable_parameters())).to(args.device)

#=========
    def run(self, args):

        # If TTPBaseServer has train/val flow
        if hasattr(super(), "run"):
            super().run(args)

        # Directory fallback
        import os
        # if getattr(args, "history_path", "none") == "none":
        #     args.history_path = os.path.join(
        #         os.path.dirname(os.path.abspath(__file__)),
        #         "../../history/cifar10/label/atp_resnet18_pseed_0_seed_0.pkl"
        #     )
        args.history_path = os.path.normpath(args.history_path)

        print(f"[CAL] will write calibration & prior into: {args.history_path}")

        # Execute calibration + save
        self.calibrate_and_save(args)
    
        # Print key fields that were written
        hist = self.history.data
        cal = hist.get("calibration", {})
        prior = hist.get("source_prior", None)
        taus = hist.get("src_thresholds", None)
        print(f"[CAL][done] type={cal.get('type', None)} | "
            f"prior_len={len(prior) if prior else 0} | "
            f"taus_len={len(taus) if taus is not None else 0}")


    # ---- Collect source-domain validation logits/labels ----
    @torch.no_grad()
    def collect_source_calib(self, max_clients=5, split='test', multi_label=True):
        self.model.eval()
        device = next(self.model.parameters()).device
        logits_all, y_all = [], []
        cnt = 0
        for cid, client in self.train_clients.items():
            dl = client.dataloaders[split] 
            for *X, Y in dl:
                X = [x.to(device).float() for x in X]
                logits = self.model(*X)            # [B,C]
                logits_all.append(logits.cpu())
                y_all.append(Y.cpu())
            cnt += 1
            if cnt >= max_clients:
                break
        logits = torch.cat(logits_all, dim=0)
        Y = torch.cat(y_all, dim=0)
        return logits, Y

    def _export_calib_plot_data(self, logits, y_true, p_src, args):
        """
        Export data needed for calibration plots:
        - conf_raw: uncalibrated sigmoid probabilities (flatten)
        - conf_cal: BVS-calibrated probabilities (flatten)
        - y_flat : multi-label ground-truth (flatten, 0/1)

        Save as a .npz file.
        """
        # Move to CPU then convert to numpy
        logits_cpu = logits.detach().cpu()
        y_cpu      = y_true.detach().cpu()
        p_src_cpu  = p_src.detach().cpu()

        # Uncalibrated probabilities = sigmoid(logits)
        conf_raw = torch.sigmoid(logits_cpu).reshape(-1).numpy()
        # Calibrated probabilities
        conf_cal = p_src_cpu.reshape(-1).numpy()
        # Multi-label GT
        y_flat   = y_cpu.reshape(-1).numpy()

        # Output path: same directory as history_path, same name with suffix
        base_dir  = os.path.dirname(os.path.abspath(args.history_path))
        base_name = os.path.splitext(os.path.basename(args.history_path))[0]
        out_path  = os.path.join(base_dir, f"{base_name}_calib_plot_src.npz")

        np.savez(out_path,
                 conf_raw=conf_raw,
                 conf_cal=conf_cal,
                 y=y_flat)

        print(f"[CAL-PLOT] exported calib plot data to: {out_path}")
        print(f"[CAL-PLOT] #points = {len(y_flat)}, "
              f"mean(conf_raw)={conf_raw.mean():.4f}, mean(conf_cal)={conf_cal.mean():.4f}")
    def calibrate_and_save(self, args):
        # 1) Collect source-domain logits / labels
        logits, Y = self.collect_source_calib(max_clients=5, split='test', multi_label=True)
        C = logits.shape[1]
        y_true = _as_multihot(Y, C)
        # 2) Fit calibrator & estimate source prior =====
        calib_type = getattr(args, "calibration", "bvs")
        calib_payload = {"type": calib_type, "params": {}}

        if calib_type == "bvs":
            calib = BinaryVectorScaling(C).to(logits.device)
            calib.fit(logits,y_true)

            # y_true_onehot = F.one_hot(Y.long(), num_classes=8).float()
            # calib.fit(logits, y_true_onehot)

            p_src = calib.predict_proba(logits)  # source-domain calibrated probabilities
            p_y_src = p_src.mean(dim=0).detach().cpu().numpy().tolist()
            calib_payload["params"] = {
                "a": calib.a.detach().cpu().numpy().tolist(),
                "b": calib.b.detach().cpu().numpy().tolist(),
            }
        elif calib_type == "bcts":
            calib = BCTS(C).to(logits.device)
            calib.fit(logits, Y.long())
            p_src = calib.predict_proba(logits)
            p_y_src = p_src.mean(dim=0).detach().cpu().numpy().tolist()
            calib_payload["params"] = {
                "T": calib.T.detach().cpu().numpy().tolist(),
                "bias": calib.bias.detach().cpu().numpy().tolist(),
            }
        elif calib_type == "vs":
            calib = VS(C).to(logits.device)
            calib.fit(logits, Y.long())
            p_src = calib.predict_proba(logits)
            p_y_src = p_src.mean(dim=0).detach().cpu().numpy().tolist()
            calib_payload["params"] = {
                "W": calib.W.detach().cpu().numpy().tolist(),
                "b": calib.b.detach().cpu().numpy().tolist(),
            }
        else:
            # No calibration: use sigmoid probabilities directly as source-domain probabilities
            calib = None
            p_src = torch.sigmoid(logits)
            p_y_src = p_src.mean(dim=0).detach().cpu().numpy().tolist()
        # Generate plots
        if getattr(args, "save_calib_plot", 1):
            self._export_calib_plot_data(logits, y_true, p_src, args)
        # Method: maximize per-class bACC (0.5*(TPR+TNR)) over a grid of 199 thresholds in [0.01, 0.99]
        def find_src_thresholds(probs: torch.Tensor, y: torch.Tensor):
            """
            probs: [N, C] probabilities
            y:     [N] (single-label classes) or [N, C] (multi-label one-hot)
            """
            device = probs.device
            N, C = probs.shape

            # If single-label, convert to one-hot
            if y.dim() == 1:
                y = F.one_hot(y.long(), num_classes=C).float().to(device)
            elif y.dim() == 2 and y.size(1) == C:
                y = y.float().to(device)
            else:
                raise ValueError(f"Unexpected y shape: {tuple(y.shape)}; expect [N] or [N,{C}]")

            grid = torch.linspace(0.01, 0.99, steps=199, device=device)  # [T]
            T = grid.numel()

            taus = torch.empty(C, device=device)
            with torch.no_grad():
                for c in range(C):
                    p = probs[:, c].view(N, 1).expand(N, T)      # [N, T]
                    pred = (p >= grid.view(1, T)).float()        # [N, T]
                    y_c = y[:, c].view(N, 1)                     # [N, 1]

                    tp = (pred * y_c).sum(dim=0)
                    fn = ((1 - pred) * y_c).sum(dim=0)
                    tn = ((1 - pred) * (1 - y_c)).sum(dim=0)
                    fp = (pred * (1 - y_c)).sum(dim=0)

                    tpr = tp / (tp + fn + 1e-8)
                    tnr = tn / (tn + fp + 1e-8)
                    bacc = 0.5 * (tpr + tnr)                     # [T]

                    best_idx = torch.argmax(bacc)
                    taus[c] = grid[best_idx]
            return taus

        src_taus = find_src_thresholds(p_src, y_true)  # [C]
        print(f"[TAU][train] src_thresholds mean={float(src_taus.mean()):.4f}")

        # ===== 4) Write to history and persist =====
        self.history.data["calibration"] = calib_payload
        self.history.data["source_prior"] = p_y_src
        self.history.data["src_thresholds"] = src_taus.detach().cpu().numpy().tolist()

        args.eval_thresholds = src_taus.detach().cpu().tolist()
        print(f"[CAL] thresholds saved to args.eval_thresholds (len={len(args.eval_thresholds)})")

        if args.history_path != 'none':
            content = {"args": args, "history": self.history.data}
            pickle_save(content, args.history_path, mode='ab')

        print("[INFO] Calibration, source_prior, src_thresholds saved into history.data")



#=========
class ATPClient(BaseClient):
    """
    A class for debug
    """

    def __init__(self, cid, datasets, args):
        BaseClient.__init__(self, cid, datasets, args)

        self.lr = args.lm_lr  # the learning rate of adaptation rates

    def adapt_one_step(self, model, adapt_lrs, X, Y, unspv_loss_func, args):

        model.eval()

        logits = model(*X)

        loss = unspv_loss_func(logits, Y)

        loss.backward()

        model.set_running_stat_grads()

        unspv_grad = [p.grad.clone() for p in model.trainable_parameters()]

        with torch.no_grad():
            for i, (p, g) in enumerate(zip(model.trainable_parameters(), unspv_grad)):
                p -= adapt_lrs[i] * g

        model.zero_grad()

        model.clip_bn_running_vars()  # some BN running vars may be smaller than 0, which cause NaN problem.

        return unspv_grad

    def local_train(self, model, adapt_lrs, args, dataset='test'):

        unspv_loss_func = create_loss('ent')
        spv_loss_func = create_loss('ral')
        metric_func = create_metric('final', thresholds=getattr(args, 'eval_thresholds', None))

        total_examples, total_loss, total_metric = 0, 0, 0

        dataloader = self.dataloaders[dataset]
        num_data = self.num_data[dataset]

        state = deepcopy(model.state_dict())

        for b_idx, (*X, Y) in enumerate(dataloader):
            model.load_state_dict(state)
            X = [x.to(self.device) for x in X]
            Y = Y.to(self.device)

            # ---- Print shape once on first batch to confirm label format ----
            if total_examples == 0:
                with torch.no_grad():
                    tmp_logits = model(*X)
                # print(f"[DEBUG][train] logits shape={tuple(tmp_logits.shape)}  "
                    #   f"Y shape={tuple(Y.shape)}  Y(min,max)=({float(Y.min())},{float(Y.max())})")

            # 1. unsupervised adaptation

            unspv_grad = self.adapt_one_step(model, adapt_lrs, X, Y, unspv_loss_func, args)

            # 2. supervised evaluation

            model.eval()

            logits = model(*X)
            spv_loss = spv_loss_func(logits, Y)

            spv_grad = torch.autograd.grad(spv_loss, model.trainable_parameters())

            # 3. update the adaptation rate
            with torch.no_grad():

                # manual resize

                if args.grad_norm == 'none':
                    g = torch.zeros_like(adapt_lrs)
                    for i, (g1, g2) in enumerate(zip(spv_grad, unspv_grad)):
                        g[i] += (g1 * g2).sum()

                elif args.grad_norm == 'numel':
                    g = torch.zeros_like(adapt_lrs)
                    l = torch.zeros_like(adapt_lrs)
                    for i, (g1, g2) in enumerate(zip(spv_grad, unspv_grad)):
                        g[i] += (g1 * g2).sum()
                        l[i] += g1.numel()

                    g /= l

                elif args.grad_norm == 'sqrt_numel':
                    g = torch.zeros_like(adapt_lrs)
                    l = torch.zeros_like(adapt_lrs)
                    for i, (g1, g2) in enumerate(zip(spv_grad, unspv_grad)):
                        g[i] += (g1 * g2).sum()
                        l[i] += g1.numel()

                    g /= torch.sqrt(l)

                elif args.grad_norm == 'manual':
                    g = torch.zeros_like(adapt_lrs)
                    for i, (g1, g2) in enumerate(zip(spv_grad, unspv_grad)):
                        if i in args.idx_params:
                            g[i] += (g1 * g2).sum()
                        elif i in args.idx_stats:
                            g[i] += 100 * (g1 * g2).sum()

                elif args.grad_norm == 'params_only':
                    g = torch.zeros_like(adapt_lrs)
                    for i, (g1, g2) in enumerate(zip(spv_grad, unspv_grad)):
                        if i in args.idx_params:
                            g[i] += (g1 * g2).sum()
                            g[i] /= g1.numel()

                elif args.grad_norm == 'stats_only':
                    g = torch.zeros_like(adapt_lrs)
                    for i, (g1, g2) in enumerate(zip(spv_grad, unspv_grad)):
                        if i in args.idx_stats:
                            g[i] += (g1 * g2).sum()
                            g[i] /= g1.numel()

                else:
                    raise NotImplementedError

                adapt_lrs += self.lr * g

            with torch.no_grad():
                num_examples = len(X[0])
                total_examples += num_examples
                total_loss += spv_loss.item() * num_examples
                metric = metric_func(logits, Y)
                total_metric += metric.item() * num_examples

        avg_loss, avg_metric = total_loss / total_examples, total_metric / total_examples

        return avg_loss, avg_metric, num_data

    def local_eval(self, model, adapt_lrs, args, dataset='test'):

        unspv_loss_func = create_loss('ent')
        spv_loss_func = create_loss('ral')
        # metric_func = create_metric('bacc')
        metric_func = create_metric('final', thresholds=getattr(args, 'eval_thresholds', None))

        total_examples, total_loss, total_metric = 0, 0, 0

        dataloader = self.dataloaders[dataset]
        num_data = self.num_data[dataset]
#============
        # print(f"[DEBUG] Client {self.cid}, Dataset '{dataset}', num_data: {num_data}, len(dataloader): {len(dataloader)}")
        # if len(dataloader) == 0 or num_data == 0:
        #     print(f"[WARNING] Client {self.cid} has no data for dataset '{dataset}'. Skipping local_train.")
        #     return 0.0, 0.0, 0
#============
        state = deepcopy(model.state_dict())

        for i, (*X, Y) in enumerate(dataloader):
            model.load_state_dict(state)

            # Get a batch of data
            X = [x.to(self.device) for x in X]
            Y = Y.to(self.device)

            # Print once on first batch to confirm label format
            if total_examples == 0:
                with torch.no_grad():
                    tmp_logits = model(*X)
                # print(f"[DEBUG][eval]  logits shape={tuple(tmp_logits.shape)}  "
                    #   f"Y shape={tuple(Y.shape)}  Y(min,max)=({float(Y.min())},{float(Y.max())})")

            # 1. unsupervised adaptation

            self.adapt_one_step(model, adapt_lrs, X, Y, unspv_loss_func, args)

            # 2. supervised evaluation

            model.eval()

            with torch.no_grad():
                logits = model(*X)
                spv_loss = spv_loss_func(logits, Y)

                # record the loss and accuracy
                num_examples = len(X[0])
                total_examples += num_examples
                total_loss += spv_loss.item() * num_examples
                metric = metric_func(logits, Y)
                total_metric += metric.item() * num_examples

        avg_loss, avg_metric = total_loss / total_examples, total_metric / total_examples

        return avg_loss, avg_metric, num_data
