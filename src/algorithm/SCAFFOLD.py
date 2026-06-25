"""
SCAFFOLD: Stochastic Controlled Averaging for Federated Learning
"""

import torch
from copy import deepcopy
from tqdm import tqdm

from model import create_model, create_loss, create_metric, create_optimizer
from .Base import BaseServer, BaseClient


class SCAFFOLDServer(BaseServer):
    def __init__(self, train_datasets, test_datasets, args):
        super().__init__(train_datasets, test_datasets, args)

        assert args.gm_opt == 'sgd'
        assert args.gm_lr == 1.0
        self.gm_rounds = args.gm_rounds

        # Number of clients per round
        self.cohort_size = max(1, round(self.num_train_clients * args.part_rate))

        # Initialize clients
        self.train_clients = {
            cid: SCAFFOLDClient(cid, datasets, args)
            for cid, datasets in train_datasets.items()
        }
        self.test_clients = {
            cid: SCAFFOLDClient(cid, datasets, args)
            for cid, datasets in test_datasets.items()
        }

        # Global model
        self.model = create_model(args)

        # ====== Initialize control variates only for trainable parameters (ref. Long buffer) ======
        self.param_names = [name for name, _ in self.model.named_parameters()]
        self.c_global = {
            name: torch.zeros_like(param.data, device=args.device)
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

    def run(self, args):
        for rnd in range(1, self.gm_rounds + 1):
            tqdm.write('Round: %d / %d' % (rnd, self.gm_rounds))
            self.train(self.model, args)

            if rnd % 20 == 0:
                self.eval_part(self.model, args)
                self.eval_unpart(self.model, args)

    def train(self, model, args):
        # Current global model params (full state_dict for FedAvg aggregation)
        global_state = deepcopy(model.updated_state_dict())
        next_state = None

        weights, losses, metrics = [], [], []

        # Select clients
        selected_idxs = sorted(
            list(torch.randperm(self.num_train_clients)[:self.cohort_size].numpy())
        )
        selected_cids = [self.train_idx2cid[idx] for idx in selected_idxs]

        # Weighted accumulation for updating c_global
        delta_c_global = {
            name: torch.zeros_like(t, device=args.device)
            for name, t in self.c_global.items()
        }

        for cid in tqdm(selected_cids):
            client = self.train_clients[cid]
            # Start from global model
            model.load_state_dict(global_state, strict=False)

            # Local training: returns loss / metric / num_samples / delta_c_i
            loss, metric, num_data, delta_c_i = client.local_train(
                model, args, self.c_global, dataset='train'
            )
            local_state = model.updated_state_dict()

            weights.append(num_data)
            losses.append(loss)
            metrics.append(metric)

            # FedAvg-style parameter aggregation
            if next_state is None:
                next_state = deepcopy(local_state)
                for k in next_state.keys():
                    next_state[k] = local_state[k] * num_data
            else:
                for k in next_state.keys():
                    next_state[k] += local_state[k] * num_data

            # Aggregate delta_c_i (all float type, no conflict with long tensors)
            for name in self.c_global.keys():
                delta_c_global[name] += delta_c_i[name] * num_data

        # Compute weighted average loss / metric
        sum_weight = sum(weights)
        agg_loss = sum(w * l for w, l in zip(weights, losses)) / sum_weight
        agg_metric = sum(w * m for w, m in zip(weights, metrics)) / sum_weight
        tqdm.write('\t Train: Loss: %.4f \t Metric: %.4f' % (agg_loss, agg_metric))

        # Aggregate model parameters
        for k in next_state.keys():
            next_state[k] = next_state[k] / sum_weight
        model.load_state_dict(next_state)

        # Update c_global (simple version: sample-weighted average of Δc_i)
        for name in self.c_global.keys():
            self.c_global[name] = self.c_global[name] + (delta_c_global[name] / sum_weight)

        log_dict = {
            'train_selected_idxs': selected_idxs,
            'train_selected_cids': selected_cids,
            'train_losses': losses,
            'train_metrics': metrics,
            'train_wavg_loss': agg_loss,
            'train_wavg_metric': agg_metric,
        }
        self.history.append(log_dict)

    # Evaluation reuses FedAvg's multi-metric method
    def _eval_clients_all_metrics(self, model, args, clients, desc='Eval'):
        from model import create_loss, create_metric

        loss_func = create_loss(args.loss)
        device = args.device
        model.eval()

        all_logits, all_targets = [], []
        total_loss, total_examples = 0.0, 0

        for cid, client in tqdm(clients.items(), desc=f'{desc} clients'):
            dataloader = client.dataloaders['test']
            for *X, Y in dataloader:
                if model.drop_last and len(Y) < args.batch_size:
                    continue
                X = [x.to(device) for x in X]
                Y = Y.to(device)
                with torch.no_grad():
                    logits = model(*X)
                    loss = loss_func(logits, Y)
                num_examples = len(X[0])
                total_examples += num_examples
                total_loss += loss.item() * num_examples
                all_logits.append(logits.cpu())
                all_targets.append(Y.cpu())

        if total_examples == 0:
            tqdm.write(f'\t {desc}: no test data.')
            return None

        all_logits = torch.cat(all_logits, dim=0).to(device)
        all_targets = torch.cat(all_targets, dim=0).to(device)

        metric_bacc = create_metric('bacc')
        metric_kappa = create_metric('kappa')
        metric_f1 = create_metric('f1')
        metric_auc = create_metric('auc')
        metric_final = create_metric('final')

        bacc = float(metric_bacc(all_logits, all_targets))
        kappa = float(metric_kappa(all_logits, all_targets))
        f1 = float(metric_f1(all_logits, all_targets))
        auc = float(metric_auc(all_logits, all_targets))
        final = float(metric_final(all_logits, all_targets))

        avg_loss = total_loss / total_examples

        tqdm.write(
            f'\t {desc}: Loss: {avg_loss:.4f}  '
            f'bACC: {bacc:.4f}  Kappa: {kappa:.4f}  '
            f'F1: {f1:.4f}  AUC: {auc:.4f}  Final: {final:.4f}'
        )

        log_dict = {
            f'{desc}_loss': avg_loss,
            f'{desc}_bacc': bacc,
            f'{desc}_kappa': kappa,
            f'{desc}_f1': f1,
            f'{desc}_auc': auc,
            f'{desc}_final': final,
        }
        self.history.append(log_dict)
        return log_dict

    def eval_part(self, model, args):
        self._eval_clients_all_metrics(model, args, self.train_clients, desc='Eval(part)')

    def eval_unpart(self, model, args):
        self._eval_clients_all_metrics(model, args, self.test_clients, desc='Eval(unpart)')


class SCAFFOLDClient(BaseClient):
    def __init__(self, cid, datasets, args):
        super().__init__(cid, datasets, args)
        self.c_i = None  # local control variate: parameter names only

    def _init_c_i(self, model):
        if self.c_i is None:
            self.c_i = {
                name: torch.zeros_like(param.data, device=self.device)
                for name, param in model.named_parameters()
                if param.requires_grad
            }

    def local_train(self, model, args, c_global, dataset='train'):
        """
        Local training:
        - Use SCAFFOLD corrected gradient g_t = grad - c_i + c_global
        - After training, update c_i using w_before/w_after, return delta_c_i to server
        """
        loss_func = create_loss(args.loss)
        metric_func = create_metric(args.metric)
        num_epochs = args.lm_epochs
        batch_size = args.batch_size

        dataloader = self.dataloaders[dataset]
        num_data = self.num_data[dataset]

        self._init_c_i(model)

        model.train()
        total_examples, total_loss, total_metric = 0, 0.0, 0.0

        # Record parameters at start of round (trainable params only)
        w_before = {
            name: param.data.clone().detach()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        for epoch in range(num_epochs):
            for *X, Y in dataloader:
                if model.drop_last and len(Y) < batch_size:
                    continue

                X = [x.to(self.device) for x in X]
                Y = Y.to(self.device)

                model.zero_grad()
                logits = model(*X)
                loss = loss_func(logits, Y)
                loss.backward()

                # ---- SCAFFOLD corrected gradient: grad - c_i + c_global ----
                with torch.no_grad():
                    for name, p in model.named_parameters():
                        if (not p.requires_grad) or (p.grad is None):
                            continue
                        # All these names are in c_i / c_global as float tensors
                        g = p.grad
                        g_t = g - self.c_i[name] + c_global[name]
                        p.data = p.data - args.lm_lr * g_t

                with torch.no_grad():
                    num_examples = len(X[0])
                    total_examples += num_examples
                    total_loss += loss.item() * num_examples
                    metric = metric_func(logits, Y)
                    # metric may be tensor or float
                    m_val = metric.item() if isinstance(metric, torch.Tensor) else float(metric)
                    total_metric += m_val * num_examples

        avg_loss = total_loss / total_examples
        avg_metric = total_metric / total_examples

        # Parameters after update
        w_after = {
            name: param.data.clone().detach()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        # Estimate local step count
        local_steps = max(1, total_examples // batch_size)

        # Update c_i using w_before/w_after, construct delta_c_i for server aggregation
        delta_c_i = {}
        with torch.no_grad():
            for name in self.c_i.keys():
                # (w_before - w_after) / (eta * steps) ≈ grad - c_i + c_global
                delta_c = (w_before[name] - w_after[name]) / (args.lm_lr * local_steps)
                # Simplified update: approach "grad - c_i + c_global"
                self.c_i[name] = self.c_i[name] + (delta_c - self.c_i[name] + c_global[name])
                delta_c_i[name] = delta_c.clone()

        return avg_loss, avg_metric, num_data, delta_c_i
