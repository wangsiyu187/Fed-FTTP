import torch
from copy import deepcopy
from tqdm import tqdm

from model import create_model, create_loss, create_metric, create_optimizer
from .Base import BaseServer, BaseClient


class PerFedAvgServer(BaseServer):
    def __init__(self, train_datasets, test_datasets, args):
        super().__init__(train_datasets, test_datasets, args)

        self.gm_rounds = args.gm_rounds
        self.cohort_size = max(1, round(self.num_train_clients * args.part_rate))

        self.train_clients = {
            cid: PerFedAvgClient(cid, datasets, args)
            for cid, datasets in train_datasets.items()
        }
        self.test_clients = {
            cid: PerFedAvgClient(cid, datasets, args)
            for cid, datasets in test_datasets.items()
        }

        self.model = create_model(args)

    def run(self, args):
        for rnd in range(1, self.gm_rounds + 1):
            tqdm.write(f'Round: {rnd} / {self.gm_rounds}')
            self.train(self.model, args)

            if rnd % 20 == 0:
                self.eval_part(self.model, args)
                self.eval_unpart(self.model, args)

    def train(self, model, args):
        # Current global model
        global_state = deepcopy(model.updated_state_dict())
        next_state = None
        weights, losses, metrics = [], [], []

        # Sample clients
        selected_idxs = sorted(
            list(torch.randperm(self.num_train_clients)[:self.cohort_size].numpy())
        )
        selected_cids = [self.train_idx2cid[idx] for idx in selected_idxs]

        for cid in tqdm(selected_cids):
            client = self.train_clients[cid]

            # Start from global model
            model.load_state_dict(global_state, strict=False)

            # local_train does inner + outer, two-stage update
            loss, metric, num_data, local_state = client.local_train(
                model, args, dataset='train'
            )

            weights.append(num_data)
            losses.append(loss)
            metrics.append(metric)

            # FedAvg-style aggregation
            if next_state is None:
                next_state = deepcopy(local_state)
                for k in next_state.keys():
                    next_state[k] = local_state[k] * num_data
            else:
                for k in next_state.keys():
                    next_state[k] += local_state[k] * num_data

        sum_weight = sum(weights)
        agg_loss = sum(w * l for w, l in zip(weights, losses)) / sum_weight
        agg_metric = sum(w * m for w, m in zip(weights, metrics)) / sum_weight
        tqdm.write(f'\t Train: Loss: {agg_loss:.4f} \t Metric: {agg_metric:.4f}')

        for k in next_state.keys():
            next_state[k] = next_state[k] / sum_weight
        model.load_state_dict(next_state)

        log_dict = {
            'train_selected_idxs': selected_idxs,
            'train_selected_cids': selected_cids,
            'train_losses': losses,
            'train_metrics': metrics,
            'train_wavg_loss': agg_loss,
            'train_wavg_metric': agg_metric,
        }
        self.history.append(log_dict)

    # ===== Multi-metric evaluation helper (mirrors FedAvgServer) =====
    def _eval_clients_all_metrics(self, model, args, clients, desc='Eval'):
        """
        Evaluate on a set of clients: bACC, Kappa, F1, AUC, Final
        - clients: self.train_clients or self.test_clients
        - desc: label to distinguish part vs unpart in logs
        """
        loss_func = create_loss(args.loss)
        device = args.device
        model.eval()

        all_logits = []
        all_targets = []
        total_loss = 0.0
        total_examples = 0

        # Aggregate test data from all clients
        for cid, client in tqdm(clients.items(), desc=f'{desc} clients'):
            dataloader = client.dataloaders['test']

            for *X, Y in dataloader:
                # Consistent with local_train
                if hasattr(model, "drop_last") and model.drop_last and len(Y) < args.batch_size:
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

        # Multi-metric
        metric_bacc  = create_metric('bacc')
        metric_kappa = create_metric('kappa')
        metric_f1    = create_metric('f1')
        metric_auc   = create_metric('auc')
        metric_final = create_metric('final')

        bacc  = float(metric_bacc(all_logits,  all_targets))
        kappa = float(metric_kappa(all_logits, all_targets))
        f1    = float(metric_f1(all_logits,    all_targets))
        auc   = float(metric_auc(all_logits,   all_targets))
        final = float(metric_final(all_logits, all_targets))

        avg_loss = total_loss / total_examples

        tqdm.write(
            f'\t {desc}: Loss: {avg_loss:.4f}  '
            f'bACC: {bacc:.4f}  Kappa: {kappa:.4f}  '
            f'F1: {f1:.4f}  AUC: {auc:.4f}  Final: {final:.4f}'
        )

        log_dict = {
            f'{desc}_loss':  avg_loss,
            f'{desc}_bacc':  bacc,
            f'{desc}_kappa': kappa,
            f'{desc}_f1':    f1,
            f'{desc}_auc':   auc,
            f'{desc}_final': final,
        }
        self.history.append(log_dict)

        return log_dict

    def eval_part(self, model, args):
        """
        Evaluate on participating clients (source clients) with 4 metrics.
        """
        self._eval_clients_all_metrics(model, args, self.train_clients, desc='Eval(part)')

    def eval_unpart(self, model, args):
        """
        Evaluate on unseen clients (target clients) with 4 metrics.
        """
        self._eval_clients_all_metrics(model, args, self.test_clients, desc='Eval(unpart)')


class PerFedAvgClient(BaseClient):
    def __init__(self, cid, datasets, args):
        super().__init__(cid, datasets, args)

    def local_train(self, model, args, dataset='train'):
        """
        A simplified Per-FedAvg variant:
        1) inner loop: 1 round of fast adaptation from w_global (not directly communicated)
        2) outer loop: 1 additional training round on post-inner weights for meta-update, contribute final local_state to server
        """
        loss_func = create_loss(args.loss)
        metric_func = create_metric(args.metric)
        batch_size = args.batch_size
        dataloader = self.dataloaders[dataset]
        num_data = self.num_data[dataset]

        # ==== inner loop: one round of fast adaptation on w_global ====
        inner_lr = args.lm_lr
        inner_epochs = 1

        # Copy parameters
        inner_state = deepcopy(model.updated_state_dict())

        model.train()
        for epoch in range(inner_epochs):
            for *X, Y in dataloader:
                if hasattr(model, "drop_last") and model.drop_last and len(Y) < batch_size:
                    continue
                X = [x.to(self.device) for x in X]
                Y = Y.to(self.device)

                # Load from inner_state once per batch
                model.load_state_dict(inner_state, strict=False)
                model.zero_grad()
                logits = model(*X)
                loss = loss_func(logits, Y)
                loss.backward()

                # One SGD step to manually update inner_state
                with torch.no_grad():
                    for name, p in model.named_parameters():
                        if p.grad is None:
                            continue
                        if name in inner_state:  # only update tracked weights
                            inner_state[name] = p.data - inner_lr * p.grad
            # Only run one epoch here

        # ==== outer loop: use inner_state on new data for 1 epoch of "meta update" ====
        model.load_state_dict(inner_state, strict=False)
        outer_optimizer = create_optimizer(model, args.lm_opt, args.lm_lr)
        outer_epochs = 1

        total_examples = 0
        total_loss = 0.0
        total_metric = 0.0

        for epoch in range(outer_epochs):
            for *X, Y in dataloader:
                if hasattr(model, "drop_last") and model.drop_last and len(Y) < batch_size:
                    continue
                X = [x.to(self.device) for x in X]
                Y = Y.to(self.device)

                logits = model(*X)
                loss = loss_func(logits, Y)
                outer_optimizer.zero_grad()
                loss.backward()
                outer_optimizer.step()

                with torch.no_grad():
                    num_examples = len(X[0])
                    total_examples += num_examples
                    total_loss += loss.item() * num_examples
                    metric = metric_func(logits, Y)
                    if isinstance(metric, torch.Tensor):
                        m_val = metric.item()
                    else:
                        m_val = float(metric)
                    total_metric += m_val * num_examples

        avg_loss = total_loss / total_examples
        avg_metric = total_metric / total_examples

        local_state = model.updated_state_dict()
        return avg_loss, avg_metric, num_data, local_state
