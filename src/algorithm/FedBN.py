import torch
from copy import deepcopy
from tqdm import tqdm

from model import create_model, create_loss, create_metric
from .Base import BaseServer
from .FedAvg import FedAvgClient


class FedBNServer(BaseServer):
    """
    FedBN: average NON-BN params across clients; keep BN params unaggregated.
    """

    def __init__(self, train_datasets, test_datasets, args):
        super(FedBNServer, self).__init__(train_datasets, test_datasets, args)

        assert args.gm_opt == 'sgd'
        assert args.gm_lr == 1.0
        self.gm_rounds = args.gm_rounds

        self.cohort_size = max(1, round(self.num_train_clients * args.part_rate))

        # FedBN still uses FedAvgClient for local training (interchangeable)
        self.train_clients = {
            cid: FedAvgClient(cid, datasets, args) for cid, datasets in train_datasets.items()
        }
        self.test_clients = {
            cid: FedAvgClient(cid, datasets, args) for cid, datasets in test_datasets.items()
        }

        self.model = create_model(args)

    def _is_bn_param(self, key: str) -> bool:
        """
        Check whether a state_dict key is a BN parameter or buffer.
        """
        key_lower = key.lower()
        if 'running_mean' in key_lower or 'running_var' in key_lower:
            return True
        if 'bn' in key_lower:         # e.g. layer1.0.bn1.weight
            return True
        if 'bias' in key_lower and 'bn' in key_lower:
            return True
        return False

    def run(self, args):
        for rnd in range(1, self.gm_rounds + 1):
            tqdm.write('Round(FedBN): %d / %d' % (rnd, self.gm_rounds))
            self.train(self.model, args)

            if rnd % 20 == 0:
                self.eval_part(self.model, args)
                self.eval_unpart(self.model, args)

    def train(self, model, args):
        # Current global model
        global_state = deepcopy(model.updated_state_dict())

        next_state = None
        weights, losses, metrics = [], [], []

        # Sample a subset of clients
        selected_idxs = sorted(list(torch.randperm(self.num_train_clients)[:self.cohort_size].numpy()))
        selected_cids = [self.train_idx2cid[idx] for idx in selected_idxs]

        for cid in tqdm(selected_cids):
            client = self.train_clients[cid]
            model.load_state_dict(global_state, strict=False)  # Load from current global

            loss, metric, num_data = client.local_train(model, args, 'train')
            local_state = model.updated_state_dict()

            weights.append(num_data)
            losses.append(loss)
            metrics.append(metric)

            if next_state is None:
                next_state = deepcopy(local_state)
                for k in next_state.keys():
                    next_state[k] = torch.mul(local_state[k], num_data)
            else:
                for k in next_state.keys():
                    next_state[k] += torch.mul(local_state[k], num_data)

        sum_weight = sum(weights)
        agg_loss = sum([w * l for w, l in zip(weights, losses)]) / sum_weight
        agg_metric = sum([w * m for w, m in zip(weights, metrics)]) / sum_weight
        tqdm.write('\t Train(FedBN): Loss: %.4f \t Metric: %.4f' % (agg_loss, agg_metric))

        # FedBN aggregation: weighted average BN params, non-BN params use global_state
        for k in next_state.keys():
            if self._is_bn_param(k):
                # Keep the previous round's global BN (unaggregated)
                next_state[k] = global_state[k]
            else:
                next_state[k] = torch.div(next_state[k], sum_weight)

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

    # Below: directly ported from the previous multi-metric version
    def _eval_clients_all_metrics(self, model, args, clients, desc='Eval'):
        loss_func = create_loss(args.loss)
        device = args.device
        model.eval()

        all_logits = []
        all_targets = []
        total_loss = 0.0
        total_examples = 0

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
        self._eval_clients_all_metrics(model, args, self.train_clients, desc='Eval(part)')

    def eval_unpart(self, model, args):
        self._eval_clients_all_metrics(model, args, self.test_clients, desc='Eval(unpart)')
