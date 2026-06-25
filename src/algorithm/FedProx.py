import torch
from copy import deepcopy
from tqdm import tqdm

from model import create_loss, create_metric, create_optimizer
from .Base import BaseServer, BaseClient


class FedProxServer(BaseServer):
    """
    Server of FedProx (FedAvg + proximal term in local training)
    """

    def __init__(self, train_datasets, test_datasets, args):
        super(FedProxServer, self).__init__(train_datasets, test_datasets, args)

        assert args.gm_opt == 'sgd'
        assert args.gm_lr == 1.0
        self.gm_rounds = args.gm_rounds
        self.mu = args.prox_mu

        # sample a subset of clients per communication round
        self.cohort_size = max(1, round(self.num_train_clients * args.part_rate))

        # init clients (using FedProxClient)
        self.train_clients = {
            cid: FedProxClient(cid, datasets, args) for cid, datasets in train_datasets.items()
        }
        self.test_clients = {
            cid: FedProxClient(cid, datasets, args) for cid, datasets in test_datasets.items()
        }

        # model = global model
        from model import create_model
        self.model = create_model(args)

    def run(self, args):
        """
        Run the training and testing pipeline
        """
        for rnd in range(1, self.gm_rounds + 1):
            tqdm.write('Round: %d / %d' % (rnd, self.gm_rounds))
            self.train(self.model, args)

            if rnd % 20 == 0:
                self.eval_part(self.model, args)
                self.eval_unpart(self.model, args)

    def train(self, model, args):
        """
        One communication round (same as FedAvg, but FedProxClient computes proximal loss internally)
        """
        # current global model
        global_state = deepcopy(model.updated_state_dict())

        next_state = None
        weights = []
        losses = []
        metrics = []

        # sample a subset of clients
        selected_idxs = sorted(list(torch.randperm(self.num_train_clients)[:self.cohort_size].numpy()))
        selected_cids = [self.train_idx2cid[idx] for idx in selected_idxs]

        # iterate randomly selected clients
        for cid in tqdm(selected_cids):
            client = self.train_clients[cid]
            model.load_state_dict(global_state, strict=False)  # start from global model

            loss, metric, num_data = client.local_train(model, args, dataset='train')
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

        # train loss and metric
        sum_weight = sum(weights)
        agg_loss = sum([w * l for w, l in zip(weights, losses)]) / sum_weight
        agg_metric = sum([w * m for w, m in zip(weights, metrics)]) / sum_weight
        tqdm.write('\t Train(FedProx): Loss: %.4f \t Metric: %.4f' % (agg_loss, agg_metric))

        # aggregate
        for k in next_state.keys():
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

    # -------- Below: directly ported from the multi-metric version --------
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

        from model import create_metric
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


class FedProxClient(BaseClient):
    """
    Client of FedProx
    """

    def __init__(self, cid, datasets, args):
        super(FedProxClient, self).__init__(cid, datasets, args)
        self.mu = args.prox_mu

    def local_train(self, model, args, dataset='train'):
        """
        Local training with proximal term:
            loss = task_loss + (mu/2) * ||w - w_global||^2
        Store w_global as the model parameters before this round's training
        """

        loss_func = create_loss(args.loss)
        metric_func = create_metric(args.metric)
        optimizer = create_optimizer(model, args.lm_opt, args.lm_lr)
        num_epochs = args.lm_epochs
        batch_size = args.batch_size

        dataloader = self.dataloaders[dataset]
        num_data = self.num_data[dataset]

        # Record global parameters before this round for the proximal term
        global_params = {
            name: p.detach().clone().to(self.device)
            for name, p in model.named_parameters()
        }

        model.train()
        total_examples, total_loss, total_metric = 0, 0.0, 0.0

        for epoch in range(num_epochs):
            for *X, Y in dataloader:

                if model.drop_last and len(Y) < batch_size:
                    continue

                X = [x.to(self.device) for x in X]
                Y = Y.to(self.device)

                logits = model(*X)
                task_loss = loss_func(logits, Y)

                # Compute proximal term
                prox_term = 0.0
                if self.mu > 0:
                    for name, param in model.named_parameters():
                        prox_term = prox_term + torch.sum(
                            (param - global_params[name]) ** 2
                        )
                    prox_term = 0.5 * self.mu * prox_term
                    loss = task_loss + prox_term
                else:
                    loss = task_loss

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                with torch.no_grad():
                    num_examples = len(X[0])
                    total_examples += num_examples

                    total_loss += loss.item() * num_examples

                    metric = metric_func(logits, Y)
                    total_metric += metric.item() * num_examples

        avg_loss = total_loss / total_examples
        avg_metric = total_metric / total_examples

        return avg_loss, avg_metric, num_data
