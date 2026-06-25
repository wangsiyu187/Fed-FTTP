from tqdm import tqdm
from copy import deepcopy

from model import create_model, create_loss, create_metric, create_optimizer
from utils import History

from .Base import BaseServer, BaseClient
import torch

class TestServer(BaseServer):
    """
    Test a model without updating it.
    """

    def __init__(self, train_datasets, test_datasets, args):
        BaseServer.__init__(self, train_datasets, test_datasets, args)

        # check or set hyperparameters
        assert args.gm_opt == 'sgd'
        assert args.gm_lr == 1.0
        self.gm_rounds = args.gm_rounds

        # sample a subset of clients per communication round
        self.cohort_size = max(1, round(self.num_train_clients * args.part_rate))

        # init clients
        self.train_clients = {cid: BaseClient(cid, datasets, args) for cid, datasets in train_datasets.items()}
        self.test_clients = {cid: BaseClient(cid, datasets, args) for cid, datasets in test_datasets.items()}

        # model
        self.model = create_model(args)

    def run(self, args):
        # No Training, Direct Evaluation
        self.eval(args, 'valid')
        self.eval(args, 'test')

    def eval(self, args, mode='test'):
        """
        Collect logits / labels on each client's test set,
        compute Loss, bACC, Kappa, F1, AUC, Final.
        mode: 'valid' uses train_clients, 'test' uses test_clients
        """
        device = args.device
        model = self.model
        model.eval()

        loss_func = create_loss(args.loss)

        if mode == 'valid':
            clients = self.train_clients
            desc = "Eval(valid)"
        else:
            clients = self.test_clients
            desc = "Eval(test)"

        all_logits = []
        all_targets = []
        total_loss = 0.0
        total_examples = 0

        for cid, client in tqdm(clients.items(), desc=f"{desc} clients"):
            dataloader = client.dataloaders['test']

            for *X, Y in dataloader:
                # Consistent with FedAvg: drop_last on incomplete final batch to avoid single-sample batches
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
            tqdm.write(f"\t {desc}: no test data.")
            return

        # Concatenate logits / labels from all clients
        all_logits = torch.cat(all_logits, dim=0).to(device)
        all_targets = torch.cat(all_targets, dim=0).to(device)

        # Compute metrics: bACC, Kappa, F1, AUC, Final
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
            f"\t {desc}: Loss: {avg_loss:.4f}  "
            f"bACC: {bacc:.4f}  Kappa: {kappa:.4f}  "
            f"F1: {f1:.4f}  AUC: {auc:.4f}  Final: {final:.4f}"
        )

        # Record to history for later plotting / pkl storage
        log_dict = {
            f'{mode}_loss':  avg_loss,
            f'{mode}_bacc':  bacc,
            f'{mode}_kappa': kappa,
            f'{mode}_f1':    f1,
            f'{mode}_auc':   auc,
            f'{mode}_final': final,
        }
        self.history.append(log_dict)