import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from copy import deepcopy

from model import create_model, create_loss, create_metric, create_optimizer

from .Base import BaseServer, BaseClient
from .TTABase import TTABaseServer

# from utils import pickle_load

class BatchNormServer(TTABaseServer):

    def __init__(self, train_datasets, test_datasets, args):
        TTABaseServer.__init__(self, train_datasets, test_datasets, args)

        self.train_clients = {cid: BatchNormClient(cid, datasets, args) for cid, datasets in train_datasets.items()}
        self.test_clients = {cid: BatchNormClient(cid, datasets, args) for cid, datasets in test_datasets.items()}

        # 1) Load model
        self.model = create_model(args)

        # 2) Load pretrained FedAvg weights -- use torch.load instead of pickle_load
        if getattr(args, "load_model_path", None) is not None and args.load_model_path != "none":
            try:
                state_dict = torch.load(args.load_model_path, map_location="cpu")

                # If the checkpoint wraps an extra dict layer, handle it:
                if isinstance(state_dict, dict) and "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]

                self.model.load_state_dict(state_dict, strict=False)
                print(f"[BatchNormServer] Loaded pretrained model from {args.load_model_path}")
            except Exception as e:
                print(f"[BatchNormServer] Warning: failed to load pretrained model from {args.load_model_path}")
                print(f"Error: {e}")
        else:
            print("[BatchNormServer] No pretrained model path provided, model will be randomly initialized!")

        # 3) Apply BN prior
        prior = args.prior_strength / (args.prior_strength + args.batch_size)
        print(f"[BatchNormServer] Using BN prior={prior:.4f}")


        self.model.change_bn(mode='prior', prior=prior)
        self.model.eval()

    def _eval_clients_all_metrics(self, model, args, clients, desc="Eval"):
        """
        Directly mirrors FedAvgServer._eval_clients_all_metrics:
        one pass over clients to get Loss, bACC, Kappa, F1, AUC, Final
        Evaluate on a group of clients: Loss, bACC, Kappa, F1, AUC, Final
        """
        loss_func = create_loss(args.loss)
        device = args.device
        model.eval()

        all_logits = []
        all_targets = []
        total_loss = 0.0
        total_examples = 0

        # Collect test data from all clients
        for cid, client in tqdm(clients.items(), desc=f"{desc} clients"):
            dataloader = client.dataloaders["test"]

            for *X, Y in dataloader:
                # Consistent drop_last logic with FedAvg
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
            return None

        all_logits = torch.cat(all_logits, dim=0).to(device)
        all_targets = torch.cat(all_targets, dim=0).to(device)

        # Multi-metric evaluation
        metric_bacc  = create_metric("bacc")
        metric_kappa = create_metric("kappa")
        metric_f1    = create_metric("f1")
        metric_auc   = create_metric("auc")
        metric_final = create_metric("final")

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

        log_dict = {
            f"{desc}_loss":  avg_loss,
            f"{desc}_bacc":  bacc,
            f"{desc}_kappa": kappa,
            f"{desc}_f1":    f1,
            f"{desc}_auc":   auc,
            f"{desc}_final": final,
        }
        self.history.append(log_dict)

        return log_dict

    def run(self, args):
        """
        Overrides TTABaseServer.run, performs a single eval:
        - test_type == off_site: evaluate on target clients (test_clients)
        - test_type == on_site: evaluate on source clients (train_clients)
        """
        test_type = getattr(args, "test_type", "off_site")
        if test_type == "on_site":
            desc = "Eval(on_site)"
            clients = self.train_clients
        else:
            desc = "Eval(off_site)"
            clients = self.test_clients

        tqdm.write(f"[BatchNormServer] Running BN-adapt evaluation on {test_type} domain")
        self._eval_clients_all_metrics(self.model, args, clients, desc=desc)


class BatchNormClient(BaseClient):

    def local_eval(self, model, args, dataset='test'):

        spv_loss_func = create_loss(args.loss)
        metric_func = create_metric(args.metric)

        model.eval()

        total_examples, total_loss, total_metric = 0, 0, 0

        for *X, Y in self.dataloaders[dataset]:
            # Get a batch of data
            X = [x.to(self.device) for x in X]
            Y = Y.to(self.device)

            with torch.no_grad():
                logits = model(*X)
                spv_loss = spv_loss_func(logits, Y)

                # record the loss and accuracy
                num_examples = len(X[0])
                total_examples += num_examples
                # total_loss += spv_loss.item() * num_examples
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                if isinstance(spv_loss, torch.Tensor):
                    loss_val = spv_loss.item()
                else:
                    loss_val = float(spv_loss)
                total_loss += loss_val * num_examples
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                metric = metric_func(logits, Y)
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                metric = metric_func(logits, Y)
                if isinstance(metric, torch.Tensor):
                    m_val = metric.item()
                else:
                    m_val = float(metric)
                total_metric += m_val * num_examples
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                # total_metric += metric.item() * num_examples

        avg_loss, avg_metric = total_loss / total_examples, total_metric / total_examples

        return avg_loss, avg_metric, total_examples









