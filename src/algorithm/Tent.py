import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from copy import deepcopy

from model import create_model, create_loss, create_metric, create_optimizer

from .Base import BaseServer, BaseClient
from .TTABase import TTABaseServer


class TentServer(TTABaseServer):

    def __init__(self, train_datasets, test_datasets, args):
        TTABaseServer.__init__(self, train_datasets, test_datasets, args)

        self.train_clients = {cid: TentClient(cid, datasets, args) for cid, datasets in train_datasets.items()}
        self.test_clients = {cid: TentClient(cid, datasets, args) for cid, datasets in test_datasets.items()}

        # load a pre-trained model (loading in main.py)
        self.model = create_model(args)

        if getattr(args, "load_model_path", None) is not None and args.load_model_path != "none":
            try:
                state_dict = torch.load(args.load_model_path, map_location="cpu")
                if isinstance(state_dict, dict) and "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                self.model.load_state_dict(state_dict, strict=False)
                print(f"[TentServer] Loaded pretrained model from {args.load_model_path}")
            except Exception as e:
                print(f"[TentServer] Warning: failed to load pretrained model from {args.load_model_path}")
                print(f"Error: {e}")
        else:
            print("[TentServer] No pretrained model path provided, model will be randomly initialized!")

        # train mode, because tent optimizes the model to minimize entropy
        self.model.train()
        # disable grad, to (re-)enable only what tent updates
        self.model.requires_grad_(False)
        # configure norm for tent updates: enable grad + force batch statisics
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None

                # print('one')


class TentClient(BaseClient):

    def local_eval(self, model, args, dataset='test'):
        unspv_loss_func = create_loss('ent')
        spv_loss_func = create_loss(args.loss)
        metric_func = create_metric(args.metric)   # If metric specified, use it to select best model
        optimizer = create_optimizer(model, optimizer_name=args.lm_opt, lr=args.lm_lr)

        # Only collect the final multi-metric here
        all_logits = []
        all_targets = []

        total_examples, total_loss, total_metric = 0, 0, 0

        for *X, Y in self.dataloaders[dataset]:
            X = [x.to(self.device) for x in X]
            Y = Y.to(self.device)

            logits = model(*X)
            loss = unspv_loss_func(logits, Y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            with torch.no_grad():
                spv_loss = spv_loss_func(logits, Y)
                metric = metric_func(logits, Y)
                num_examples = len(X[0])

                total_examples += num_examples

                if isinstance(spv_loss, torch.Tensor):
                    loss_val = spv_loss.item()
                else:
                    loss_val = float(spv_loss)
                total_loss += loss_val * num_examples

                if isinstance(metric, torch.Tensor):
                    m_val = metric.item()
                else:
                    m_val = float(metric)
                total_metric += m_val * num_examples

                # === Collect logits / targets ===
                all_logits.append(logits.detach().cpu())
                all_targets.append(Y.detach().cpu())

        avg_loss, avg_metric = total_loss / total_examples, total_metric / total_examples

        # === Compute 5 metrics on current client ===
        all_logits = torch.cat(all_logits, dim=0).to(self.device)
        all_targets = torch.cat(all_targets, dim=0).to(self.device)

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

        print(f"[TentClient cid={self.cid}] "
              f"bACC: {bacc:.4f}  Kappa: {kappa:.4f}  "
              f"F1: {f1:.4f}  AUC: {auc:.4f}  Final: {final:.4f}")

        # Note: values are returned; final aggregation is done by the caller (TTABaseServer)
        return avg_loss, avg_metric, total_examples