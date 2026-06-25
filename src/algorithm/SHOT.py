import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from copy import deepcopy
import numpy as np

from model import create_model, create_loss, create_metric, create_optimizer

from .Base import BaseServer, BaseClient
from .TTABase import TTABaseServer


class SHOTServer(TTABaseServer):

    def __init__(self, train_datasets, test_datasets, args):
        TTABaseServer.__init__(self, train_datasets, test_datasets, args)

        self.train_clients = {cid: SHOTClient(cid, datasets, args) for cid, datasets in train_datasets.items()}
        self.test_clients = {cid: SHOTClient(cid, datasets, args) for cid, datasets in test_datasets.items()}

        # load a pre-trained model (loading in main.py)
        self.model = create_model(args)

        if getattr(args, "load_model_path", None) is not None and args.load_model_path != "none":
            try:
                state_dict = torch.load(args.load_model_path, map_location="cpu")
                if isinstance(state_dict, dict) and "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                self.model.load_state_dict(state_dict, strict=False)
                print(f"[SHOTServer] Loaded pretrained model from {args.load_model_path}")
            except Exception as e:
                print(f"[SHOTServer] Warning: failed to load pretrained model from {args.load_model_path}")
                print(f"Error: {e}")
        else:
            print("[SHOTServer] No pretrained model path provided, model will be randomly initialized!")


        if args.model == 'cnn':
            self.model.linear5.requires_grad_(False)
        else: # resnet
            self.model.backbone.fc.requires_grad_(False)  # freeze classifier


class SHOTClient(BaseClient):

    def classifier_forward(self, z):
        cls = self.classifier
        if isinstance(cls, nn.ModuleList):
            outs = []
            for m in cls:
                outs.append(m(z))  # each head: (N, 1)
            return torch.cat(outs, dim=1)  # (N, num_classes)
        else:
            return cls(z)
    def local_eval(self, model, args, dataset='test'):
        unspv_loss_func = create_loss('ent')
        spv_loss_func = create_loss(args.loss)
        metric_func = create_metric(args.metric)
        optimizer = create_optimizer(model, optimizer_name=args.lm_opt, lr=args.lm_lr)

        featurizer = model.get_featurizer()
        self.classifier = model.get_classifier()
        total_examples, total_loss, total_metric = 0, 0, 0

        all_logits = []
        all_targets = []

        for *X, Y in self.dataloaders[dataset]:
            X = [x.to(self.device) for x in X]
            Y = Y.to(self.device)

            feature = featurizer(*X)
            logits = self.classifier_forward(feature)
            # entropy
            loss_ent = unspv_loss_func(logits, Y)
            # diversity
            pred_dist = torch.softmax(logits, dim=1).mean(dim=0)
            loss_div = torch.sum(pred_dist * torch.log(pred_dist + 1e-9))

            if args.shot_beta == 0:
                loss = loss_ent + loss_div
            else:
                with torch.no_grad():
                    pred = torch.softmax(logits, dim=1)
                    pred = pred / pred.sum(dim=0)
                    prototype = torch.matmul(pred.transpose(1, 0), feature)
                    PL = torch.zeros_like(Y)
                    for i in range(len(Y)):
                        dist = (feature[i] - prototype).norm(dim=1)
                        PL[i] = torch.argmin(dist)

                loss_pl = spv_loss_func(logits, PL)
                loss = loss_ent + loss_div + args.shot_beta * loss_pl

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            with torch.no_grad():
                spv_loss = spv_loss_func(logits, Y)
                metric = metric_func(logits, Y)
                num_examples = len(X[0])

                total_examples += num_examples
                total_loss += spv_loss.item() * num_examples
                total_metric += metric.item() * num_examples

                all_logits.append(logits.detach().cpu())
                all_targets.append(Y.detach().cpu())

        avg_loss, avg_metric = total_loss / total_examples, total_metric / total_examples

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

        print(f"[SHOTClient cid={self.cid}] "
              f"bACC: {bacc:.4f}  Kappa: {kappa:.4f}  "
              f"F1: {f1:.4f}  AUC: {auc:.4f}  Final: {final:.4f}")

        return avg_loss, avg_metric, total_examples
