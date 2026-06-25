import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from copy import deepcopy
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np

from model import create_model, create_loss, create_metric, create_optimizer

from .Base import BaseServer, BaseClient
from .TTABase import TTABaseServer


class MEMOServer(TTABaseServer):

    def __init__(self, train_datasets, test_datasets, args):
        TTABaseServer.__init__(self, train_datasets, test_datasets, args)

        self.train_clients = {cid: MEMOClient(cid, datasets, args) for cid, datasets in train_datasets.items()}
        self.test_clients = {cid: MEMOClient(cid, datasets, args) for cid, datasets in test_datasets.items()}

        # load a pre-trained model (loading in main.py)
        self.model = create_model(args)

        if getattr(args, "load_model_path", None) is not None and args.load_model_path != "none":
            try:
                state_dict = torch.load(args.load_model_path, map_location="cpu")
                if isinstance(state_dict, dict) and "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                self.model.load_state_dict(state_dict, strict=False)
                print(f"[MEMOServer] Loaded pretrained model from {args.load_model_path}")
            except Exception as e:
                print(f"[MEMOServer] Warning: failed to load pretrained model from {args.load_model_path}")
                print(f"Error: {e}")
        else:
            print("[MEMOServer] No pretrained model path provided, model will be randomly initialized!")


        prior = args.prior_strength / (args.prior_strength + 1)

        # self.model.change_bn(mode='prior', prior=prior)

        self.model.eval()


class MEMOClient(BaseClient):

    def __init__(self, cid, datasets, args):
        BaseClient.__init__(self, cid, datasets, args)

        if args.dataset in ['pacs_aug', 'odir_multi', 'odir']:
            self.tr_pre = transforms.Compose([
                transforms.Normalize((0, 0, 0), (1 / 0.229, 1 / 0.224, 1 / 0.225)),
                transforms.Normalize((-0.485, -0.456, -0.406), (1, 1, 1)),
                transforms.ToPILImage()
            ])
            self.tr_post = transforms.Compose([
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
            self.aug = transforms.Compose([
                transforms.RandomCrop(224, padding=32, padding_mode="symmetric"),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            raise NotImplementedError(f'MEMO not supported for dataset={args.dataset}')
        self.batch_size = 1

        self.dataloaders = {}
        for key, dataset in self.datasets.items():
            if key in ['train', ]:
                # for training set, we shuffle the data, we drop too small batch in the training if necessary
                self.dataloaders[key] = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False,
                                                   num_workers=self.num_workers)

            elif key in ['valid', 'test', ]:
                # for testing set, it is not necessary to shuffle
                self.dataloaders[key] = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, drop_last=False,
                                                   num_workers=self.num_workers)


    def adapt_single(self, model, image, optimizer, args):
        model.eval()
        image = self.tr_pre(image)
        inputs = [self.tr_post(self.aug(image)) for _ in range(args.memo_aug_size)]
        inputs = torch.stack(inputs).to(self.device)

        # print(inputs.shape)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss, logits = marginal_entropy(outputs)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    def local_eval(self, model, args, dataset='test'):

        spv_loss_func = create_loss(args.loss)
        metric_func = create_metric(args.metric)
        optimizer = create_optimizer(model, optimizer_name=args.lm_opt, lr=args.lm_lr)

        total_examples, total_loss, total_metric = 0, 0, 0

        dataloader = self.dataloaders[dataset]

        state = deepcopy(model.state_dict())

        all_logits = []
        all_targets = []

        for *X, Y in dataloader:
            model.load_state_dict(state)

            image = X[0][0]
            Y = Y.to(self.device)

            self.adapt_single(model, image, optimizer, args)
            model.eval()

            X = [x.to(self.device) for x in X]

            with torch.no_grad():
                logits = model(*X)
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

        print(f"[MEMOClient cid={self.cid}] "
              f"bACC: {bacc:.4f}  Kappa: {kappa:.4f}  "
              f"F1: {f1:.4f}  AUC: {auc:.4f}  Final: {final:.4f}")

        return avg_loss, avg_metric, total_examples






def marginal_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1), avg_logits

