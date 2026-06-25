import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from copy import deepcopy

from model import create_model, create_loss, create_metric, create_optimizer

from .Base import BaseServer, BaseClient
from .TTABase import TTABaseServer


class T3AServer(TTABaseServer):

    def __init__(self, train_datasets, test_datasets, args):
        TTABaseServer.__init__(self, train_datasets, test_datasets, args)

        self.train_clients = {cid: T3AClient(cid, datasets, args) for cid, datasets in train_datasets.items()}
        self.test_clients = {cid: T3AClient(cid, datasets, args) for cid, datasets in test_datasets.items()}

        # load a pre-trained model
        self.model = create_model(args)

        if getattr(args, "load_model_path", None) is not None and args.load_model_path != "none":
            try:
                state_dict = torch.load(args.load_model_path, map_location="cpu")
                if isinstance(state_dict, dict) and "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                self.model.load_state_dict(state_dict, strict=False)
                print(f"[T3AServer] Loaded pretrained model from {args.load_model_path}")
            except Exception as e:
                print(f"[T3AServer] Warning: failed to load pretrained model from {args.load_model_path}")
                print(f"Error: {e}")
        else:
            print("[T3AServer] No pretrained model path provided, model will be randomly initialized!")


        self.model.eval()


class T3AClient(BaseClient):

    # ====== helper: multi-head classifier forward ======
    def classifier_forward(self, z):
        cls = self.classifier
        if isinstance(cls, nn.ModuleList):
            outs = []
            for m in cls:
                outs.append(m(z))  # each head: (N, 1)
            return torch.cat(outs, dim=1)  # (N, num_classes)
        else:
            # Single-head case (vanilla T3A)
            return cls(z)

    def select_supports(self):
        ent_s = self.ent
        y_hat = self.labels.argmax(dim=1).long()
        filter_K = self.filter_K
        if filter_K == -1:
            indices = torch.LongTensor(list(range(len(ent_s))))
        else:
            indices = []
            indices1 = torch.LongTensor(list(range(len(ent_s))))
            for i in range(self.num_classes):
                _, indices2 = torch.sort(ent_s[y_hat == i])
                indices.append(indices1[y_hat == i][indices2][:filter_K])
            indices = torch.cat(indices)

        self.supports = self.supports[indices]
        self.labels = self.labels[indices]
        self.ent = self.ent[indices]

        return self.supports, self.labels

    # Note: *X, not a single x
    def forward(self, *X, adapt=False):
        # Like SHOT, feed featurizer with *X
        z = self.featurizer(*X)  # (N, feat_dim)

        if adapt:
            # Use our classifier_forward, not direct self.classifier(...)
            p = self.classifier_forward(z)  # (N, num_classes)
            yhat = torch.nn.functional.one_hot(
                p.argmax(1), num_classes=self.num_classes
            ).float()
            ent = softmax_entropy(p)

            self.supports = self.supports.to(z.device)
            self.labels = self.labels.to(z.device)
            self.ent = self.ent.to(z.device)
            self.supports = torch.cat([self.supports, z])
            self.labels = torch.cat([self.labels, yhat])
            self.ent = torch.cat([self.ent, ent])

        supports, labels = self.select_supports()
        supports = torch.nn.functional.normalize(supports, dim=1)
        weights = supports.T @ labels  # (feat_dim, num_classes)
        return z @ torch.nn.functional.normalize(weights, dim=0)  # (N, num_classes)

    def local_eval(self, model, args, dataset='test'):
        self.featurizer = model.get_featurizer()
        self.classifier = model.get_classifier()
        self.num_classes = args.num_labels

        cls = self.classifier
        # print("[DEBUG] classifier =", self.classifier)

        # ========= Build warmup_supports & warmup_prob =========
        if isinstance(cls, nn.ModuleList):
            weight_list = []
            head_modules = []

            for m in cls:
                # Recursively find first Linear in each head (512->256)
                lin = None
                for sub in m.modules():
                    if isinstance(sub, nn.Linear):
                        lin = sub
                        break
                if lin is None:
                    continue

                w = lin.weight  # (out_dim, feat_dim), e.g. (256, 512)
                if w.ndim == 2 and w.size(0) == 1:
                    w = w[0]      # -> (feat_dim,)
                else:
                    w = w.mean(dim=0)  # average weights across outputs -> (feat_dim,)
                weight_list.append(w)
                head_modules.append(m)

            if len(weight_list) == 0:
                # Fallback: random prototype initialization
                sample_X, _ = next(iter(self.dataloaders['test']))
                sample_X = [x.to(self.device) for x in sample_X]
                with torch.no_grad():
                    feat = self.featurizer(*sample_X)
                feat_dim = feat.size(1)

                warmup_supports = torch.randn(
                    self.num_classes, feat_dim, device=self.device
                )
                self.warmup_supports = warmup_supports

                prototypes = warmup_supports
                logit_list = []
                for m in cls:
                    logit_list.append(m(prototypes))  # (K,1) or (K,out_dim)
                warmup_prob = torch.cat(logit_list, dim=1)  # (K, K')
            else:
                warmup_supports = torch.stack(weight_list, dim=0)  # (K, feat_dim)
                self.warmup_supports = warmup_supports

                prototypes = warmup_supports.to(self.device)
                logit_list = []
                for m in head_modules:
                    logit_list.append(m(prototypes))  # (K,1)
                warmup_prob = torch.cat(logit_list, dim=1)  # (K, K_head)

        else:
            # Original T3A case: single Linear(feat_dim, num_classes)
            warmup_supports = cls.weight.data          # (K, feat_dim)
            self.warmup_supports = warmup_supports
            warmup_prob = cls(self.warmup_supports)    # (K, K)

        # ========= Subsequent logic: same as before =========
        self.warmup_ent = softmax_entropy(warmup_prob)
        self.warmup_labels = torch.nn.functional.one_hot(
            warmup_prob.argmax(1), num_classes=self.num_classes
        ).float()

        self.supports = self.warmup_supports.data
        self.labels = self.warmup_labels.data
        self.ent = self.warmup_ent.data

        self.filter_K = args.t3p_filter_k
        self.softmax = torch.nn.Softmax(-1)

        spv_loss_func = create_loss(args.loss)
        metric_func = create_metric(args.metric)

        total_examples, total_loss, total_metric = 0, 0, 0
        all_logits = []
        all_targets = []

        for *X, Y in self.dataloaders[dataset]:
            X = [x.to(self.device) for x in X]
            Y = Y.to(self.device)

            with torch.no_grad():
                logits = self.forward(*X, adapt=True)  # note: forward(*X,...)

                spv_loss = spv_loss_func(logits, Y)
                metric = metric_func(logits, Y)
                num_examples = len(X[0])

                total_examples += num_examples

                loss_val = float(spv_loss.item() if isinstance(spv_loss, torch.Tensor) else spv_loss)
                total_loss += loss_val * num_examples

                m_val = float(metric.item() if isinstance(metric, torch.Tensor) else metric)
                total_metric += m_val * num_examples

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

        print(f"[T3AClient cid={self.cid}] "
              f"bACC: {bacc:.4f}  Kappa: {kappa:.4f}  "
              f"F1: {f1:.4f}  AUC: {auc:.4f}  Final: {final:.4f}")

        return avg_loss, avg_metric, total_examples





@torch.jit.script
def softmax_entropy(x):
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)
