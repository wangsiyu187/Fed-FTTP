import math
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18
from collections import OrderedDict

from .Model import Model
from .MyBatchNorm2d import MyBatchNorm2d, ModifiedBatchNorm2d


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, consistent with ML-GCN style.
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        """
        input: [C, in_features]
        adj:   [C, C]
        """
        support = torch.matmul(input, self.weight)      # [C, out_features]
        output = torch.matmul(adj, support)             # [C, out_features]
        if self.bias is not None:
            return output + self.bias
        else:
            return output


def gen_A(num_classes, t=0.0, adj_file=None):
    """
    Generate initial adjacency matrix A:
    - If adj_file=None, use identity matrix (self-connections only), a “degenerate” GCN
    - For real co-occurrence graphs, save adj.pkl / .npy in ML-GCN format and pass via adj_file
    """
    if adj_file is None or adj_file == "none":
        A = np.eye(num_classes, dtype=np.float32)
        return A

    # Supports both original ML-GCN adj.pkl and saved .npy format
    try:
        if adj_file.endswith(".npy"):
            A = np.load(adj_file).astype(np.float32)
            if A.shape[0] != num_classes or A.shape[1] != num_classes:
                raise ValueError(f"adj.shape={A.shape} does not match num_classes={num_classes}")
            return A

        import pickle
        result = pickle.load(open(adj_file, "rb"))
        _adj = result["adj"]
        _nums = result["nums"][:, np.newaxis]
        _adj = _adj / _nums
        _adj[_adj < t] = 0
        _adj[_adj >= t] = 1
        _adj = _adj * 0.25 / (_adj.sum(0, keepdims=True) + 1e-6)
        _adj = _adj + np.identity(num_classes, np.int32)
        return _adj.astype(np.float32)
    except Exception as e:
        print(f"[MLGCN] Warning: failed to load adj_file '{adj_file}': {e}. Fallback to identity adjacency.")
        A = np.eye(num_classes, dtype=np.float32)
        return A


def gen_adj(A):
    """
    Normalize adjacency matrix, consistent with ML-GCN: D^{-1/2} A D^{-1/2}
    A: [C, C]
    """
    if isinstance(A, np.ndarray):
        A = torch.from_numpy(A)
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj


class ResNet18MLGCN(Model):
    """
    ResNet18 + GCN multi-label model:
    - backbone: resnet18 (fc replaced with Identity, outputs 512-dim features)
    - head: label embedding + 2 GraphConv layers + feature * label_feat
    Output shape [B, num_classes], consistent with ResNet18MultiLabel for sharing RAL / metric / ATP code.
    """

    def __init__(self, shape_out=8, in_channel=300, t=0.4, adj_file=None):
        super(ResNet18MLGCN, self).__init__()

        self.num_classes = int(shape_out)

        # ======== ResNet18 backbone: reuses original ATP implementation ========
        self.backbone = resnet18(pretrained=True)
        # Remove final fc layer so backbone(x) outputs 512-dim features directly
        self.backbone.fc = nn.Identity()

        # ======== GCN component: adapted from ML-GCN to 512-dim features ========
        # label embedding dim:  -> 256 -> 512, to align with image feature dim
        self.gc1 = GraphConvolution(in_channel, 256)
        self.gc2 = GraphConvolution(256, 512)
        self.relu = nn.LeakyReLU(0.2)

        # Adjacency matrix A: no gradient on A, consistent with original ML-GCN
        _adj = gen_A(self.num_classes, t, adj_file)
        self.A = nn.Parameter(torch.from_numpy(_adj).float(), requires_grad=False)

        # Learnable label embeddings: no dependency on GloVe
        self.label_embed = nn.Parameter(
            torch.randn(self.num_classes, in_channel) * 0.1
        )

        # Consistent with ResNet18MultiLabel
        self.drop_last = True

    def forward(self, x):
        # Convert grayscale to 3 channels
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # ======== Extract image features: B x 512 ========
        # Note: using self.backbone(x) directly so after change_bn all MyBatchNorm2d
        # layers are actually used in forward, so snapshot_mean/snapshot_var get assigned correctly.
        feat = self.backbone(x)                # [B, 512]

        # ======== Get per-class feature vectors via GCN: C x 512 ========
        adj = gen_adj(self.A).detach()         # [C, C]
        inp = self.label_embed                 # [C, in_channel]

        g = self.gc1(inp, adj)                 # [C, 256]
        g = self.relu(g)
        g = self.gc2(g, adj)                   # [C, 512]
        g = g.transpose(0, 1)                  # [512, C]

        # ======== Image features x class features -> multi-label logits: B x C ========
        out = torch.matmul(feat, g)            # [B, C]
        return out

    # ------------------------------------------------------------------
    # The methods below provide BN manipulation APIs compatible with ResNet18MultiLabel for ATP/TTA
    # ------------------------------------------------------------------
    def change_bn(self, mode='grad', prior=0):
        """
        Replace backbone BatchNorm2d with MyBatchNorm2d / ModifiedBatchNorm2d,
        enabling gradient computation on running stats and clipping at test time.
        """
        model = self.backbone

        if mode == 'grad':

            model.bn1 = MyBatchNorm2d(model.bn1)

            model.layer1[0].bn1 = MyBatchNorm2d(model.layer1[0].bn1)
            model.layer1[0].bn2 = MyBatchNorm2d(model.layer1[0].bn2)
            model.layer1[1].bn1 = MyBatchNorm2d(model.layer1[1].bn1)
            model.layer1[1].bn2 = MyBatchNorm2d(model.layer1[1].bn2)

            model.layer2[0].bn1 = MyBatchNorm2d(model.layer2[0].bn1)
            model.layer2[0].bn2 = MyBatchNorm2d(model.layer2[0].bn2)
            model.layer2[0].downsample[1] = MyBatchNorm2d(model.layer2[0].downsample[1])
            model.layer2[1].bn1 = MyBatchNorm2d(model.layer2[1].bn1)
            model.layer2[1].bn2 = MyBatchNorm2d(model.layer2[1].bn2)

            model.layer3[0].bn1 = MyBatchNorm2d(model.layer3[0].bn1)
            model.layer3[0].bn2 = MyBatchNorm2d(model.layer3[0].bn2)
            model.layer3[0].downsample[1] = MyBatchNorm2d(model.layer3[0].downsample[1])
            model.layer3[1].bn1 = MyBatchNorm2d(model.layer3[1].bn1)
            model.layer3[1].bn2 = MyBatchNorm2d(model.layer3[1].bn2)

            model.layer4[0].bn1 = MyBatchNorm2d(model.layer4[0].bn1)
            model.layer4[0].bn2 = MyBatchNorm2d(model.layer4[0].bn2)
            model.layer4[0].downsample[1] = MyBatchNorm2d(model.layer4[0].downsample[1])
            model.layer4[1].bn1 = MyBatchNorm2d(model.layer4[1].bn1)
            model.layer4[1].bn2 = MyBatchNorm2d(model.layer4[1].bn2)

        elif mode == 'prior':

            model.bn1 = ModifiedBatchNorm2d(model.bn1, prior=prior)

            model.layer1[0].bn1 = ModifiedBatchNorm2d(model.layer1[0].bn1, prior=prior)
            model.layer1[0].bn2 = ModifiedBatchNorm2d(model.layer1[0].bn2, prior=prior)
            model.layer1[1].bn1 = ModifiedBatchNorm2d(model.layer1[1].bn1, prior=prior)
            model.layer1[1].bn2 = ModifiedBatchNorm2d(model.layer1[1].bn2, prior=prior)

            model.layer2[0].bn1 = ModifiedBatchNorm2d(model.layer2[0].bn1, prior=prior)
            model.layer2[0].bn2 = ModifiedBatchNorm2d(model.layer2[0].bn2, prior=prior)
            model.layer2[0].downsample[1] = ModifiedBatchNorm2d(model.layer2[0].downsample[1], prior=prior)
            model.layer2[1].bn1 = ModifiedBatchNorm2d(model.layer2[1].bn1, prior=prior)
            model.layer2[1].bn2 = ModifiedBatchNorm2d(model.layer2[1].bn2, prior=prior)

            model.layer3[0].bn1 = ModifiedBatchNorm2d(model.layer3[0].bn1, prior=prior)
            model.layer3[0].bn2 = ModifiedBatchNorm2d(model.layer3[0].bn2, prior=prior)
            model.layer3[0].downsample[1] = ModifiedBatchNorm2d(model.layer3[0].downsample[1], prior=prior)
            model.layer3[1].bn1 = ModifiedBatchNorm2d(model.layer3[1].bn1, prior=prior)
            model.layer3[1].bn2 = ModifiedBatchNorm2d(model.layer3[1].bn2, prior=prior)

            model.layer4[0].bn1 = ModifiedBatchNorm2d(model.layer4[0].bn1, prior=prior)
            model.layer4[0].bn2 = ModifiedBatchNorm2d(model.layer4[0].bn2, prior=prior)
            model.layer4[0].downsample[1] = ModifiedBatchNorm2d(model.layer4[0].downsample[1], prior=prior)
            model.layer4[1].bn1 = ModifiedBatchNorm2d(model.layer4[1].bn1, prior=prior)
            model.layer4[1].bn2 = ModifiedBatchNorm2d(model.layer4[1].bn2, prior=prior)

        else:
            raise NotImplementedError

    def set_running_stat_grads(self):
        """
        Called by ATP adapt_one_step: enables gradients on running_mean/var of all MyBatchNorm2d layers.
        """
        for m in self.backbone.modules():
            if isinstance(m, MyBatchNorm2d):
                # After forward pass, snapshot_* should already be populated;
                # skip this BN as a safety measure if snapshots are not ready yet.
                if getattr(m, "snapshot_mean", None) is None or getattr(m, "snapshot_var", None) is None:
                    continue
                m.set_running_stat_grads()

    def clip_bn_running_vars(self):
        """
        Called after each ATP update step to prevent BN running_var from becoming negative (NaN).
        """
        for m in self.backbone.modules():
            if isinstance(m, MyBatchNorm2d):
                m.clip_running_var()

    def freeze_bn_stats(self):
        """
        Do not update running stats of batch norm layers.
        """
        for m in self.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.track_running_stats = False
                m.eval()

    def set_layers_to_adapt(self, mode='all'):
        """
        Same interface as ResNet18MultiLabel for selecting learnable layers by mode in ATP.
        """
        print(mode)
        self.backbone.train()
        if mode in ['bn_all', 'bn_stat', 'bn_params']:
            # disable grad first
            self.backbone.requires_grad_(False)
            # configure norm for tent/BN updates
            for m in self.backbone.modules():
                if isinstance(m, nn.BatchNorm2d):
                    if mode in ['bn_all', 'bn_params']:
                        m.requires_grad_(True)
                    if mode in ['bn_all', 'bn_stat']:
                        m.track_running_stats = True
                        m.momentum = 1.0
                    else:
                        m.track_running_stats = False

        elif mode == 'tent':
            self.backbone.requires_grad_(False)
            for m in self.backbone.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.requires_grad_(True)
                    m.track_running_stats = False
                    m.running_mean = None
                    m.running_var = None

        elif mode == 'last_layer':
            # Freeze backbone, only tune GCN + label embedding
            self.backbone.requires_grad_(False)
            # Do not update BN running stats
            for m in self.backbone.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.track_running_stats = False
            # Enable gradients for GCN and label_embed
            for p in self.gc1.parameters():
                p.requires_grad_(True)
            for p in self.gc2.parameters():
                p.requires_grad_(True)
            self.label_embed.requires_grad_(True)

        elif mode == 'all':
            # Default: all layers trainable
            pass

        else:
            raise NotImplementedError

    def surgical(self, mode='all'):
        """
        Optional: mirrors ResNet18MultiLabel surgical mode for fine-tuning specific blocks or last layer.
        """
        # First freeze backbone + disable BN running stats
        self.backbone.requires_grad_(False)
        for m in self.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.track_running_stats = False

        if mode == 'block1':
            self.backbone.layer1.requires_grad_(True)
            for m in self.backbone.layer1.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.track_running_stats = True

        elif mode == 'block2':
            self.backbone.layer2.requires_grad_(True)
            for m in self.backbone.layer2.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.track_running_stats = True

        elif mode == 'block3':
            self.backbone.layer3.requires_grad_(True)
            for m in self.backbone.layer3.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.track_running_stats = True

        elif mode == 'block4':
            self.backbone.layer4.requires_grad_(True)
            for m in self.backbone.layer4.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.track_running_stats = True

        elif mode == 'last_layer':
            # surgical mode: only tune GCN + label_embed
            for p in self.gc1.parameters():
                p.requires_grad_(True)
            for p in self.gc2.parameters():
                p.requires_grad_(True)
            self.label_embed.requires_grad_(True)

        else:
            raise NotImplementedError

    def get_featurizer(self):
        """
        For extracting features in other algorithms, consistent with ResNet18MultiLabel.
        """
        resnet = self.backbone
        model = nn.Sequential(OrderedDict([
            ('conv1', resnet.conv1),
            ('bn1', resnet.bn1),
            ('relu', resnet.relu),
            ('maxpool', resnet.maxpool),
            ('layer1', resnet.layer1),
            ('layer2', resnet.layer2),
            ('layer3', resnet.layer3),
            ('layer4', resnet.layer4),
            ('avgpool', nn.AdaptiveAvgPool2d((1, 1))),
            ('flatten', nn.Flatten()),
        ]))
        return model

    def get_classifier(self):
        """
        For ML-GCN, the classifier is GCN + label_embed.
        Returns a simple Module wrapping these parameters for safe external .parameters() calls.
        """
        class _GCNClassifier(nn.Module):
            def __init__(self, outer):
                super().__init__()
                self.gc1 = outer.gc1
                self.gc2 = outer.gc2
                self.label_embed = outer.label_embed

            def forward(self, x):
                raise NotImplementedError

        return _GCNClassifier(self)
