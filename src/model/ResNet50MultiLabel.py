import torch
import torch.nn as nn
from torchvision.models import resnet50
from collections import OrderedDict

from .Model import Model
from .MyBatchNorm2d import MyBatchNorm2d, ModifiedBatchNorm2d


class ResNet50MultiLabel(Model):
    """
    ResNet50 MultiLabel for OIA-ODIR (multi-head).
    """

    def __init__(self, shape_out=8, hidden_dim=256, dropout=0.2, use_hidden=True):
        super(ResNet50MultiLabel, self).__init__()

        self.num_classes = int(shape_out)

        # ========= backbone: ResNet50 =========
        self.backbone = resnet50(pretrained=True)
        # Remove original fc, outputs 2048-dim features directly
        self.backbone.fc = nn.Identity()

        in_dim = 2048  # ResNet50 feature dimension before fc

        # ========= multi-head classifier, same structure as ResNet18MultiLabel =========
        heads = []
        for _ in range(self.num_classes):
            if use_hidden:
                heads.append(nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 1)
                ))
            else:
                heads.append(nn.Linear(in_dim, 1))
        self.classifiers = nn.ModuleList(heads)

        self.drop_last = True

    def forward(self, x):
        if x.shape[1] == 1:  # convert 1-channel image to 3 channel
            x = x.repeat(1, 3, 1, 1)
        feats = self.backbone(x)         # [B, 2048]
        outs = [head(feats) for head in self.classifiers]  # list of [B,1]
        return torch.cat(outs, dim=1)    # [B, C]

    # ==================== BN modification, based on legacy ResNet50.change_bn ====================
    def change_bn(self, mode='grad', prior=0):
        model = self.backbone

        if mode == 'grad':
            model.bn1 = MyBatchNorm2d(model.bn1)

            model.layer1[0].bn1 = MyBatchNorm2d(model.layer1[0].bn1)
            model.layer1[0].bn2 = MyBatchNorm2d(model.layer1[0].bn2)
            model.layer1[0].bn3 = MyBatchNorm2d(model.layer1[0].bn3)
            model.layer1[0].downsample[1] = MyBatchNorm2d(model.layer1[0].downsample[1])
            model.layer1[1].bn1 = MyBatchNorm2d(model.layer1[1].bn1)
            model.layer1[1].bn2 = MyBatchNorm2d(model.layer1[1].bn2)
            model.layer1[1].bn3 = MyBatchNorm2d(model.layer1[1].bn3)
            model.layer1[2].bn1 = MyBatchNorm2d(model.layer1[2].bn1)
            model.layer1[2].bn2 = MyBatchNorm2d(model.layer1[2].bn2)
            model.layer1[2].bn3 = MyBatchNorm2d(model.layer1[2].bn3)

            model.layer2[0].bn1 = MyBatchNorm2d(model.layer2[0].bn1)
            model.layer2[0].bn2 = MyBatchNorm2d(model.layer2[0].bn2)
            model.layer2[0].bn3 = MyBatchNorm2d(model.layer2[0].bn3)
            model.layer2[0].downsample[1] = MyBatchNorm2d(model.layer2[0].downsample[1])
            model.layer2[1].bn1 = MyBatchNorm2d(model.layer2[1].bn1)
            model.layer2[1].bn2 = MyBatchNorm2d(model.layer2[1].bn2)
            model.layer2[1].bn3 = MyBatchNorm2d(model.layer2[1].bn3)
            model.layer2[2].bn1 = MyBatchNorm2d(model.layer2[2].bn1)
            model.layer2[2].bn2 = MyBatchNorm2d(model.layer2[2].bn2)
            model.layer2[2].bn3 = MyBatchNorm2d(model.layer2[2].bn3)
            model.layer2[3].bn1 = MyBatchNorm2d(model.layer2[3].bn1)
            model.layer2[3].bn2 = MyBatchNorm2d(model.layer2[3].bn2)
            model.layer2[3].bn3 = MyBatchNorm2d(model.layer2[3].bn3)

            model.layer3[0].bn1 = MyBatchNorm2d(model.layer3[0].bn1)
            model.layer3[0].bn2 = MyBatchNorm2d(model.layer3[0].bn2)
            model.layer3[0].bn3 = MyBatchNorm2d(model.layer3[0].bn3)
            model.layer3[0].downsample[1] = MyBatchNorm2d(model.layer3[0].downsample[1])
            model.layer3[1].bn1 = MyBatchNorm2d(model.layer3[1].bn1)
            model.layer3[1].bn2 = MyBatchNorm2d(model.layer3[1].bn2)
            model.layer3[1].bn3 = MyBatchNorm2d(model.layer3[1].bn3)
            model.layer3[2].bn1 = MyBatchNorm2d(model.layer3[2].bn1)
            model.layer3[2].bn2 = MyBatchNorm2d(model.layer3[2].bn2)
            model.layer3[2].bn3 = MyBatchNorm2d(model.layer3[2].bn3)
            model.layer3[3].bn1 = MyBatchNorm2d(model.layer3[3].bn1)
            model.layer3[3].bn2 = MyBatchNorm2d(model.layer3[3].bn2)
            model.layer3[3].bn3 = MyBatchNorm2d(model.layer3[3].bn3)
            model.layer3[4].bn1 = MyBatchNorm2d(model.layer3[4].bn1)
            model.layer3[4].bn2 = MyBatchNorm2d(model.layer3[4].bn2)
            model.layer3[4].bn3 = MyBatchNorm2d(model.layer3[4].bn3)
            model.layer3[5].bn1 = MyBatchNorm2d(model.layer3[5].bn1)
            model.layer3[5].bn2 = MyBatchNorm2d(model.layer3[5].bn2)
            model.layer3[5].bn3 = MyBatchNorm2d(model.layer3[5].bn3)

            model.layer4[0].bn1 = MyBatchNorm2d(model.layer4[0].bn1)
            model.layer4[0].bn2 = MyBatchNorm2d(model.layer4[0].bn2)
            model.layer4[0].bn3 = MyBatchNorm2d(model.layer4[0].bn3)
            model.layer4[0].downsample[1] = MyBatchNorm2d(model.layer4[0].downsample[1])
            model.layer4[1].bn1 = MyBatchNorm2d(model.layer4[1].bn1)
            model.layer4[1].bn2 = MyBatchNorm2d(model.layer4[1].bn2)
            model.layer4[1].bn3 = MyBatchNorm2d(model.layer4[1].bn3)
            model.layer4[2].bn1 = MyBatchNorm2d(model.layer4[2].bn1)
            model.layer4[2].bn2 = MyBatchNorm2d(model.layer4[2].bn2)
            model.layer4[2].bn3 = MyBatchNorm2d(model.layer4[2].bn3)

        elif mode == 'prior':
            # prior mode not tested yet (conservatively copying full ResNet50.py version)
            # Currently following the single-head implementation
            model.bn1 = ModifiedBatchNorm2d(model.bn1, prior=prior)

            model.layer1[0].bn1 = ModifiedBatchNorm2d(model.layer1[0].bn1, prior=prior)
            model.layer1[0].bn2 = ModifiedBatchNorm2d(model.layer1[0].bn2, prior=prior)
            model.layer1[0].bn3 = ModifiedBatchNorm2d(model.layer1[0].bn3, prior=prior)
            model.layer1[0].downsample[1] = ModifiedBatchNorm2d(model.layer1[0].downsample[1], prior=prior)
            model.layer1[1].bn1 = ModifiedBatchNorm2d(model.layer1[1].bn1, prior=prior)
            model.layer1[1].bn2 = ModifiedBatchNorm2d(model.layer1[1].bn2, prior=prior)
            model.layer1[1].bn3 = ModifiedBatchNorm2d(model.layer1[1].bn3, prior=prior)
            model.layer1[2].bn1 = ModifiedBatchNorm2d(model.layer1[2].bn1, prior=prior)
            model.layer1[2].bn2 = ModifiedBatchNorm2d(model.layer1[2].bn2, prior=prior)
            model.layer1[2].bn3 = ModifiedBatchNorm2d(model.layer1[2].bn3, prior=prior)

            model.layer2[0].bn1 = ModifiedBatchNorm2d(model.layer2[0].bn1, prior=prior)
            model.layer2[0].bn2 = ModifiedBatchNorm2d(model.layer2[0].bn2, prior=prior)
            model.layer2[0].bn3 = ModifiedBatchNorm2d(model.layer2[0].bn3, prior=prior)
            model.layer2[0].downsample[1] = ModifiedBatchNorm2d(model.layer2[0].downsample[1], prior=prior)
            model.layer2[1].bn1 = ModifiedBatchNorm2d(model.layer2[1].bn1, prior=prior)
            model.layer2[1].bn2 = ModifiedBatchNorm2d(model.layer2[1].bn2, prior=prior)
            model.layer2[1].bn3 = ModifiedBatchNorm2d(model.layer2[1].bn3, prior=prior)
            model.layer2[2].bn1 = ModifiedBatchNorm2d(model.layer2[2].bn1, prior=prior)
            model.layer2[2].bn2 = ModifiedBatchNorm2d(model.layer2[2].bn2, prior=prior)
            model.layer2[2].bn3 = ModifiedBatchNorm2d(model.layer2[2].bn3, prior=prior)
            model.layer2[3].bn1 = ModifiedBatchNorm2d(model.layer2[3].bn1, prior=prior)
            model.layer2[3].bn2 = ModifiedBatchNorm2d(model.layer2[3].bn2, prior=prior)
            model.layer2[3].bn3 = ModifiedBatchNorm2d(model.layer2[3].bn3, prior=prior)

            model.layer3[0].bn1 = ModifiedBatchNorm2d(model.layer3[0].bn1, prior=prior)
            model.layer3[0].bn2 = ModifiedBatchNorm2d(model.layer3[0].bn2, prior=prior)
            model.layer3[0].bn3 = ModifiedBatchNorm2d(model.layer3[0].bn3, prior=prior)
            model.layer3[0].downsample[1] = ModifiedBatchNorm2d(model.layer3[0].downsample[1], prior=prior)
            model.layer3[1].bn1 = ModifiedBatchNorm2d(model.layer3[1].bn1, prior=prior)
            model.layer3[1].bn2 = ModifiedBatchNorm2d(model.layer3[1].bn2, prior=prior)
            model.layer3[1].bn3 = ModifiedBatchNorm2d(model.layer3[1].bn3, prior=prior)
            model.layer3[2].bn1 = ModifiedBatchNorm2d(model.layer3[2].bn1, prior=prior)
            model.layer3[2].bn2 = ModifiedBatchNorm2d(model.layer3[2].bn2, prior=prior)
            model.layer3[2].bn3 = ModifiedBatchNorm2d(model.layer3[2].bn3, prior=prior)
            model.layer3[3].bn1 = ModifiedBatchNorm2d(model.layer3[3].bn1, prior=prior)
            model.layer3[3].bn2 = ModifiedBatchNorm2d(model.layer3[3].bn2, prior=prior)
            model.layer3[3].bn3 = ModifiedBatchNorm2d(model.layer3[3].bn3, prior=prior)
            model.layer3[4].bn1 = ModifiedBatchNorm2d(model.layer3[4].bn1, prior=prior)
            model.layer3[4].bn2 = ModifiedBatchNorm2d(model.layer3[4].bn2, prior=prior)
            model.layer3[4].bn3 = ModifiedBatchNorm2d(model.layer3[4].bn3, prior=prior)
            model.layer3[5].bn1 = ModifiedBatchNorm2d(model.layer3[5].bn1, prior=prior)
            model.layer3[5].bn2 = ModifiedBatchNorm2d(model.layer3[5].bn2, prior=prior)
            model.layer3[5].bn3 = ModifiedBatchNorm2d(model.layer3[5].bn3, prior=prior)

            model.layer4[0].bn1 = ModifiedBatchNorm2d(model.layer4[0].bn1, prior=prior)
            model.layer4[0].bn2 = ModifiedBatchNorm2d(model.layer4[0].bn2, prior=prior)
            model.layer4[0].bn3 = ModifiedBatchNorm2d(model.layer4[0].bn3, prior=prior)
            model.layer4[0].downsample[1] = ModifiedBatchNorm2d(model.layer4[0].downsample[1], prior=prior)
            model.layer4[1].bn1 = ModifiedBatchNorm2d(model.layer4[1].bn1, prior=prior)
            model.layer4[1].bn2 = ModifiedBatchNorm2d(model.layer4[1].bn2, prior=prior)
            model.layer4[1].bn3 = ModifiedBatchNorm2d(model.layer4[1].bn3, prior=prior)
            model.layer4[2].bn1 = ModifiedBatchNorm2d(model.layer4[2].bn1, prior=prior)
            model.layer4[2].bn2 = ModifiedBatchNorm2d(model.layer4[2].bn2, prior=prior)
            model.layer4[2].bn3 = ModifiedBatchNorm2d(model.layer4[2].bn3, prior=prior)

    def set_running_stat_grads(self):
        for m in self.backbone.modules():
            if isinstance(m, MyBatchNorm2d):
                m.set_running_stat_grads()

    def clip_bn_running_vars(self):
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

    # ==================== Methods below ported directly from ResNet18MultiLabel ====================
    def set_layers_to_adapt(self, mode='all'):
        print(mode)
        self.backbone.train()
        if mode in ['bn_all', 'bn_stat', 'bn_params']:
            self.backbone.requires_grad_(False)
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
            self.backbone.requires_grad_(False)
            # Multi-head: separate gradient for each head
            for head in self.classifiers:
                for p in head.parameters():
                    p.requires_grad_(True)
            for m in self.backbone.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.track_running_stats = False

        elif mode == 'last_bias':
            self.backbone.requires_grad_(False)
            for m in self.backbone.modules():
                if isinstance(m, nn.Linear):
                    m.bias.requires_grad_(True)
                elif isinstance(m, nn.BatchNorm2d):
                    m.track_running_stats = False

        elif mode == 'first_conv':
            self.backbone.requires_grad_(False)
            for m in self.backbone.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.track_running_stats = False
            self.backbone.conv1.requires_grad_(True)

        elif mode == 'block1':
            self.backbone.requires_grad_(False)
            self.backbone.layer1.requires_grad_(True)
            for m in self.backbone.layer1.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.track_running_stats = True

        elif mode == 'block2':
            self.backbone.requires_grad_(False)
            self.backbone.layer2.requires_grad_(True)
            for m in self.backbone.layer2.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.track_running_stats = True

        elif mode == 'block3':
            self.backbone.requires_grad_(False)
            self.backbone.layer3.requires_grad_(True)
            for m in self.backbone.layer3.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.track_running_stats = True

        elif mode == 'block4':
            self.backbone.requires_grad_(False)
            self.backbone.layer4.requires_grad_(True)
            for m in self.backbone.layer4.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.track_running_stats = True

        elif mode == 'all':
            pass
        else:
            raise NotImplementedError

    def surgical(self, mode='all'):

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
            for head in self.classifiers:
                for p in head.parameters():
                    p.requires_grad_(True)
        else:
            raise NotImplementedError

    def get_classifier(self):
        return self.classifiers

    def get_featurizer(self):
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
            ('avgpool', resnet.avgpool),
            ('flatten', nn.Flatten()),
        ]))
        return model


def test():
    model = ResNet50MultiLabel(shape_out=8)
    model.change_bn()
    total_num = sum(p.numel() for _, p in model.state_dict().items())
    print("Params:", total_num)

    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    print("Output shape:", y.shape)  # expected shape: [2, 8]
