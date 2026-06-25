from utils import pickle_load

from .CNN import ShallowCNN
from .ResNet18 import ResNet18
from .ResNet50 import ResNet50
import torch

from .ResNet18MultiLabel import ResNet18MultiLabel
from .ResNet50MultiLabel import ResNet50MultiLabel
from .MLGCN import ResNet18MLGCN
def create_model(args):
    """
    Create model
    :param args:
    :return:
    """
    shape_in = args.shape_in
    shape_out = args.shape_out

    if args.model == 'resnet18':
        model = ResNet18(shape_out=shape_out)
#===========
    elif args.model == 'resnet18_multi':
        model = ResNet18MultiLabel(shape_out=shape_out, hidden_dim=256, dropout=0.2, use_hidden=True)

    elif args.model == 'resnet50_multi':
        model = ResNet50MultiLabel(shape_out=shape_out, hidden_dim=256, dropout=0.2, use_hidden=True)

    elif args.model == 'resnet18_mlgcn':  
        t = getattr(args, 'mlgcn_t', 0.4)
        adj_file = getattr(args, 'mlgcn_adj_file', 'none')
        in_ch = getattr(args, 'mlgcn_in_channel', 300)
        model = ResNet18MLGCN(shape_out=shape_out,in_channel=in_ch,t=t,adj_file=adj_file if adj_file != 'none' else None,)

#===========
    elif args.model == 'resnet50':
        model = ResNet50(shape_out=shape_out)
    elif args.model == 'cnn':
        model = ShallowCNN(shape_in=shape_in, shape_out=shape_out)
    else:
        raise NotImplementedError('Unknown model. ')

    model.to(args.device)

    if args.load_model_path != 'none':
        # state = pickle_load(args.load_model_path)
        # model.load_state_dict(state)
        # Load state_dict
        state = torch.load(args.load_model_path, map_location=args.device)
        model.load_state_dict(state, strict=False)

    return model
