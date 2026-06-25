import warnings
warnings.filterwarnings("ignore")
import os
import torch
import numpy as np
import random
from copy import deepcopy

from dataset import create_fed_dataset
from algorithm import create_system
from utils import pickle_load, pickle_save

from options import args_parser

from dataset.load_dataset import load_odir, ODIRDataFrame, get_transforms, federated_dataset_split

import yaml

# Load parameters
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
def _ensure_parent_dir(path: str):
    """Create parent directory if it does not exist; ignore if path is 'none' or empty."""
    if not path or str(path).lower() == 'none':
        return
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def main(args):
    args_backup = deepcopy(args)

    # get dataset
    client_datasets_dict, test_datasets = create_fed_dataset(args)

    # get system
    server = create_system(client_datasets_dict, test_datasets, args)

    # run experiments
    server.run(args)

    # Save training history
    if args.history_path != 'none':
        _ensure_parent_dir(args.history_path)
        content = {'args': args_backup, 'history': server.history.data}
        # Use 'ab' append mode to stay consistent with original behavior
        pickle_save(content, args.history_path, mode='ab')
        print(f"[main] Saved history to {args.history_path}")

    # Save model weights (state_dict)
    if args.save_model_path != 'none':
        _ensure_parent_dir(args.save_model_path)
        torch.save(server.model.state_dict(), args.save_model_path)
        print(f"[main] Saved model state_dict to {args.save_model_path}")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # the following three lines seem not necessary
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    args = args_parser()
    setup_seed(args.seed)
    main(args)
