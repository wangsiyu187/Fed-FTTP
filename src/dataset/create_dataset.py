from torch.utils.data import ConcatDataset, Subset, TensorDataset

from utils import pickle_load
import os
import torchvision.transforms as transforms

from torchvision import datasets as tv_datasets
from .odir_multilabel import ODIRMultiLabelFlat

def create_dataset(dataset_name, data_img, test_type, label_txt=None, input_size=256):
    """
    Create the ODIR dataset.
    :param args:
    :return:
    """
    if test_type == "on_site":
        test_subdir = "On-site Test Set"
    elif test_type == "off_site":
        test_subdir = "Off-site Test Set"
    else:
        raise ValueError(f"Unknown test_type: {test_type}. Expected 'on_site' or 'off_site'.")

    # Load ODIR dataset (legacy format)
    if dataset_name == "odir":
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor()
        ])
        train_dataset = tv_datasets.ImageFolder(os.path.join(data_img, "Training Set"), transform=transform)
        test_dataset = tv_datasets.ImageFolder(os.path.join(data_img, test_subdir), transform=transform)
        return [train_dataset, test_dataset]

    # ODIR multi-label format: flat directory + full label txt
    if dataset_name == "odir_multi":
        if not label_txt:
            raise ValueError("dataset=odir_multi requires a label_txt path")
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor()
        ])
        train_dir = os.path.join(data_img, "Training Set")
        test_dir  = os.path.join(data_img, test_subdir)
        train_dataset = ODIRMultiLabelFlat(train_dir, label_txt, transform=transform)
        test_dataset  = ODIRMultiLabelFlat(test_dir, label_txt, transform=transform)
        return [train_dataset, test_dataset]

    raise NotImplementedError(f'Unknown dataset: {dataset_name}. Supported: odir, odir_multi')


def load_processed_dataset(path):
    print('Load Processed Data:', path)
    obj = pickle_load(path)
    X = obj['X']
    Y = obj['Y']
    dataset = TensorDataset(X, Y)
    return dataset


def create_fed_dataset(args, config=None):
    dataset_name = args.dataset
    data_img = args.data_img
    partition_path = args.partition_path

    if args.corruption == "none":
        label_txt = getattr(args, "label_txt", None)
        datasets = create_dataset(dataset_name, data_img, args.test_type, label_txt=label_txt)
        dataset = ConcatDataset(datasets)
    else:
        dataset = load_processed_dataset(args.corruption_path)

    print(len(dataset))
    train_client_sample_id, test_client_sample_id = pickle_load(partition_path)

    train_datasets = {
        cid: {part: Subset(dataset, indices) for part, indices in sids.items()} for cid, sids in
        train_client_sample_id.items()
    }

    test_datasets = {
        cid: {part: Subset(dataset, indices) for part, indices in sids.items()} for cid, sids in
        test_client_sample_id.items()
    }

    return train_datasets, test_datasets
