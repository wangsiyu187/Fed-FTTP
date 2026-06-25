"""
Partition a centralized dataset odir to clients in FL
"""
import numpy as np

from dataset import create_dataset, shapes_out  # , create_dataset_natural_shift
from partition import create_partition
from utils import pickle_save
from options import args_parser
from torch.utils.data import ConcatDataset


# from visual import visualize_label_distribution


from PIL import Image, ImageEnhance, ImageFilter
import os
import torch
from torchvision import datasets, transforms

def add_gaussian_noise(img, mean=0, std=0.1):
    np_img = np.array(img).astype(np.float32) / 255.0
    noise = np.random.normal(mean, std, np_img.shape)
    noisy = np.clip(np_img + noise, 0, 1)
    return Image.fromarray((noisy * 255).astype(np.uint8))

#======================================================
def _apply_corruption(img, corruption_type="gaussian_noise", severity=1):
    if corruption_type == "gaussian_noise":
        return add_gaussian_noise(img, std=0.1 * severity)
    elif corruption_type == "blur":
        return img.filter(ImageFilter.GaussianBlur(radius=severity))
    elif corruption_type == "brightness":
        return ImageEnhance.Brightness(img).enhance(1 + 0.2 * severity)
    elif corruption_type == "contrast":
        return ImageEnhance.Contrast(img).enhance(1 + 0.3 * severity)
    else:
        return img
#=======================================================
def make_odir_c(partition_idxs, is_train, datasets, mode="ood", corruption_type="gaussian_noise", severity=1, input_size=256):
    """
    Generate ODIR corruption dataset (modeled after make_cifar10_c)

    Args:
        partition_idxs: dict, {cid: idxs}
        is_train: dict, {cid: bool}  whether it is a training client
        data_dir: return of create_dataset(...): [train_ds, onsite_ds]
        mode: str, 'ood' or 'iid'
        corruption_type: str, corruption type
        severity: int, corruption severity
        input_size: int, input image size

    Returns:
        X: torch tensor [N, C, H, W]
        Y: torch tensor [N]
        corruption: str
    """
    # Check if data is valid
    assert isinstance(datasets, (list, tuple)) and len(datasets) >= 1
    train_ds = datasets[0]
    onsite_ds = datasets[1] if len(datasets) > 1 else None
    len_train = len(train_ds)

# Read underlying info
    def _get_item(global_idx):
        if global_idx < len_train:
            ds = train_ds
            local = global_idx
        else:
            assert onsite_ds is not None, "global_idx points to On-site but onsite_ds not provided"
            ds = onsite_ds
            local = global_idx - len_train
        fname, y = ds.samples[local]
        path = os.path.join(ds.root_dir, fname)  # root_dir saved in ODIRMultiLabelFlat.__init__
        return path, y
    
    tfm = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor()
    ])

    X_list, Y_list = [], []
    for cid, idxs in partition_idxs.items():
        idxs = np.asarray(idxs, dtype=int)
        for gidx in idxs:
            path, y = _get_item(gidx)
            img = Image.open(path).convert("RGB")
            # Only corrupt test clients (mode="ood"), or both train/test (mode="iid")
            need_corrupt = (mode == "iid") or (mode == "ood" and not is_train.get(cid, False))
            if need_corrupt:
                img = _apply_corruption(img, corruption_type=corruption_type, severity=severity)
            x = tfm(img)  # [3,H,W]
            X_list.append(x)
            Y_list.append(y.float())

    X = torch.stack(X_list, dim=0)
    Y = torch.stack(Y_list, dim=0)
    return X, Y, corruption_type
def main(args):
    
    data_dir = "./data/OIA-ODIR_dataset_multi/RGB_preprocessed"
    
    # get dataset
    data_img  = args.data_img       
    label_txt = args.label_txt
    datasets = create_dataset(args.dataset, data_img, args.test_type, label_txt=label_txt)
    
    # get partition of datasets
    *partitions, partition_idxs = create_partition(datasets, args)
    train_clients, test_clients = partitions 
    # check correctness
    num_before = sum([len(dataset) for dataset in datasets])

    all_sample = set()
#======================================================
    for parts_dict in partitions:  # iterate over train_clients / test_clients tables
        for _cid, part_map in parts_dict.items():
            for _part, idxs in part_map.items():
                all_sample |= set(map(int, idxs))
#======================================================
    num_after = len(all_sample)
    assert num_before == num_after

    # Save partition result
    pickle_save(obj=partitions, file=args.partition_path, mode='wb')

    # generate corrupted dataset
    if args.corruption == "ood" or args.corruption == "iid":

        is_train = {}

        train_clients, test_clients = partitions

        for cid in train_clients:
            is_train[cid] = True

        for cid in test_clients:
            is_train[cid] = False

        if args.dataset in ['odir', 'odir_multi']:
            # cifar_c, labels, corruption = make_odir_c(datasets, corruption_type=args.corruption)
            X_c, Y_c, corruption = make_odir_c(
                partition_idxs=partition_idxs,
                is_train=is_train,
                datasets=datasets,
                mode=args.corruption,
                corruption_type="gaussian_noise",   # can change to blur/contrast/brightness
                severity=1,
                input_size=256
            )
        else:
            raise NotImplementedError(f"Corruption not implemented for dataset: {args.dataset}")

        obj = {
            'X': X_c,
            'Y':  Y_c,
            'corruption': corruption,
        }

        pickle_save(obj=obj, file=args.corruption_path, mode='wb')



def set_seed(seed):
    np.random.seed(seed)


if __name__ == '__main__':
    args = args_parser()
    set_seed(args.partition_seed)
    main(args)
