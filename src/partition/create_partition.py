# src/partition/create_partition.py
import numpy as np
from torch.utils.data import ConcatDataset

from .step_partition import step_partition
from .patient_partition import patient_step_partition, patient_stratified_split
from .stat import print_quantity_stat, print_label_distribution_stat


def _get_train_and_onsite(datasets_obj):
    """
    Extract train_ds / onsite_ds from the structure passed by dataset.create_dataset(...).
    Convention: datasets = [train_ds, onsite_ds]; if already Concat'd, take datasets.datasets[0/1].
    """
    if isinstance(datasets_obj, (list, tuple)):
        train_ds = datasets_obj[0]
        onsite_ds = datasets_obj[1] if len(datasets_obj) > 1 else None
    elif isinstance(datasets_obj, ConcatDataset):
        train_ds = datasets_obj.datasets[0]
        onsite_ds = datasets_obj.datasets[1] if len(datasets_obj.datasets) > 1 else None
    else:
        train_ds = datasets_obj
        onsite_ds = None
    return train_ds, onsite_ds


def create_partition(datasets, args):
    """
      - Only do label-shift (step_partition) on Training Set, split into num_clients training clients
      - Each training client further split into 'train'/'test' by data_holdout
      - On-site Set as a whole is 1 test client (cid = num_clients)
      - Only print class distribution for training clients (Training Set)
    """
    num_clients = int(args.num_clients)
    num_labels  = int(args.num_labels)
    data_holdout = float(args.data_holdout)

    # 1) Get train/onsite blocks with lengths and offsets
    train_ds, onsite_ds = _get_train_and_onsite(datasets)
    len_train  = len(train_ds)
    len_onsite = len(onsite_ds) if onsite_ds is not None else 0
    offset_onsite = len_train  # in Concat([train, onsite]), onsite's global index start

    # 2) Apply label-shift partitioning only to Training Set (multi-label: utils.get_labels uses argmax proxy)
    partition_str = args.partition
    use_patient_level = partition_str.startswith('patient_')
    if use_patient_level:
        # Strip 'patient_' prefix, then parse remaining as step partition params
        inner = partition_str[len('patient_'):]
        # Support both "patient_step_2_16" and "patient_2_16" formats
        if inner.startswith('step_'):
            inner = inner[len('step_'):]
        alg = 'step'  # patient-level always uses step partition
        params = inner.split('_')
    else:
        alg, *params = partition_str.split('_')

    if alg == 'step':
        num_major = int(params[0])
        alpha     = float(params[1])
    elif alg == 'stratified':
        num_major = 2
        alpha     = 1.0
    else:
        raise NotImplementedError('Unknown data partition algorithm.')

    # Local index partition of the training set
    if use_patient_level:
        print(f"[Partition] Using PATIENT-LEVEL step partition (num_major={num_major}, alpha={alpha})")
        partition_idxs_train = patient_step_partition(train_ds, num_labels, num_clients, num_major, alpha)
    else:
        partition_idxs_train = step_partition(train_ds, num_labels, num_clients, num_major, alpha)

    # 3) Training clients: use local indices (0..len_train-1) directly as Concat global indices (training set first, no offset)
    #    and split into 'train'/'test' by data_holdout
    #    When using patient-level partition, split PATIENTS (not images) to prevent leakage
    train_client_sample_id = {}
    for cid in range(num_clients):
        idxs = np.asarray(partition_idxs_train[cid], dtype=int)
        if use_patient_level:
            # Patient-level stratified split: patient groups stay together
            part_dict = {'train': 1.0 - data_holdout, 'test': data_holdout}
            split_result = patient_stratified_split(train_ds, idxs, num_labels, part_dict)
            train_client_sample_id[cid] = split_result
        else:
            np.random.shuffle(idxs)
            pivot = round((1.0 - data_holdout) * len(idxs))
            train_client_sample_id[cid] = {
                'train': idxs[:pivot],
                'test' : idxs[pivot:],
            }

    # 4) Test client: On-site as a whole is one client (cid = num_clients), global indices need offset
    test_client_sample_id = {}
    partition_idxs = {cid: np.asarray(partition_idxs_train[cid], dtype=int) for cid in range(num_clients)}
    if len_onsite > 0:
        test_cid = num_clients
        onsite_global = np.arange(len_onsite, dtype=int) + offset_onsite
        test_client_sample_id[test_cid] = {'test': onsite_global}
        partition_idxs[test_cid] = onsite_global

    # 5) Only print training set (train/test distributions can match the 80/20 split)
    print("Label Distribution (TRAIN clients only):")
    print_label_distribution_stat(
        dataset=train_ds,
        num_labels=num_labels,
        partition_idxs={cid: v['train'] for cid, v in train_client_sample_id.items()},
        visualize=getattr(args, 'visualize', False),
        resize=0.2
    )
    print_quantity_stat({cid: v['train'] for cid, v in train_client_sample_id.items()})

    return train_client_sample_id, test_client_sample_id, partition_idxs


# Compatibility: keep original partition(...) (normally not called by create_partition,
# but may be called directly from elsewhere)
def partition(dataset, num_labels, num_clients, partition_config):
    alg, *params = partition_config.split('_')
    if alg == 'step':
        num_major = int(params[0])
        alpha     = float(params[1])
    elif alg == 'stratified':
        num_major = 2
        alpha     = 1.0
    else:
        raise NotImplementedError('Unknown data partition algorithm.')
    return step_partition(dataset, num_labels, num_clients, num_major, alpha)
