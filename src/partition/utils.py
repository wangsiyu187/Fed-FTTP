import numpy as np
from torch.utils.data import Subset
import torch

# def get_labels(dataset, num_labels):
#     """
#     Get label-related information from dataset
#     :param dataset: torch.utils.data.Dataset
#     :return:
#         labels: numpy.array the labels of data in dataset
#         idxs_by_class: dictionary {label:idxs}, where idxs is a (shuffled) list of data idxs with given label
#         num_labels: int, how many labels
#         num_samples_per_label: numpy.array, how many samples of each label
#     """
#     labels = [Y for *X, Y in dataset]
#     labels = np.array(labels)

#     idxs_by_class = {}
#     num_samples_per_label = np.zeros(num_labels, dtype=int)

#     for label in range(num_labels):
#         idxs = np.where(labels == label)[0]  # np.where returns a tuple with length 1
#         np.random.shuffle(idxs)
#         idxs_by_class[label] = idxs
#         num_samples_per_label[label] = len(idxs)

#     return labels, idxs_by_class, num_samples_per_label
#===============================================================================
def _as_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)
def get_labels(dataset, num_labels):
    """
    Fast label retrieval, multi-label compatible:
    - If dataset.targets_proxy exists: use directly (recommended to generate in dataset class)
    - Otherwise if dataset.targets exists:
        * 1D -> treat as single-label
        * 2D -> treat as multi-label, take argmax as "single class"
    - Otherwise fallback to slow path (per-sample __getitem__), avoid if possible
    Returned idx are local indices within the "sub-dataset" (0..len(subdataset)-1)
    """
    # Support Subset: extract underlying dataset and original global idxs
    base = dataset
    subset_indices = None
    if isinstance(dataset, Subset):
        base = dataset.dataset
        subset_indices = np.asarray(dataset.indices)

    # 1) Prefer targets_proxy
    if hasattr(base, "targets_proxy"):
        labels_all = _as_numpy(getattr(base, "targets_proxy")).astype(np.int64)

    # 2) Next targets (1D single-label / 2D multi-label compatible)
    elif hasattr(base, "targets"):
        t = getattr(base, "targets")
        t_np = _as_numpy(t)
        if t_np.ndim == 2:         # multi-label case: one multi-hot per row
            labels_all = t_np.argmax(axis=1).astype(np.int64)
        else:                       # single-label
            labels_all = t_np.astype(np.int64)
    # 3) Then labels (used by some datasets)
    elif hasattr(base, "labels"):
        t = getattr(base, "labels")
        t_np = _as_numpy(t)
        if t_np.ndim == 2:
            labels_all = t_np.argmax(axis=1).astype(np.int64)
        else:
            labels_all = t_np.astype(np.int64)

    # 4) Last resort: slow path (triggers __getitem__, involves IO)
    else:
        labels_list = []
        # NOTE: if this is a Subset, dataset is the subset, not the base
        src = dataset if subset_indices is None else Subset(base, subset_indices)
        for _, y in src:
            y_np = _as_numpy(y)
            if y_np.ndim == 0:
                labels_list.append(int(y_np.item()))
            elif y_np.ndim == 1:
                labels_list.append(int(y_np.argmax()))
            else:
                labels_list.append(int(np.argmax(y_np)))
        labels = np.asarray(labels_list, dtype=np.int64)
        # Build idxs_by_class and return
        idxs_by_class = {k: np.where(labels == k)[0] for k in range(num_labels)}
        for k in idxs_by_class:
            np.random.shuffle(idxs_by_class[k])
        num_samples_per_label = np.array([len(idxs_by_class[k]) for k in range(num_labels)], dtype=int)
        return labels, idxs_by_class, num_samples_per_label

    # If Subset, slice to subset indices; otherwise use full set
    labels = labels_all if subset_indices is None else labels_all[subset_indices]

    # Assemble per-class local indices (relative to current subdataset)
    idxs_by_class = {}
    num_samples_per_label = np.zeros(num_labels, dtype=int)
    local_indices = np.arange(len(labels))

    for k in range(num_labels):
        idxs = local_indices[labels == k]
        np.random.shuffle(idxs)
        idxs_by_class[k] = idxs
        num_samples_per_label[k] = idxs.size

    return labels, idxs_by_class, num_samples_per_label
#===============================================================================

def stratified_split(dataset, idxs, num_labels, part_rate_dict):
    """
    Stratified Split
    Split a Data subset (dataset, idxs) to multiple parts,
    each with a given rate, while controlling the label distribution
    """

    if idxs is None:
        subdataset = dataset
        idxs = np.arange(len(dataset))
    else:
        subdataset = Subset(dataset, idxs)

    labels, idxs_by_label, num_samples_per_label = get_labels(subdataset, num_labels)
    idx2part = {idx:part for idx, part in enumerate(part_rate_dict)}
    part2idx = {part:idx for idx, part in idx2part.items()}
    num_parts = len(part_rate_dict)

    part_vector = np.zeros(num_parts)
    for part, rate in part_rate_dict.items():
        part_vector[part2idx[part]] = rate
    matrix = np.tile(part_vector, (num_labels, 1)).transpose(1, 0)

    # cumulative matrix
    cumulate = matrix.cumsum(axis=0) * num_samples_per_label
    cumulate = (cumulate + 0.49).astype(int)  # round to integer
    cumulate = np.vstack([np.zeros((1, num_labels), dtype=int), cumulate])

    partition_idxs = dict()

    for pid in range(num_parts):
        subidxs = []
        for label in range(num_labels):
            subidxs.append(idxs_by_label[label][cumulate[pid, label]:cumulate[pid + 1, label]])

        partition_idxs[idx2part[pid]] = idxs[np.concatenate(subidxs)]

    return partition_idxs
