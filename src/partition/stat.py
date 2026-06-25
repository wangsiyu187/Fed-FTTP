import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Subset

#====================================================
def _as_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _extract_proxy_labels(dataset):
    """Extract 1D class labels from a dataset (supports Subset, targets_proxy, one-hot->argmax)."""
    base = dataset
    subset_indices = None
    if isinstance(dataset, Subset):
        base = dataset.dataset
        subset_indices = np.asarray(dataset.indices)

    if hasattr(base, "targets_proxy"):
        lab_all = _as_numpy(getattr(base, "targets_proxy")).astype(np.int64)
    elif hasattr(base, "targets"):
        t_np = _as_numpy(getattr(base, "targets"))
        lab_all = (t_np.argmax(axis=1) if t_np.ndim == 2 else t_np).astype(np.int64)
    elif hasattr(base, "labels"):
        t_np = _as_numpy(getattr(base, "labels"))
        lab_all = (t_np.argmax(axis=1) if t_np.ndim == 2 else t_np).astype(np.int64)
    else:
        # Slow path: iterate samples
        labs = []
        src = dataset if subset_indices is None else Subset(base, subset_indices)
        for _, y in src:
            y_np = _as_numpy(y)
            if y_np.ndim == 0:
                labs.append(int(y_np.item()))
            else:
                labs.append(int(y_np.argmax()))
        return np.asarray(labs, dtype=np.int64)

    # Slice to subset indices
    return lab_all if subset_indices is None else lab_all[subset_indices]
#================================================================
def print_label_distribution_stat(dataset, num_labels, partition_idxs, visualize=False, resize=0.2):
    """
    Compute per-client class distribution (for "multi-class" labels only; no image IO triggered).
    dataset: raw Dataset or Subset
    partition_idxs: dict[cid] -> list of all indices for that client
    dataset: raw Dataset or Subset
    partition_idxs: dict[cid] -> global index array for this client
    """
    # First wrap dataset as Subset for easy local indexing
    if isinstance(dataset, Subset):
        global_to_local = {g: i for i, g in enumerate(dataset.indices)}
    else:
        global_to_local = None

    labels_local = _extract_proxy_labels(dataset)  # shape [len(dataset)]

    cids = sorted(list(partition_idxs.keys()))
    cid_to_row = {cid: r for r, cid in enumerate(cids)}
    num_clients = len(cids)

    label_dist = np.zeros((num_clients, num_labels), dtype=int)

    for cid, gidxs in partition_idxs.items():
        row = cid_to_row[cid]
        if global_to_local is None:
            loc = np.asarray(gidxs, dtype=int)
        else:
            loc = [global_to_local[g] for g in gidxs if g in global_to_local]
            if not loc:
                continue
            loc = np.asarray(loc, dtype=int)

        cls = labels_local[loc]
        mask = (cls >= 0) & (cls < num_labels)
        if mask.any():
            binc = np.bincount(cls[mask], minlength=num_labels)
            label_dist[row, :num_labels] += binc.astype(int)

    # Print summary statistics
    total_per_client = label_dist.sum(axis=1)
    total_all = int(total_per_client.sum())
    print("Label Distribution:")
    for r, cid in enumerate(cids):
        if total_per_client[r] == 0:
            print(f"client {cid}: EMPTY")
        else:
            print(f"client {cid}: {label_dist[r].tolist()} (sum={int(total_per_client[r])})")
    if num_clients:
        print("Quantity: ")
        mu, sd = float(total_per_client.mean()), float(total_per_client.std())
        q = np.quantile(total_per_client, [0.0, 0.25, 0.5, 0.75, 1.0])
        print(f"\tQuantiles: {int(q[0])} {q[1]:.1f} {q[2]:.1f} {q[3]:.1f} {int(q[4])}")
        print(f"\tMean +- Std: {mu:.1f} +- {sd:.1f}")

    # Visualization (requires matplotlib, supports integer cid)
    if visualize:
        import matplotlib.pyplot as plt
        # Bubble chart: x = row index (not cid), y = label, size = count
        x = np.tile(np.arange(num_clients), num_labels)
        y = np.repeat(np.arange(num_labels), num_clients)
        size = label_dist.T.flatten()  # [num_labels*num_clients]
        plt.scatter(x, y, s=(size * resize))
        plt.xticks(range(num_clients), [str(c) for c in cids], rotation=0)
        plt.yticks(range(num_labels))
        plt.xlabel('Clients (by row order)')
        plt.ylabel('Labels')
        plt.title('Label Distribution (proxy)')
        plt.tight_layout()
        plt.show()


def print_quantity_stat(partition_idxs, visualize=False):
    cids = sorted(list(partition_idxs.keys()))
    quantities = [len(partition_idxs[cid]) for cid in cids]
    lo = np.min(quantities)
    lo4 = np.quantile(quantities, 1 / 4)
    md = np.median(quantities)
    hi4 = np.quantile(quantities, 3 / 4)
    hi = np.max(quantities)
    mu = np.mean(quantities)
    sd = np.std(quantities, ddof=1)
    print('Quantity: \n\tQuantiles:', lo, lo4, md, hi4, hi)
    print('\tMean +- Std: %f +- %f' % (mu, sd))

    if visualize:
        plt.hist(quantities)
        plt.title('Quantity Distribution')
        plt.show()
