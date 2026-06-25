"""
Patient-level partition: ensures left/right eyes of the SAME patient
are always grouped together in the same client and same train/test split.

Usage: called from create_partition.py when partition starts with 'patient_'
  e.g. --partition patient_2_16  → num_major=2, alpha=16, patient-level
"""
import numpy as np
import os
import re
from collections import defaultdict
from .utils import get_labels


def _extract_patient_id(filename: str) -> str:
    """
    Extract patient ID from ODIR filename.
    Format: {patient_id}_{left|right}.jpg  e.g. "1005_left.jpg" -> "1005"
    """
    # Remove extension and split by underscore
    base = os.path.splitext(filename)[0]
    parts = base.split('_')
    # The patient ID is everything before the _left/_right suffix
    if parts[-1] in ('left', 'right'):
        return '_'.join(parts[:-1])
    # Fallback: entire basename
    return base


def _get_patient_primary_label(patient_images, num_labels):
    """
    Determine primary label for a patient with possibly multiple images.
    Strategy: take the UNION of all images' multi-hot vectors,
    then argmax to get dominant class.
    If still tie, use the label with most positive images.

    patient_images: list of (filename, multi_hot_vector) for one patient
    Returns: int primary_label
    """
    if len(patient_images) == 1:
        _, y = patient_images[0]
        return int(np.argmax(y))

    # Sum all multi-hot vectors
    summed = np.zeros(num_labels)
    for _, y in patient_images:
        summed += y

    # Primary label = class with maximum sum
    return int(np.argmax(summed))


def patient_step_partition(dataset, num_labels, num_clients, num_major, alpha):
    """
    Step partition with patient-level grouping.

    1. Group all images by patient ID
    2. Assign primary label to each patient
    3. Run step partition on patients (not images)
    4. Expand back: all images of patient P go to the assigned client

    Args:
        dataset: ODIRMultiLabelFlat dataset (must have .samples with filenames)
        num_labels: number of classes
        num_clients: number of training clients
        num_major: number of major classes per client (step partition param)
        alpha: concentration parameter (higher = more IID)

    Returns:
        partition_idxs: dict {cid: np.array of image indices}
    """
    # Step 1: Get labels and group by patient
    labels, idxs_by_label, num_samples_per_label = get_labels(dataset, num_labels)

    # Build patient groups from dataset samples
    patient_groups = defaultdict(list)  # {patient_id: [(global_idx, multi_hot_vec), ...]}

    # Access underlying dataset samples
    base = dataset
    subset_indices = None
    if hasattr(dataset, 'dataset'):  # Subset
        base = dataset.dataset
        subset_indices = np.asarray(dataset.indices)

    for local_idx in range(len(dataset)):
        global_idx = subset_indices[local_idx] if subset_indices is not None else local_idx
        fname, y = base.samples[global_idx]
        pid = _extract_patient_id(fname)
        patient_groups[pid].append((global_idx, y.numpy()))

    print(f"[PatientPartition] {len(patient_groups)} unique patients from {len(dataset)} images")

    # Step 2: For each patient, determine primary label
    patient_list = []  # [(patient_id, primary_label, [global_indices])]
    for pid, images in patient_groups.items():
        primary_label = _get_patient_primary_label(images, num_labels)
        indices = [idx for idx, _ in images]
        patient_list.append((pid, primary_label, indices))

    # Step 3: Build per-label patient index lists
    patients_by_label = defaultdict(list)
    for i, (pid, plabel, indices) in enumerate(patient_list):
        patients_by_label[plabel].append(i)

    num_patients_per_label = np.array([len(patients_by_label[k]) for k in range(num_labels)], dtype=int)

    # Step 4: Run step partition on patients (same algorithm)
    prior = num_patients_per_label / max(num_patients_per_label.sum(), 1)

    if alpha == float('inf'):
        matrix = np.ones((num_clients, num_labels))
    else:
        matrix = np.ones((num_clients, num_labels))

    # ODIR: 8 labels, num_major=2
    if num_labels == 8:
        for cid in range(num_clients):
            for j in range(num_major):
                matrix[cid, (cid + j) % num_labels] += (alpha - 1)

    # Normalize
    matrix = matrix / matrix.sum(axis=0)

    # Cumulative matrix
    cumulate = matrix.cumsum(axis=0) * num_patients_per_label
    cumulate = (cumulate + 0.5).astype(int)
    cumulate = np.vstack([np.zeros((1, num_labels), dtype=int), cumulate])

    # Step 5: Assign patients to clients
    partition_patient_idxs = {}  # {cid: [patient_indices]}

    for cid in range(num_clients):
        patient_idxs = []
        for label in range(num_labels):
            start = cumulate[cid, label]
            end = cumulate[cid + 1, label]
            # Get patient indices within this label
            plist = patients_by_label[label]
            patient_idxs.extend(plist[start:end])
        partition_patient_idxs[cid] = patient_idxs

    # Step 6: Expand patients back to images
    partition_idxs = {}
    for cid in range(num_clients):
        image_idxs = []
        for pidx in partition_patient_idxs[cid]:
            _, _, indices = patient_list[pidx]
            image_idxs.extend(indices)
        partition_idxs[cid] = np.array(image_idxs, dtype=int)

    # Print stats
    print(f"[PatientPartition] Image distribution across clients:")
    for cid in range(num_clients):
        n_patients = len(partition_patient_idxs[cid])
        n_images = len(partition_idxs[cid])
        # Count primary labels
        label_counts = np.zeros(num_labels, dtype=int)
        for pidx in partition_patient_idxs[cid]:
            _, plabel, _ = patient_list[pidx]
            label_counts[plabel] += 1
        label_str = ", ".join([f"L{k}:{label_counts[k]}" for k in range(num_labels) if label_counts[k] > 0])
        print(f"  Client {cid}: {n_images} images from {n_patients} patients | {label_str}")

    return partition_idxs


def patient_stratified_split(dataset, idxs, num_labels, part_rate_dict):
    """
    Stratified split WITH patient-level grouping.
    Ensures all images from the same patient stay in the same split.
    """
    from torch.utils.data import Subset

    if idxs is None:
        subdataset = dataset
        idxs_array = np.arange(len(dataset))
    else:
        subdataset = Subset(dataset, idxs)
        idxs_array = np.asarray(idxs)

    # Get patient groups within these indices
    base = dataset
    if hasattr(dataset, 'dataset'):
        base = dataset.dataset

    patient_groups = defaultdict(list)  # {pid: [local_positions]}
    for local_pos, global_idx in enumerate(idxs_array):
        fname, y = base.samples[global_idx]
        pid = _extract_patient_id(fname)
        patient_groups[pid].append(local_pos)

    # Determine primary label for each patient
    patient_entries = []
    for pid, local_positions in patient_groups.items():
        # Get primary label from all images
        summed = np.zeros(num_labels)
        for lp in local_positions:
            fname, y = base.samples[idxs_array[lp]]
            summed += y.numpy()
        primary_label = int(np.argmax(summed))
        patient_entries.append((pid, primary_label, local_positions))

    # Now do stratified split on patients
    patients_by_label = defaultdict(list)
    for i, (pid, plabel, positions) in enumerate(patient_entries):
        patients_by_label[plabel].append(i)

    num_patients_per_label = np.array([len(patients_by_label[k]) for k in range(num_labels)], dtype=int)

    num_parts = len(part_rate_dict)
    part_names = list(part_rate_dict.keys())
    part_vector = np.array([part_rate_dict[p] for p in part_names])
    matrix = np.tile(part_vector, (num_labels, 1)).transpose(1, 0)

    cumulate = matrix.cumsum(axis=0) * num_patients_per_label
    cumulate = (cumulate + 0.49).astype(int)
    cumulate = np.vstack([np.zeros((1, num_labels), dtype=int), cumulate])

    partition_idxs = {}
    for pid_idx in range(num_parts):
        patient_subset = []
        for label in range(num_labels):
            start = cumulate[pid_idx, label]
            end = cumulate[pid_idx + 1, label]
            plist = patients_by_label[label]
            patient_subset.extend(plist[start:end])

        # Expand patients back to local positions
        local_positions = []
        for pidx in patient_subset:
            _, _, positions = patient_entries[pidx]
            local_positions.extend(positions)

        partition_idxs[part_names[pid_idx]] = idxs_array[np.array(local_positions, dtype=int)]

    return partition_idxs
