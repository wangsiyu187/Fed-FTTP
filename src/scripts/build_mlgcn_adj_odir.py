import numpy as np

# The 8 label order must match your training setup!
# If your OIA-ODIR label order is N, D, G, C, A, H, M, O, write:
LABELS = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
num_classes = len(LABELS)

# Corresponds to label_txt in options.py
label_txt = "./data/oia_odir/train_labels.txt"

# === 1. Read labels and build N x C binary matrix ===
all_labels = []  # each image is a length-C 0/1 list

with open(label_txt, 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        # You need to modify this based on your actual format!!
        # Example: if each line is "img001.jpg 0 1 0 0 1 0 0 0"
        parts = line.split()
        # Image name first, followed by 0/1
        y = [int(x) for x in parts[1:1+num_classes]]
        if len(y) != num_classes:
            raise ValueError(f"Unexpected label length {len(y)}, expected {num_classes}")
        all_labels.append(y)

Y = np.array(all_labels).astype(np.float32)  # [N, C]

print("Loaded labels:", Y.shape)

# === 2. Compute co-occurrence matrix ===
# Co-occurrence: C_{i,j} = number of samples where both are 1
C = np.matmul(Y.T, Y)   # [C, C]
print("Raw co-occurrence matrix:\n", C)

# === 3. Normalize + add self-loops ===
# Simple approach: row-normalize to conditional probability P(j | i)
row_sum = C.sum(axis=1, keepdims=True) + 1e-6
A = C / row_sum

# Add self-loops (diagonal = 1), and re-normalize rows (optional)
np.fill_diagonal(A, 1.0)
row_sum = A.sum(axis=1, keepdims=True) + 1e-6
A = A / row_sum

print("Normalized adjacency:\n", A)

# === 4. Save as .npy for ML-GCN ===
save_path = "./data/oia_odir/mlgcn_adj_odir_multi.npy"
np.save(save_path, A)
print("Saved adjacency to:", save_path)
