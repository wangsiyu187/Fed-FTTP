# Dataset Setup

## OIA-ODIR Dataset

The primary dataset is the OIA-ODIR multi-label ocular disease dataset. It contains fundus images with 8 disease labels: N (Normal), D (Diabetes), G (Glaucoma), C (Cataract), A (AMD), H (Hypertension), M (Myopia), O (Other).

### Directory Structure

```
data/
  OIA-ODIR_dataset_multi/
    RGB_preprocessed/
      Training Set/
      On-site Test Set/
      Off-site Test Set/
  oia_odir/
    train_labels.txt          # Multi-label annotations
    mlgcn_adj_odir_multi.npy  # Pre-computed ML-GCN adjacency matrix
```

### Obtaining the Data

1. Download the OIA-ODIR dataset from the official source
2. Preprocess images to RGB format
3. Place images in the directory structure shown above
4. Copy `src/config.yaml.template` to `src/config.yaml` and fill in your paths

## External Test Datasets

For cross-dataset evaluation, the following datasets are supported:

- **DDR**: Diabetic Retinopathy grading dataset
- **REFUGE**: Retinal Fundus Glaucoma Challenge dataset
- **PAPILA**: PapilaDB dataset
- **HRF**: High-Resolution Fundus dataset
- **Bajwa**: Bajwa eye diseases dataset

Place external datasets under `data/` and configure paths in evaluation scripts.

## ML-GCN Adjacency Matrix

Generate the co-occurrence adjacency matrix for ML-GCN:

```bash
python src/build_mlgcn_adj_odir.py
```

This reads `data/oia_odir/train_labels.txt` and outputs `data/oia_odir/mlgcn_adj_odir_multi.npy`.
