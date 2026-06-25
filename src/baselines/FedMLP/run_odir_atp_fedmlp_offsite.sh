#!/bin/bash
# FedMLP ATP-aligned: step_2_16 + gaussian_noise + pretrained ResNet18
# 3 seeds, 300 rounds, GPU 3
set -e
source "${HOME}/anaconda3/etc/profile.d/conda.sh"
conda activate atp
cd "$(dirname "$0")"

OUTDIR="outputs_odir_atp"
mkdir -p "$OUTDIR"
GPU=3

ATP_DATA_DIR="./data/atp/odir_multi"
ATP_PART_DIR="~/data/atp/partition/odir_multi"

for SEED in 0 1 2; do
    LOGFILE="${OUTDIR}/fedmlp_atp_seed${SEED}.log"
    echo "===== [FedMLP ATP] Seed $SEED start: $(date) ====="
    python main.py \
        --exp FedMLP --dataset ODIR_ATP \
        --model Resnet18 \
        --n_clients 5 --n_classes 8 --annotation_num 8 \
        --batch_size 32 --base_lr 3e-5 --local_ep 1 \
        --rounds_warmup 300 --rounds_FedMLP_stage1 50 \
        --iid 0 --seed "$SEED" --runs 1 --deterministic 1 --train 1 \
        --gpu $GPU --test_set offsite \
        --atp_data_dir "$ATP_DATA_DIR" \
        --atp_partition_dir "$ATP_PART_DIR" \
        > "$LOGFILE" 2>&1
    echo "===== [FedMLP ATP] Seed $SEED done: $(date) ====="
done

echo "[FedMLP ATP] All seeds done at $(date)"
for SEED in 0 1 2; do
    echo "--- Seed $SEED ---"
    grep -E '\[Off-site\] -----> mAP|-----> mAP' "${OUTDIR}/fedmlp_atp_seed${SEED}.log" | tail -1
done
