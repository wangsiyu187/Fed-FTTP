#!/bin/bash
# RANK ATP-aligned baseline on ODIR (same .mat data as DICNet)
# 3 seeds, 250 epochs, GPU 0
set -e
source "${HOME}/anaconda3/etc/profile.d/conda.sh"
conda activate atp

cd "$(dirname "$0")"
OUTDIR="./records_odir_atp"
mkdir -p "$OUTDIR"
DATA_DIR="../DICNet/data/ODIR_ATP"
GPU=0

echo "seed BACC Kappa F1 AUC" > "${OUTDIR}/rank_atp_results.txt"

for SEED in 0 1 2; do
    LOGFILE="${OUTDIR}/rank_atp_seed${SEED}.log"
    echo "===== [RANK ATP] Seed $SEED start: $(date) ====="
    python train_odir.py \
        --data_dir "$DATA_DIR" --gpu $GPU --epochs 250 \
        --seed $SEED --batch_size 128 \
        > "$LOGFILE" 2>&1
    echo "[RANK ATP] Seed $SEED done: $(date)"

    BEST_LINE=$(grep "Best (Off-site):" "$LOGFILE" | tail -1)
    BACC=$(echo "$BEST_LINE" | grep -oP 'BACC=\K[0-9.]+')
    KAPPA=$(echo "$BEST_LINE" | grep -oP 'Kappa=\K[0-9.]+')
    F1=$(echo "$BEST_LINE" | grep -oP 'F1=\K[0-9.]+')
    AUC=$(echo "$BEST_LINE" | grep -oP 'AUC=\K[0-9.]+')
    echo "$SEED $BACC $KAPPA $F1 $AUC" >> "${OUTDIR}/rank_atp_results.txt"
    echo "  BACC=$BACC, Kappa=$KAPPA, F1=$F1, AUC=$AUC"
done

echo ""
echo "===== RANK ATP Summary ====="
cat "${OUTDIR}/rank_atp_results.txt"
python -c "
import numpy as np
data = np.loadtxt('${OUTDIR}/rank_atp_results.txt', skiprows=1)
if data.ndim == 1: data = data.reshape(1, -1)
print(f'RANK ATP-aligned (3 seeds):')
print(f'  BACC: {data[:,1].mean():.2f}±{data[:,1].std():.2f}')
print(f'  Kappa: {data[:,2].mean():.2f}±{data[:,2].std():.2f}')
print(f'  F1: {data[:,3].mean():.2f}±{data[:,3].std():.2f}')
print(f'  AUC: {data[:,4].mean():.2f}±{data[:,4].std():.2f}')
" | tee "${OUTDIR}/summary_atp.txt"
echo "[RANK ATP] All done at $(date)"
