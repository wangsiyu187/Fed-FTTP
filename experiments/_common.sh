#!/bin/bash
# ==============================================================================
# Fed-OCTTP Experiment Common Configuration
# ==============================================================================
# Source this file in experiment scripts under exp/odir/{hybrid,feat,label}/,
# override variables as needed, then call the appropriate helper function.
#
# Usage examples:
#
#   # Data preparation
#   source ../../_common.sh
#   partition='stratified'
#   run_data_prepare
#
#   # Pretraining
#   source ../../_common.sh
#   algorithm='fedprox'
#   test_type='on_site'
#   run_pretrain --prox_mu 0.01
#
#   # ATP training
#   source ../../_common.sh
#   run_atp_train
#
#   # ATP test
#   source ../../_common.sh
#   run_atp_test --labelshift em --calibration bvs
#
#   # TTA test
#   source ../../_common.sh
#   algorithm='tent'
#   lm_lr=0.001
#   run_test
#
#   # TTA test with metric sweep (label shift)
#   source ../../_common.sh
#   algorithm='bn'
#   run_test_metric_loop --labelshift em --calibration bvs

# ==============================================================================
# Default configuration — override in leaf scripts before calling helpers
# ==============================================================================

# GPU
gpu="${gpu:-0}"

# Dataset
dataset='odir_multi'
num_clients=5
partition='step_2_16'
data_holdout=0.2
client_holdout=0.2
corruption='ood'
shift_type="${shift_type:-hybrid}"

# Model
model="${model:-resnet18_multi}"

# Paths — set DATA_DIR before running experiments
_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="$(dirname "${_COMMON_DIR}")"
DATA_DIR="${PROJECT_ROOT}/data"
data_img="${DATA_DIR}/OIA-ODIR_dataset_multi/RGB_preprocessed"
label_txt="${DATA_DIR}/oia_odir/train_labels.txt"
adj_file="${DATA_DIR}/oia_odir/mlgcn_adj_odir_multi.npy"

# Federated learning
algorithm="${algorithm:-fedavg}"
gm_rounds=200
part_rate=1.0

# Local training
lm_lr=0.01
lm_epochs=1
batch_size=20

# Loss
loss="${loss:-ral}"
loss_gamma_pos=0.0
loss_gamma_neg=3.0
loss_tau=0.05
loss_lam=1.5
loss_M=2
loss_N=2

# Testing
test_type="${test_type:-off_site}"

# Reproducibility
partition_seed=0

# ML-GCN
mlgcn_t=0.4
mlgcn_in_ch=300

# ==============================================================================
# Internal helpers
# ==============================================================================

_build_loss_args() {
    echo "--loss ${loss} --loss-gamma-pos ${loss_gamma_pos} --loss-gamma-neg ${loss_gamma_neg} --loss-tau ${loss_tau} --loss-lam ${loss_lam} --loss-M ${loss_M} --loss-N ${loss_N}"
}

_build_base_args() {
    echo "--dataset ${dataset} --num_clients ${num_clients} --partition ${partition} --data_holdout ${data_holdout} --client_holdout ${client_holdout} --partition_seed ${partition_seed} --corruption ${corruption} --model ${model} --data_img ${data_img} --label_txt ${label_txt} --cuda $(_build_loss_args)"
}

_path_tag() {
    echo "${loss}_${model}_${test_type}"
}

_pretrain_load_path() {
    local p="../weights/${dataset}/${shift_type}/pretrain_fedavg_$(_path_tag)_pseed_${partition_seed}_seed_${seed}.pkl"
    if [ ! -f "$p" ] && [ "$test_type" = "off_site" ]; then
        local fb="../weights/${dataset}/${shift_type}/pretrain_fedavg_${loss}_${model}_on_site_pseed_${partition_seed}_seed_${seed}.pkl"
        [ -f "$fb" ] && echo "$fb" || echo "$p"
    else
        echo "$p"
    fi
}

# ==============================================================================
# Helper functions
# ==============================================================================

# Data preparation — calls odir_prepare.py directly
run_data_prepare() {
    cd "${PROJECT_ROOT}/src" || exit 1
    python ./dataset/odir_prepare.py \
        --dataset "${dataset}" \
        --num_clients "${num_clients}" \
        --partition "${partition}" \
        --data_holdout "${data_holdout}" \
        --client_holdout "${client_holdout}" \
        --corruption "${corruption}" \
        --partition_seed "${partition_seed}" \
        --data_img "${data_img}" \
        --label_txt "${label_txt}" \
        "${@}"
}

# Pretraining — trains a model from scratch and saves it
run_pretrain() {
    cd "${PROJECT_ROOT}/src" || exit 1
    for seed in {0..2}; do
        local save_path="../weights/${dataset}/${shift_type}/pretrain_${algorithm}_$(_path_tag)_pseed_${partition_seed}_seed_${seed}.pkl"
        local hist_path="../history/${dataset}/${shift_type}/pretrain_${algorithm}_$(_path_tag)_pseed_${partition_seed}_seed_${seed}.pkl"

        CUDA_VISIBLE_DEVICES=${gpu} python main.py \
            $(_build_base_args) \
            --algorithm "${algorithm}" \
            --gm_rounds "${gm_rounds}" \
            --part_rate "${part_rate}" \
            --lm_lr "${lm_lr}" \
            --lm_epochs "${lm_epochs}" \
            --batch_size "${batch_size}" \
            --seed "${seed}" \
            --history_path "${hist_path}" \
            --save_model_path "${save_path}" \
            --test_type "${test_type}" \
            "${@}"
    done
}

# ATP training — loads a pretrained model and runs ATP adaptation
run_atp_train() {
    cd "${PROJECT_ROOT}/src" || exit 1
    for seed in {0..2}; do
        local load_path="$(_pretrain_load_path)"
        local hist_path="../history/${dataset}/${shift_type}/atp_$(_path_tag)_pseed_${partition_seed}_seed_${seed}.pkl"

        CUDA_VISIBLE_DEVICES=${gpu} python main.py \
            $(_build_base_args) \
            --algorithm atp \
            --gm_rounds "${gm_rounds}" \
            --part_rate "${part_rate}" \
            --lm_lr "${lm_lr}" \
            --lm_epochs "${lm_epochs}" \
            --batch_size "${batch_size}" \
            --seed "${seed}" \
            --history_path "${hist_path}" \
            --load_model_path "${load_path}" \
            --test_type "${test_type}" \
            "${@}"
    done
}

# Standard test — loads a pretrained model and runs a TTA/test algorithm
run_test() {
    cd "${PROJECT_ROOT}/src" || exit 1
    for seed in {0..2}; do
        local load_path="$(_pretrain_load_path)"
        local hist_path="../history/${dataset}/${shift_type}/${algorithm}_$(_path_tag)_pseed_${partition_seed}_seed_${seed}.pkl"

        CUDA_VISIBLE_DEVICES=${gpu} python main.py \
            $(_build_base_args) \
            --algorithm "${algorithm}" \
            --batch_size "${batch_size}" \
            --seed "${seed}" \
            --history_path "${hist_path}" \
            --load_model_path "${load_path}" \
            --test_type "${test_type}" \
            "${@}"
    done
}

# Test with metric sweep — for label shift experiments that evaluate kappa, f1, auc, final
run_test_metric_loop() {
    cd "${PROJECT_ROOT}/src" || exit 1
    for seed in {0..2}; do
        for metric in kappa f1 auc final; do
            local load_path="$(_pretrain_load_path)"
            local hist_path="../history/${dataset}/${shift_type}/${algorithm}_${metric}_$(_path_tag)_pseed_${partition_seed}_seed_${seed}.pkl"

            echo "[$(echo ${algorithm} | tr '[:lower:]' '[:upper:]')] metric=${metric}, seed=${seed}"

            CUDA_VISIBLE_DEVICES=${gpu} python main.py \
                $(_build_base_args) \
                --algorithm "${algorithm}" \
                --batch_size "${batch_size}" \
                --seed "${seed}" \
                --history_path "${hist_path}" \
                --load_model_path "${load_path}" \
                --test_type "${test_type}" \
                --metric "${metric}" \
                "${@}"
        done
    done
}

# ATP test — runs both batch and online_avg test modes with a pretrained+adapted model
run_atp_test() {
    cd "${PROJECT_ROOT}/src" || exit 1
    local tests=('batch' 'online_avg')
    local load_adapt_idx=0
    local load_adapt_round=-1

    for seed in {0..2}; do
        for i in {0,1}; do
            echo "${tests[i]}"
            local load_path="../weights/${dataset}/${shift_type}/pretrain_fedavg_$(_path_tag)_pseed_${partition_seed}_seed_${seed}.pkl"
            local adapt_path="../history/${dataset}/${shift_type}/atp_$(_path_tag)_pseed_${partition_seed}_seed_${seed}.pkl"
            local hist_path="../history/${dataset}/${shift_type}/atp_test_${tests[i]}_$(_path_tag)_pseed_${partition_seed}_seed_${seed}.pkl"

            CUDA_VISIBLE_DEVICES=${gpu} python main.py \
                $(_build_base_args) \
                --algorithm atptest \
                --test "${tests[i]}" \
                --load_adapt_path "${adapt_path}" \
                --load_adapt_idx "${load_adapt_idx}" \
                --load_adapt_round "${load_adapt_round}" \
                --batch_size "${batch_size}" \
                --seed "${seed}" \
                --history_path "${hist_path}" \
                --load_model_path "${load_path}" \
                --test_type "${test_type}" \
                "${@}"
        done
    done
}
