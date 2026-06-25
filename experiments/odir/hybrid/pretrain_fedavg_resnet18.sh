#!/bin/bash
# Fed-OCTTP: FedAvg pretraining on ODIR hybrid shift
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "${DIR}/../../_common.sh"
gpu=0
run_pretrain
