#!/bin/bash
# Fed-OCTTP: ATP training on ODIR hybrid shift
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "${DIR}/../../_common.sh"
gpu=1
part_rate=0.25
lm_lr=0.1
test_type='on_site'
run_atp_train
