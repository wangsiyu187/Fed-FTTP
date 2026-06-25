#!/bin/bash
# Fed-OCTTP: ODIR hybrid shift data preparation
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "${DIR}/../../_common.sh"
run_data_prepare
