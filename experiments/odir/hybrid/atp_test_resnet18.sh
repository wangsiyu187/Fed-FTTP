#!/bin/bash
# Fed-OCTTP: ATP test (batch + online_avg) on ODIR hybrid shift
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "${DIR}/../../_common.sh"
test_type='on_site'
run_atp_test
