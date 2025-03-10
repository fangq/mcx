#!/bin/sh

# first run CW baseline simulation
../../bin/mcx -f baseline.json -q 1 "$@"

# then run RF replay to get RF Jacobian, output data has 6 dimensions
../../bin/mcx -f rfreplay.json -E baseline_detp.jdat -d 0 -F jnii "$@"
