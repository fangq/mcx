#!/bin/sh

# run polarized (Horizontal polarization) MC simulation inside a homogeneous slab (cyclic boundary condition along x and y)
../../bin/mcx -f onelayer.json "$@"
