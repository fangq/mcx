#!/bin/sh
#time ../../bin/mcx -t 1792 -g 50 -n 1e8 -f validation_b.inp -s semi_infinite_b -r 160 -a 0 -b 1 -B 0 -U 1
time ../../bin/mcx -A -n 1e8 -f validation_b.inp -s semi_infinite_b -a 0 -b 1 -B 0 -U 1 -R 5
