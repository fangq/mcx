#!/bin/sh

time ../../bin/mcx --autopilot 1 \
 --photon 1e6 --input bubble.inp --session bubble0 --repeat 1 \
 --array 0 --reflect 0 --skipradius 0
 
time ../../bin/mcx --autopilot 1 \
 --photon 1e6 --input bubble.inp --session bubble3 --repeat 1 \
 --array 0 --reflect 0 --skipradius 3
 
time ../../bin/mcx --autopilot 1 \
 --photon 1e6 --input bubble.inp --session bubble5 --repeat 1 \
 --array 0 --reflect 0 --skipradius 5
