#!/bin/bash

nvprof --print-gpu-trace --csv "$@"  2> trace.csv
nvprof --metrics all --csv "$@"  2> metrics.csv
