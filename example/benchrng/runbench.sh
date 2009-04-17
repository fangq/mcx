#!/bin/sh
echo ============================================================
echo "running MT RNG using only registers"
make mt
../../bin/rngspeed 128 128 10000

echo ============================================================
echo "running LL5 RNG using only registers"
make logistic
../../bin/rngspeed 128 128 10000

echo ============================================================
echo "running MT RNG with global write (bottleneck)"
make mtw
../../bin/rngspeed 128 128 1000

echo ============================================================
echo "running LL5 RNG with global write (bottleneck)"
make logisticw
../../bin/rngspeed 128 128 1000

