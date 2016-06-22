#!/bin/bash

echo "Script running"

outputFile="outputs/scriptOutput"
outputFileSpeed="outputs/scriptOutputSpeed"
outputFileAbsorbtion="outputs/scriptOutputAbsorbtion"
num="1e6"
device="1"

rm $outputFile

#: <<'block1'
../../bin/mcx --autopilot --reflect 0 --photon $num --gpu $device --input 0benchmark-pencil.json 	>> $outputFile
../../bin/mcx --autopilot --reflect 0 --photon $num --gpu $device --input 1benchmark-isotropic.json 	>> $outputFile
../../bin/mcx --autopilot --reflect 0 --photon $num --gpu $device --input 2benchmark-cone.json 		>> $outputFile
../../bin/mcx --autopilot --reflect 0 --photon $num --gpu $device --input 3benchmark-gaussian.json 	>> $outputFile
../../bin/mcx --autopilot --reflect 0 --photon $num --gpu $device --input 4benchmark-planar.json 	>> $outputFile
../../bin/mcx --autopilot --reflect 0 --photon $num --gpu $device --input 6benchmark-fourier.json 	>> $outputFile
../../bin/mcx --autopilot --reflect 0 --photon $num --gpu $device --input 7benchmark-arcsine.json 	>> $outputFile
../../bin/mcx --autopilot --reflect 0 --photon $num --gpu $device --input 8benchmark-disk.json 		>> $outputFile
../../bin/mcx --autopilot --reflect 0 --photon $num --gpu $device --input 9benchmark-fourierx.json 	>> $outputFile
../../bin/mcx --autopilot --reflect 0 --photon $num --gpu $device --input 10benchmark-fourierx2d.json 	>> $outputFile
../../bin/mcx --autopilot --reflect 0 --photon $num --gpu $device --input 11benchmark-zgaussian.json 	>> $outputFile
../../bin/mcx --autopilot --reflect 0 --photon $num --gpu $device --input 13benchmark-slit.json 	>> $outputFile
#block1

: <<'block0'
../../bin/mcx --autopilot --reflect 0 --photon $num --gpu $device --input 0benchmark-pencil.json	| tee -a $outputFile
../../bin/mcx --autopilot --reflect 0 --photon $num --gpu $device --input 1benchmark-isotropic.json	| tee -a $outputFile
../../bin/mcx --autopilot --reflect 0 --photon $num --gpu $device --input 2benchmark-cone.json		| tee -a $outputFile
../../bin/mcx --autopilot --reflect 0 --photon $num --gpu $device --input 3benchmark-gaussian.json	| tee -a $outputFile
../../bin/mcx --autopilot --reflect 0 --photon $num --gpu $device --input 4benchmark-planar.json	| tee -a $outputFile
../../bin/mcx --autopilot --reflect 0 --photon $num --gpu $device --input 6benchmark-fourier.json	| tee -a $outputFile
../../bin/mcx --autopilot --reflect 0 --photon $num --gpu $device --input 7benchmark-arcsine.json	| tee -a $outputFile
../../bin/mcx --autopilot --reflect 0 --photon $num --gpu $device --input 8benchmark-disk.json		| tee -a $outputFile
../../bin/mcx --autopilot --reflect 0 --photon $num --gpu $device --input 9benchmark-fourierx.json	| tee -a $outputFile
../../bin/mcx --autopilot --reflect 0 --photon $num --gpu $device --input 10benchmark-fourierx2d.json	| tee -a $outputFile
../../bin/mcx --autopilot --reflect 0 --photon $num --gpu $device --input 11benchmark-gaussian.json	| tee -a $outputFile
../../bin/mcx --autopilot --reflect 0 --photon $num --gpu $device --input 13benchmark-slit.json		| tee -a $outputFile
block0

egrep "MCX simulation speed|MCX ERROR" $outputFile > $outputFileSpeed
egrep "absorbed:" $outputFile > $outputFileAbsorbtion

echo "=============================================="
echo "Results Simuation Speed [pencil | cone | gaussian | planar | disk]"
cat $outputFileSpeed
echo "=============================================="
echo "Results Absorption [pencil | cone | gaussian | planar | disk]"
cat $outputFileAbsorbtion
echo "=============================================="

echo "Script finished"
