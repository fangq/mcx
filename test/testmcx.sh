#!/bin/sh

fail=0
EXE=mcx
TESTDIR=$(cd "$(dirname "$0")" && pwd)
MCX=$TESTDIR/../bin/$EXE
if [ ! -f "$MCX" ]; then MCX=`which $EXE`; fi
if [ -z "$MCX" ]; then echo "can not find $EXE"; exit 100; fi

# run in a scratch folder so the outputs do not overwrite the files tracked in test/
WORKDIR=$(mktemp -d)
trap 'rm -rf "$WORKDIR"' EXIT
cd "$WORKDIR"

PARAM=$@
LDD=`which ldd`
if [ -z "$LDD" ]; then LDD="otool -L"; fi

echo "test binary file ... "

if [ ! -f "$MCX" ]; then echo "mcx binary does not exit"; fail=$((fail+1)); else echo "ok"; fi

echo "test libraries ... "
temp=`$LDD "$MCX" | grep 'not found'`
if [ ! -z "$temp" ]; then echo "library missing: $temp"; fail=$((fail+1)); else echo "ok"; fi 

echo "test execution permission ... "
if [ ! -x "$MCX" ]; then echo "binary missing executable flag"; fail=$((fail+1)); else echo "ok"; fi

echo "test version number ... "
temp=`"$MCX" --version`
if [ -z "$temp" ]; then echo "fail to print version number"; fail=$((fail+1)); else echo "ok"; fi

echo "test help info ... "
temp=`"$MCX" | grep -o -E '\-\-[a-z]+' | sort | uniq | wc -l`
if [ "$temp" -lt "50" ]; then echo "fail to print all command line flags"; fail=$((fail+1)); else echo "ok"; fi

echo "test gpu info ... "
temp=`"$MCX" -L | grep 'Global [Mm]emory'`
if [ -z "$temp" ]; then echo "fail to print GPU"; fail=$((fail+1)); else echo "ok"; fi

echo "test built-in benchmark listing ... "
temp=`"$MCX" --bench | grep 'cube60'`
if [ -z "$temp" ]; then echo "fail to print built-in benchmarks"; fail=$((fail+1)); else echo "ok"; fi

echo "test exporting json input from builtin examples ... "
temp=`"$MCX" --bench cube60 --dumpjson - | grep '"Name":\s*"cubic60"'`
if [ -z "$temp" ]; then echo "fail to dump json input shape from builtin example"; fail=$((fail+1)); else echo "ok"; fi

echo "test exporting json input from builtin examples with volume data ... "
temp=`"$MCX" --bench colin27 --dumpjson - | grep -o -E '"_ArrayZipData_":\s*"[A-Za-z0-9+/]{20}'`
if [ -z "$temp" ]; then echo "fail to dump json input with volume from builtin example"; fail=$((fail+1)); else echo "ok"; fi

echo "test default options ... "
temp=`"$MCX" --bench cube60 --dumpjson | sed -e 's/\n/|/g' | grep -o -E '"DoMismatch":\s*false,|\s*"DoSaveVolume":\s*true,|\s*"DoNormalize":\s*true,|\s*"DoPartialPath":\s*true,|\s*"DoSaveRef":\s*false,|\s*"DoSaveExit":\s*false,|\s*"DoSaveSeed":\s*false,|\s*"DoAutoThread":\s*true,|\s*"DoDCS":\s*false,|\s*"DoSpecular":\s*false,|\s*"DebugFlag":\s*0,|\s*"SaveDataMask":\s*5,|\s*"OutputFormat":\s*"jnii",|\s*"OutputType":\s*"x"' | wc -l`
if [ "$temp" -ne "14" ]; then echo "fail to verify default options "; fail=$((fail+1)); else echo "ok"; fi

echo "test exporting builtin volume with gzip compression ... "
temp=`"$MCX" --bench colin27 --dumpjson - --zip gzip | grep -o -E '"_ArrayZipData_":\s*"H4sIAAAAAAAA/\+z'`
if [ -z "$temp" ]; then echo "fail to set gzip compression for volume exporting"; fail=$((fail+1)); else echo "ok"; fi

echo "test json input modifier --json ... "
temp=`"$MCX" --bench cube60 --json '{"Optode":{"Source":{"Type":"isotropic","Pos":[29,29,29]}}}' --dumpjson 1 $PARAM | grep 'isotropic'`
if [ -z "$temp" ]; then echo "fail to modify input via --json"; fail=$((fail+1)); else echo "ok"; fi

echo "test homogeneous domain simulation ... "
temp=`"$MCX" --bench cube60 -S 0 $PARAM | grep -o -E 'absorbed:.*17\.[0-9]+%'`
if [ -z "$temp" ]; then echo "fail to run cube60 benchmark"; fail=$((fail+1)); else echo "ok"; fi

echo "test boundary reflection ... "
temp=`"$MCX" --bench cube60b -S 0 $PARAM | grep -o -E 'absorbed:.*27\.[0-9]+%'`
if [ -z "$temp" ]; then echo "fail to run cube60b benchmark"; fail=$((fail+1)); else echo "ok"; fi

echo "test boundary reflection flag -b ... "
temp=`"$MCX" --bench cube60 -b 1 -S 0 $PARAM | grep -o -E 'absorbed:.*27\.[0-9]+%'`
if [ -z "$temp" ]; then echo "fail to use -b 1 flag to enable reflection"; fail=$((fail+1)); else echo "ok"; fi

echo "test boundary condition flag -B ... "
temp=`"$MCX" --bench cube60 -b 0 -B aarraa -S 0 $PARAM | grep -o -E 'absorbed:.*27\.[0-9]+%'`
if [ -z "$temp" ]; then echo "fail to use -B flag to set facet based boundary condition"; fail=$((fail+1)); else echo "ok"; fi

echo "test cyclic boundary condition ... "
temp=`"$MCX" --bench cube60 --bc 'cccccc' $PARAM -n 1e3 -d 0 -S 0 | grep -o -E 'absorbed:.*99\.[0-9]+%'`
if [ -z "$temp" ]; then echo "fail to apply the cylic boundary condition"; fail=$((fail+1)); else echo "ok"; fi

echo "test photon detection ... "
temp=`"$MCX" --bench cube60b $PARAM | grep -o -E 'detected.*4[0-9]+ photons'`
if [ -z "$temp" ]; then echo "fail to detect photons in the cube60b benchmark"; fail=$((fail+1)); else echo "ok"; fi

echo "test planary widefield source ... "
temp=`"$MCX" --bench cube60planar $PARAM | grep -o -E 'absorbed:.*25\.[0-9]+%'`
if [ -z "$temp" ]; then echo "fail to run cube60planar benchmark"; fail=$((fail+1)); else echo "ok"; fi

echo "test isotropic source ... "
temp=`"$MCX" --bench cube60 --json '{"Optode":{"Source":{"Type":"isotropic","Pos":[29,29,29]}}}' -d 0 -S 0 $PARAM | grep -o -E 'absorbed:.*88\.[0-9]+%'`
if [ -z "$temp" ]; then echo "fail to run isotropic source"; fail=$((fail+1)); else echo "ok"; fi

echo "test cone beam source ... "
temp=`"$MCX" --bench cube60 --json '{"Domain":{"Media":[[0,0,1,1],[0.001,0.001,0,1]]},"Optode":{"Source":{"Type":"cone","Param1":[0.5,0,0,0]}}}' -d 0 -S 0 $PARAM | grep -o -E 'absorbed:.*6\.[0-9]+%'`
if [ -z "$temp" ]; then echo "fail to run cone beam source"; fail=$((fail+1)); else echo "ok"; fi

echo "test Fourier source ... "
temp=`"$MCX" --bench cube60planar --json '{"Optode":{"Source":{"Type":"fourier","Param1":[40,0,0,2]}}}' -d 0 -S 0 $PARAM | grep -o -E 'absorbed:.*25\.[0-9]+%'`
if [ -z "$temp" ]; then echo "fail to run Fourier source"; fail=$((fail+1)); else echo "ok"; fi

echo "test pencil array source ... "
temp=`"$MCX" --bench cube60planar --json '{"Optode":{"Source":{"Type":"pencilarray","Param1":[40,0,0,4],"Param2":[0,20,0,2]}}}' -d 0 -S 0 $PARAM | grep -o -E 'absorbed:.*23\.[0-9]+%'`
if [ -z "$temp" ]; then echo "fail to run pencil array source"; fail=$((fail+1)); else echo "ok"; fi

temp=`"$MCX" -Q cube60b -n 1e7 -W 10,10,10 -j '{"Optode":{"Source":{"Type":"unknown"}}}' $PARAM 2>&1`
if [ ! -z "$temp" ]; then
    echo "test unknown source type error handling ... "
    haserror=`echo $temp | grep -o -E 'MCX[A-Z]* ERROR.-.*the specified source type is not supported'`
    if [ -z "$haserror" ]; then echo "fail to catch unknown source type - should return error"; fail=$((fail+1)); else echo "ok"; fi
fi

echo "test boundary detector flags ... "
temp=`"$MCX" --bench cube60 --bc '______111111' $PARAM -n 1e4 | grep -o -E 'detected.*[0-9.]+ photons' | grep -o -E '[0-9.]+ photon' | grep -o -E '9[7-9][0-9.]+'`
if [ -z "$temp" ]; then echo "fail to detect photons in the cube60b benchmark"; fail=$((fail+1)); else echo "ok"; fi

echo "test 2d simulation ... "
temp=`"$MCX" --bench cube60b --json '{"Shapes":[{"Grid":{"Tag":1,"Size":[1,100,100]}},{"Box":{"Tag":2,"O":[0,30,10],"Size":[1,40,40]}}],"Domain":{"Media":[[0,0,1,1],[0.02,0.1,0.9,1.37],[0.02,10,0.9,6.85]]},"Optode":{"Source":{"Pos":[0,50,0],"Dir":[0,0,1]}}}' -d 0 -S 0 $PARAM | grep -o -E 'absorbed:.*6[0-9]\.[0-9]+%'`
if [ -z "$temp" ]; then echo "fail to run 2d simulation"; fail=$((fail+1)); else echo "ok"; fi

echo "test 2d disk source with Lambertian direction ... "
temp=`"$MCX" --bench cube60b --json '{"Shapes":[{"Grid":{"Tag":1,"Size":[1,100,100]}},{"Box":{"Tag":2,"O":[0,30,10],"Size":[1,40,40]}}],"Domain":{"Media":[[0,0,1,1],[0.02,0.1,0.9,1.37],[0.02,10,0.9,6.85]]},"Optode":{"Source":{"Type":"disk","Pos":[0,50,0],"Dir":[0,0,1,"-_Inf_"],"Param1":[20,0,0,0]}}}' -d 0 -S 0 -n 1e5 $PARAM | grep -o -E 'absorbed:.*7[0-9]\.[0-9]+%'`
if [ -z "$temp" ]; then echo "fail to run 2d disk source with Lambertian direction"; fail=$((fail+1)); else echo "ok"; fi

echo "test unitinmm ... "
temp=`"$MCX" --bench skinvessel -S 0 -n 1e5 $PARAM | grep -o -E 'absorbed:.*39\.[0-9]+%'`
if [ -z "$temp" ]; then echo "fail to run skinvessel benchmark"; fail=$((fail+1)); else echo "ok"; fi

echo "test saving photon seeds ... "
temp=`"$MCX" --bench cube60 -q 1 -F jnii -S 0 $PARAM | grep -o -E 'after encoding: 13[0-9]\.[0-9]+%'`
if [ -z "$temp" ]; then echo "fail to save photon seeds"; fail=$((fail+1)); else echo "ok"; fi

echo "test photon replay flag -E ... "
rm -rf replaytest.*
temp=`("$MCX" --bench cube60 -s replaytest -q 1 -S 0 $PARAM && "$MCX" --bench cube60 -E replaytest_detp.jdt -S 0 $PARAM) | sed 's/\x1b\[[0-9;]*m//g' | grep -o -E '(simulated|detected)\s+[0-9.]+ photons' | tail -2 | sed -e 's/^[a-z ]*//g' | sort | uniq -c | grep '^\s*2\s*\d*'`
if [ -z "$temp" ]; then echo "fail to run photon replay -E"; fail=$((fail+1)); else echo "ok"; fi

echo "test photon replay ... "
rm -rf replaytest.*
temp=`("$MCX" --bench cube60 -s replaytest -q 1 -S 0 $PARAM && "$MCX" --bench cube60 -E replaytest_detp.jdt -S 0 $PARAM) | sed 's/\x1b\[[0-9;]*m//g' | grep -o -E 'absorbed:.*3[0-8]\.[0-9]+%'`
if [ -z "$temp" ]; then echo "fail to run photon replay"; fail=$((fail+1)); else echo "ok"; fi

echo "test pattern source photon sharing replay ... "
rm -rf patreplay*
patsrc='{"Optode":{"Source":{"Type":"pattern","Pos":[10,10,0],"Param1":[40,0,0,2],"Param2":[0,40,0,2],"SrcNum":2,"Pattern":{"Nx":2,"Ny":2,"Nz":1,"Data":[1,2,1,2,1,2,1,2]}}}}'
patrf='{"Optode":{"Source":{"Type":"pattern","Pos":[10,10,0],"Param1":[40,0,0,2],"Param2":[0,40,0,2],"SrcNum":2,"Frequency":1e8,"Pattern":{"Nx":2,"Ny":2,"Nz":1,"Data":[1,2,1,2,1,2,1,2]}}}}'
"$MCX" --bench cube60 --json "$patsrc" -s patreplay -q 1 $PARAM > /dev/null 2>&1
# pattern 2 is 2x pattern 1, so both must report the same per-pattern absorbed fraction:
# energyabs and energytot then scale together. Dropping the per-pattern weight or the
# srcnum stride in the replay deposit leaves energyabs equal and splits them 2:1 instead.
temp=`"$MCX" --bench cube60 --json "$patsrc" -E patreplay_detp.jdt -O p -s patreplaywp $PARAM | grep -o -E '[0-9]+\.[0-9][0-9][0-9][0-9][0-9]%' | sed -e 's/\([0-9]*\.[0-9][0-9][0-9]\)[0-9]*%/\1/' | sort | uniq | wc -l`
if [ "$temp" -ne "1" ]; then echo "fail to apply per-pattern weight in wp replay"; fail=$((fail+1)); else echo "ok"; fi

echo "test pattern source photon sharing replay in RF mus mode ... "
temp=`"$MCX" --bench cube60 --json "$patrf" -E patreplay_detp.jdt -O s -s patreplayrfmus $PARAM | grep -o -E '[0-9]+\.[0-9][0-9][0-9][0-9][0-9]%' | sed -e 's/\([0-9]*\.[0-9][0-9][0-9]\)[0-9]*%/\1/' | sort | uniq | wc -l`
if [ "$temp" -ne "1" ]; then echo "fail to apply per-pattern weight in rfmus replay"; fail=$((fail+1)); else echo "ok"; fi

# short runs of the less common host paths: each must finish and report the simulation;
# runok "description" "output file that must exist, or -" mcx options...
runok() {
    desc=$1; outfile=$2; shift 2
    echo "test $desc ... "
    out=`"$MCX" "$@" $PARAM 2>&1 | sed 's/\x1b\[[0-9;]*m//g'`
    if [ -z "`echo "$out" | grep -E 'simulated [0-9]+ photons'`" ] || echo "$out" | grep -q -E 'MCX ERROR'; then
        echo "fail to run $desc"; fail=$((fail+1))
    elif [ "$outfile" != "-" ] && [ ! -s "$outfile" ]; then
        echo "fail to save $outfile for $desc"; fail=$((fail+1))
    else
        echo "ok"
    fi
}

DET1='{"Optode":{"Detector":[{"Pos":[30,20,0],"R":2,"Dir":[0,0,1]},{"Pos":[30,40,0],"R":2,"Dir":[0,0,1]}]}}'
DETRF='{"Optode":{"Source":{"Frequency":1e8},"Detector":[{"Pos":[30,20,0],"R":2,"Dir":[0,0,1]}]}}'
runok "adjoint mua Jacobian" adja.jnii --bench cube60 -n 1e5 -O a -s adja --json "$DET1"
runok "adjoint D Jacobian" adjd.jnii --bench cube60 -n 1e5 -O d -s adjd --json "$DET1"
runok "adjoint mus Jacobian" adju.jnii --bench cube60 -n 1e5 -O u -s adju --json "$DET1"
runok "adjoint mus' Jacobian" adjv.jnii --bench cube60 -n 1e5 -O v -s adjv --json "$DET1"
runok "dual adjoint mua/D Jacobians" adjw.jnii --bench cube60 -n 1e5 -O w -s adjw --json "$DET1"
runok "dual adjoint mua/mus' Jacobians" adjq.jnii --bench cube60 -n 1e5 -O q -s adjq --json "$DET1"
runok "RF dual adjoint Jacobians" adjrf.jnii --bench cube60 -n 1e5 -O w -s adjrf --json "$DETRF"
DETMAP='{"Optode":{"Detector":[{"Pos":[30,20,0],"R":2,"Dir":[0,0,1]},{"Pos":[30,40,0],"R":2,"Dir":[0,0,1]},{"Pos":[40,30,0],"R":2,"Dir":[0,0,1]}],"SDMap":[[1,3],[1,1]]}}'
runok "adjoint Jacobians of the channels in Optode.SDMap" adjmap.jnii --bench cube60 -n 1e5 -O w -s adjmap --json "$DETMAP"
echo "test adjoint Jacobian dimensions follow Optode.SDMap ... "
if [ -z "`tr -d '\n\t ' < adjmap.jnii 2>/dev/null | grep -o '"Dim":\[60,60,60,2,2\]'`" ]; then echo "fail to size the Jacobians by Optode.SDMap"; fail=$((fail+1)); else echo "ok"; fi
runok "RF forward fluence" rffwd.jnii --bench cube60 -n 1e5 -O r -s rffwd --json '{"Optode":{"Source":{"Frequency":1e8}}}'
# deposit paths no other test reaches: reflectance (-X 1) of a non-pattern source, RF replay, and RF forward and replay of patterns
printf '%s' '{"Session":{"ID":"padref","Photons":1e5},"Forward":{"T0":0,"T1":5e-9,"Dt":5e-9},"Domain":{"Dim":[62,62,62],"OriginType":1,"Media":[{"mua":0,"mus":0,"g":1,"n":1},{"mua":0.005,"mus":1,"g":0.01,"n":1}]},"Shapes":[{"Grid":{"Tag":0,"Size":[62,62,62]}},{"Box":{"Tag":1,"O":[1,1,1],"Size":[60,60,60]}}],"Optode":{"Source":{"Type":"pencil","Pos":[31,31,1],"Dir":[0,0,1]}}}' > padref.json
runok "diffuse reflectance (-X 1) of a pencil beam" padref.jnii -f padref.json -X 1 -s padref
"$MCX" --bench cube60 -n 1e5 -q 1 -s rfrep --json '{"Optode":{"Source":{"Frequency":1e8}}}' $PARAM > /dev/null 2>&1
runok "RF replay" rfrepr.jnii --bench cube60 -E rfrep_detp.jdt -O r -s rfrepr --json '{"Optode":{"Source":{"Frequency":1e8}}}'
runok "RF forward of two patterns" patrffwd.jnii --bench cube60 -n 1e5 -O r -s patrffwd --json "$patrf"
"$MCX" --bench cube60 -n 1e5 -q 1 -s patrfrep --json "$patrf" $PARAM > /dev/null 2>&1
runok "RF replay of two patterns" patrfrepr.jnii --bench cube60 -E patrfrep_detp.jdt -O r -s patrfrepr --json "$patrf"
runok "forward fluence of each source and detector slot (srcid -2)" srcm2.jnii --bench cube60 -n 1e5 --srcid -2 -s srcm2 --json "$DET1"
"$MCX" --bench cube60b -n 1e5 -s replaytest -q 1 -w DP -S 0 $PARAM > /dev/null 2>&1
runok "replay of all detectors at once" replayall.jnii --bench cube60b -E replaytest_detp.jdt -Y -1 -O j -s replayall
runok "time gates in groups (-g)" gates.jnii --bench cube60 -n 1e5 -g 2 -s gates --json '{"Forward":{"T0":0,"T1":5e-9,"Dt":1e-9}}'
runok "two simultaneous patterns" pat2.jnii --bench cube60planar -n 1e5 -s pat2 --json '{"Optode":{"Source":{"Type":"pattern","Pos":[10,10,-10],"Param1":[40,0,0,2],"Param2":[0,40,0,2],"SrcNum":2,"Pattern":{"Nx":2,"Ny":2,"Data":[1,0,0,1,0,1,1,0]}}}}'
runok "3-D pattern source" pat3d.jnii --bench cube60 -n 1e5 -s pat3d --json '{"Optode":{"Source":{"Type":"pattern3d","Pos":[20,20,20],"Param1":[2,2,2,0],"Pattern":{"Nx":2,"Ny":2,"Nz":2,"Data":[1,1,1,1,1,1,1,1]}}}}'
runok "unnormalized output (-U 0) of an energy deposit" udep.jnii --bench cube60 -n 1e5 -U 0 -O e -s udep
runok "continuous media, wide-field source, no reflection" cont0.jnii -f "$TESTDIR/builtin/mcxlab_demo_continuous_mua_mus_1.json" -n 1e5 -b 0 -s cont0 --json '{"Optode":{"Source":{"Type":"disk","Param1":[5,0,0,0]}}}'
runok "continuous media, wide-field source, with reflection" cont1.jnii -f "$TESTDIR/builtin/mcxlab_demo_continuous_mua_mus_1.json" -n 1e5 -b 1 -s cont1 --json '{"Optode":{"Source":{"Type":"disk","Param1":[5,0,0,0]}}}'
runok "polarized light with reflection, wide-field source" pol1.jnii -f "$TESTDIR/builtin/mcxlab_demo_polarized_photon_1.json" -n 1e5 -b 1 -s pol1
runok "polarized light with reflection, pencil beam" pol2.jnii -f "$TESTDIR/builtin/mcxlab_demo_polarized_photon_1.json" -n 1e5 -b 1 -s pol2 --json '{"Optode":{"Source":{"Type":"pencil"}}}'
runok "split-voxel MC, wide-field source, no reflection" svmc0.jnii -f "$TESTDIR/builtin/mcxlab_demo_svmc_cubesph_2.json" -n 1e5 -b 0 -s svmc0
runok "split-voxel MC, pencil beam, no reflection" svmc1.jnii -f "$TESTDIR/builtin/svmcsphshells_svmc_sphshells.json" -n 1e5 -b 0 -s svmc1
runok "continuous media, pencil beam, no reflection" cont2.jnii -f "$TESTDIR/builtin/mcxlab_demo_continuous_mua_mus_1.json" -n 1e5 -b 0 -s cont2 --json '{"Optode":{"Source":{"Type":"pencil"}}}'
runok "repeated runs accumulated (-r 2)" resp2.jnii --bench cube60 -n 1e5 -r 2 -s resp2
runok "photons split over repeated runs (-r -2)" respm2.jnii --bench cube60 -n 1e5 -r -2 -s respm2
runok "manual thread and block sizes (-A 0)" manual.jnii --bench cube60 -n 1e5 -A 0 -t 16384 -T 64 -g 10 -s manual
runok "3-D pattern source shared by two patterns" pat3d2.jnii --bench cube60 -n 1e5 -s pat3d2 --json '{"Optode":{"Source":{"Type":"pattern3d","Pos":[20,20,20],"Param1":[2,2,2,0],"SrcNum":2,"Pattern":{"Nx":2,"Ny":2,"Nz":2,"Data":[1,0,1,0,1,0,1,0,0,1,0,1,0,1,0,1]}}}}'
runok "energy deposit of two simultaneous patterns" pat2e.jnii --bench cube60planar -n 1e5 -O e -s pat2e --json '{"Optode":{"Source":{"Type":"pattern","Pos":[10,10,-10],"Param1":[40,0,0,2],"Param2":[0,40,0,2],"SrcNum":2,"Pattern":{"Nx":2,"Ny":2,"Data":[1,0,0,1,0,1,1,0]}}}}'

echo "test heterogeneous domain ... "
temp=`"$MCX" --bench spherebox -S 0 $PARAM | grep -o -E 'absorbed:.*1[01]\.[0-9]+%'`
if [ -z "$temp" ]; then echo "fail to run spherebox benchmark"; fail=$((fail+1)); else echo "ok"; fi

echo "test save detect photon data flag -w ... "
temp=`"$MCX" --bench cube60 -w dspxvw -F jnii -S 0 $PARAM | grep -o -E 'compressing data \[zlib\]' | uniq -c | grep '^\s*6\s*compressing'`
if [ -z "$temp" ]; then echo "fail to save detected photon data using -w flag"; fail=$((fail+1)); else echo "ok"; fi

echo "test progress bar -D P ... "
temp=`"$MCX" --bench cube60 -D P  $PARAM | grep 'Progress: .* 100%'`
if [ -z "$temp" ]; then echo "fail to print progress bar"; fail=$((fail+1)); else echo "ok"; fi

echo "test random number generator -D R ... "
temp=`"$MCX" --bench cube60 -D R $PARAM | grep 'generating 216000 random numbers'`
if [ -z "$temp" ]; then echo "fail to output random numbers via -D R"; fail=$((fail+1)); else echo "ok"; fi

echo "test random number generator ... "
rm -rf testrng222.jnii
temp=`("$MCX" --bench cube60 -d 0 -s testrng222 --json '{"Shapes":[{"Grid":{"Tag":0,"Size":[2,2,2]}}]}' -F jnii -D R) && (grep 'eJx7Vx1tP2k7k71dbK29xRJe\+w\/JJfZ3Qqvs783Ot79008UeAOTRDlI' testrng222.jnii)`
if [ -z "$temp" ] || [ ! -f testrng222.jnii ] ; then echo "fail to create random numbers"; fail=$((fail+1)); else echo "ok"; fi

echo "test saving trajectory feature -D M ... "
temp=`"$MCX" --bench cube60 -D M -S 0 -d 0 $PARAM -n 1e2 | grep -o -E 'saved trajectory positions: ([6-9][0-9]{3}|[1-9][0-9]{4,})'`
if [ -z "$temp" ]; then echo "fail to save trajectory data via -D M"; fail=$((fail+1)); else echo "ok"; fi

temp=`which valgrind 2> /dev/null`
if [ ! -z "$temp" ]; then
    echo "test memory access errors using valgrind ... "
    temp=`valgrind --log-fd=1 "$MCX" --bench cube60planar --shapes '{"Shapes":[{"Sphere":{"Tag":2,"O":[30,30,10],"R":"10"}}]}' --json '{"Optode":{"Source":{"Type":"fourier","Param1":[40,0,0,2]}}}' -w dpw $PARAM -n 1e4`
    haserror=`echo $temp | grep -o -E 'MCX[A-Z]* ERROR.-'`
    temp=`echo $temp | grep -o -E '=+\s+Invalid\s+'`
    if [ ! -z "$haserror" ] || [ ! -z "$temp" ]; then echo "fail to pass valgrind memory check"; fail=$((fail+1)); else echo "ok"; fi
fi

# cuda-memcheck was removed in CUDA 12, prefer compute-sanitizer when present
MEMCHECK=`which compute-sanitizer 2> /dev/null || which cuda-memcheck 2> /dev/null`
if [ ! -z "$MEMCHECK" ]; then
    echo "test gpu memory errors using `basename $MEMCHECK` ... "
    temp=`"$MEMCHECK" "$MCX" --bench cube60planar --shapes '{"Shapes":[{"Sphere":{"Tag":2,"O":[30,30,10],"R":"10"}}]}' -B 'ararar1' -w dpw $PARAM -n 1e5`
    haserror=`echo $temp | grep -o -E 'MCX[A-Z]* ERROR.-'`
    temp=`echo $temp | grep -o -E '=+\s+ERROR SUMMARY:\s+0\s+error'`
    if [ ! -z "$haserror" ] || [ -z "$temp" ]; then echo "fail to pass cuda memory check"; fail=$((fail+1)); else echo "ok"; fi
fi

temp=`which python3 2> /dev/null`
if [ ! -z "$temp" ] && python3 -c "import numpy" 2> /dev/null; then
    echo "test step-by-step photon trajectories against golden data ... "
    temp=`MCX="$MCX" "$TESTDIR/testtraj.sh" $PARAM 2>&1`
    if [ $? -ne 0 ]; then echo "$temp" | grep -E 'FAILED|diverges|fail' | head -10; echo "fail to reproduce golden trajectories"; fail=$((fail+1)); else echo "ok"; fi
fi

if [ "$fail" -gt "0" ];  then
    echo "failed $fail tests";
    exit $fail;
else
    echo "passed all tests!";
fi
