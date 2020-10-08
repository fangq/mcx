#!/bin/sh

fail=0
EXE=mcx
MCX=../bin/$EXE
if [ ! -f "$MCX" ]; then MCX=`which $EXE`; fi
if [ -z "$MCX" ]; then echo "can not find $EXE"; exit 100; fi

PARAM=$@
LDD=`which ldd`
if [ -z "$LDD" ]; then LDD="otool -L"; fi

echo "test binary file ... "

if [ ! -f $MCX ]; then echo "mcx binary does not exit"; fail=$((fail+1)); else echo "ok"; fi

echo "test libraries ... "
temp=`$LDD $MCX | grep 'not found'`
if [ ! -z "$temp" ]; then echo "library missing: $temp"; fail=$((fail+1)); else echo "ok"; fi 

echo "test execution permission ... "
if [ ! -x $MCX ]; then echo "binary missing executable flag"; fail=$((fail+1)); else echo "ok"; fi

echo "test version number ... "
temp=`$MCX --version`
if [ -z "$temp" ]; then echo "fail to print version number"; fail=$((fail+1)); else echo "ok"; fi

echo "test help info ... "
temp=`$MCX | grep -o -E '\-\-[a-z]+' | sort | uniq | wc -l`
if [ "$temp" -lt "50" ]; then echo "fail to print all command line flags"; fail=$((fail+1)); else echo "ok"; fi

echo "test gpu info ... "
temp=`$MCX -L | grep 'Global [Mm]emory'`
if [ -z "$temp" ]; then echo "fail to print GPU"; fail=$((fail+1)); else echo "ok"; fi

echo "test built-in benchmark listing ... "
temp=`$MCX --bench | grep 'cube60'`
if [ -z "$temp" ]; then echo "fail to print built-in benchmarks"; fail=$((fail+1)); else echo "ok"; fi

echo "test exporting json input from builtin examples ... "
temp=`$MCX --bench cube60 --dumpjson - | grep '"Name":\s*"cubic60"'`
if [ -z "$temp" ]; then echo "fail to dump json input shape from builtin example"; fail=$((fail+1)); else echo "ok"; fi

echo "test exporting json input from builtin examples with volume data ... "
temp=`$MCX --bench colin27 --dumpjson - | grep '"_ArrayZipData_":\s*"eJzs3Yl666oOBe'`
if [ -z "$temp" ]; then echo "fail to dump json input with volume from builtin example"; fail=$((fail+1)); else echo "ok"; fi

echo "test default options ... "
temp=`$MCX --bench cube60 --dumpjson | sed -e 's/\n/|/g' | grep -o -E '"DoMismatch":\s*false,|\s*"DoSaveVolume":\s*true,|\s*"DoNormalize":\s*true,|\s*"DoPartialPath":\s*true,|\s*"DoSaveRef":\s*false,|\s*"DoSaveExit":\s*false,|\s*"DoSaveSeed":\s*false,|\s*"DoAutoThread":\s*true,|\s*"DoDCS":\s*false,|\s*"DoSpecular":\s*false,|\s*"DebugFlag":\s*0,|\s*"SaveDataMask":\s*5,|\s*"OutputFormat":\s*"mc2",|\s*"OutputType":\s*"x"' | wc -l`
if [ "$temp" -ne "14" ]; then echo "fail to verify default options "; fail=$((fail+1)); else echo "ok"; fi

echo "test exporting builtin volume with gzip compression ... "
temp=`$MCX --bench colin27 --dumpjson - --zip gzip | grep -o -E '"_ArrayZipData_":\s*"H4sIAAAAAAAA..z'`
if [ -z "$temp" ]; then echo "fail to set gzip compression for volume exporting"; fail=$((fail+1)); else echo "ok"; fi

echo "test json input modifier --json ... "
temp=`$MCX --bench cube60 --json '{"Optode":{"Source":{"Type":"isotropic","Pos":[29,29,29]}}}' --dumpjson 1 $PARAM | grep 'isotropic'`
if [ -z "$temp" ]; then echo "fail to modify input via --json"; fail=$((fail+1)); else echo "ok"; fi

echo "test homogeneous domain simulation ... "
temp=`$MCX --bench cube60 -S 0 $PARAM | grep -o -E 'absorbed:.*17\.[0-9]+%'`
if [ -z "$temp" ]; then echo "fail to run cube60 benchmark"; fail=$((fail+1)); else echo "ok"; fi

echo "test boundary reflection ... "
temp=`$MCX --bench cube60b -S 0 $PARAM | grep -o -E 'absorbed:.*27\.[0-9]+%'`
if [ -z "$temp" ]; then echo "fail to run cube60b benchmark"; fail=$((fail+1)); else echo "ok"; fi

echo "test boundary reflection flag -b ... "
temp=`$MCX --bench cube60 -b 1 -S 0 $PARAM | grep -o -E 'absorbed:.*27\.[0-9]+%'`
if [ -z "$temp" ]; then echo "fail to use -b 1 flag to enable reflection"; fail=$((fail+1)); else echo "ok"; fi

echo "test boundary condition flag -B ... "
temp=`$MCX --bench cube60 -b 0 -B aarraa -S 0 $PARAM | grep -o -E 'absorbed:.*27\.[0-9]+%'`
if [ -z "$temp" ]; then echo "fail to use -B flag to set facet based boundary condition"; fail=$((fail+1)); else echo "ok"; fi

echo "test cyclic boundary condition ... "
temp=`$MCX --bench cube60 --bc 'cccccc' $PARAM -n 1e3 | grep -o -E 'absorbed:.*99\.[0-9]+%'`
if [ -z "$temp" ]; then echo "fail to apply the cylic boundary condition"; fail=$((fail+1)); else echo "ok"; fi

echo "test photon detection ... "
temp=`$MCX --bench cube60b $PARAM | grep -o -E 'detected.*4[0-9]+ photons'`
if [ -z "$temp" ]; then echo "fail to detect photons in the cube60b benchmark"; fail=$((fail+1)); else echo "ok"; fi

echo "test planary widefield source ... "
temp=`$MCX --bench cube60planar $PARAM | grep -o -E 'absorbed:.*25\.[0-9]+%'`
if [ -z "$temp" ]; then echo "fail to run cube60planar benchmark"; fail=$((fail+1)); else echo "ok"; fi

echo "test isotropic source ... "
temp=`$MCX --bench cube60 --json '{"Optode":{"Source":{"Type":"isotropic","Pos":[29,29,29]}}}' -d 0 -S 0 $PARAM | grep -o -E 'absorbed:.*88\.[0-9]+%'`
if [ -z "$temp" ]; then echo "fail to run isotropic source"; fail=$((fail+1)); else echo "ok"; fi

echo "test cone beam source ... "
temp=`$MCX --bench cube60 --json '{"Domain":{"Media":[[0,0,1,1],[0.001,0.001,0,1]]},"Optode":{"Source":{"Type":"cone","Param1":[0.5,0,0,0]}}}' -d 0 -S 0 $PARAM | grep -o -E 'absorbed:.*6\.[0-9]+%'`
if [ -z "$temp" ]; then echo "fail to run cone beam source"; fail=$((fail+1)); else echo "ok"; fi

echo "test Fourier source ... "
temp=`$MCX --bench cube60planar --json '{"Optode":{"Source":{"Type":"fourier","Param1":[40,0,0,2]}}}' -d 0 -S 0 $PARAM | grep -o -E 'absorbed:.*25\.[0-9]+%'`
if [ -z "$temp" ]; then echo "fail to run Fourier source"; fail=$((fail+1)); else echo "ok"; fi

echo "test pencil array source ... "
temp=`$MCX --bench cube60planar --json '{"Optode":{"Source":{"Type":"pencilarray","Param1":[40,0,0,4],"Param2":[0,20,0,2]}}}' -d 0 -S 0 $PARAM | grep -o -E 'absorbed:.*23\.[0-9]+%'`
if [ -z "$temp" ]; then echo "fail to run pencil array source"; fail=$((fail+1)); else echo "ok"; fi

echo "test boundary detector flags ... "
temp=`$MCX --bench cube60 --bc '______111111' $PARAM -n 1e4 | grep -o -E 'detected.*[0-9.]+ photons' | grep -o -E '[0-9.]+ photon' | grep -o -E '9[7-9][0-9.]+'`
if [ -z "$temp" ]; then echo "fail to detect photons in the cube60b benchmark"; fail=$((fail+1)); else echo "ok"; fi

echo "test saving photon seeds ... "
temp=`$MCX --bench cube60 -q 1 -F jnii -S 0 $PARAM | grep -o -E 'after encoding: 13[0-9]\.[0-9]+%'`
if [ -z "$temp" ]; then echo "fail to save photon seeds"; fail=$((fail+1)); else echo "ok"; fi

echo "test photon replay flag -E ... "
rm -rf replaytest.*
temp=`($MCX --bench cube60 -s replaytest -q 1 -S 0 $PARAM && $MCX --bench cube60 -E replaytest.mch -S 0 $PARAM) | sed 's/\x1b\[[0-9;]*m//g' | grep -o -E 'detected.*[0-9.]+ photons' | sort | uniq -c | grep '^\s*2\s*detected'`
if [ -z "$temp" ]; then echo "fail to run photon replay -E"; fail=$((fail+1)); else echo "ok"; fi

echo "test photon replay ... "
rm -rf replaytest.*
temp=`($MCX --bench cube60 -s replaytest -q 1 -S 0 $PARAM && $MCX --bench cube60 -E replaytest.mch -S 0 $PARAM) | sed 's/\x1b\[[0-9;]*m//g' | grep -o -E 'absorbed:.*3[0-8]\.[0-9]+%'`
if [ -z "$temp" ]; then echo "fail to run photon replay"; fail=$((fail+1)); else echo "ok"; fi

echo "test heterogeneous domain ... "
temp=`$MCX --bench spherebox -S 0 $PARAM | grep -o -E 'absorbed:.*1[01]\.[0-9]+%'`
if [ -z "$temp" ]; then echo "fail to run spherebox benchmark"; fail=$((fail+1)); else echo "ok"; fi

echo "test save detect photon data flag -w ... "
temp=`$MCX --bench cube60 -w dspxvw -F jnii -S 0 $PARAM | grep -o -E 'compressing data \[zlib\]' | uniq -c | grep '^\s*6\s*compressing'`
if [ -z "$temp" ]; then echo "fail to save detected photon data using -w flag"; fail=$((fail+1)); else echo "ok"; fi

echo "test progress bar -D P ... "
temp=`$MCX --bench cube60 -D P  $PARAM | grep 'Progress: .* 100%'`
if [ -z "$temp" ]; then echo "fail to print progress bar"; fail=$((fail+1)); else echo "ok"; fi

echo "test random number generator -D R ... "
temp=`$MCX --bench cube60 -D R $PARAM | grep 'generating 216000 random numbers'`
if [ -z "$temp" ]; then echo "fail to output random numbers via -D R"; fail=$((fail+1)); else echo "ok"; fi

echo "test random number generator ... "
rm -rf testrng222.jnii
temp=`($MCX --bench cube60 -d 0 -s testrng222 --json '{"Shapes":[{"Grid":{"Tag":0,"Size":[2,2,2]}}]}' -F jnii -D R) && (grep 'eJx7Vx1tP2k7k71dbK29xRJe\+w\/JJfZ3Qqvs783Ot79008UeAOTRDlI' testrng222.jnii)`
if [ -z "$temp" ] || [ ! -f testrng222.jnii ] ; then echo "fail to create random numbers"; fail=$((fail+1)); else echo "ok"; fi

echo "test saving trajectory feature -D M ... "
temp=`$MCX --bench cube60 -D M -S 0 -d 0 $PARAM -n 1e2 | grep -o -E 'saved [6-9][0-9]+ trajectory'`
if [ -z "$temp" ]; then echo "fail to save trajectory data via -D M"; fail=$((fail+1)); else echo "ok"; fi

temp=`which valgrind 2> /dev/null`
if [ ! -z "$temp" ]; then
    echo "test memory access errors using valgrind ... "
    temp=`valgrind --log-fd=1 $MCX --bench cube60planar --shapes '{"Shapes":[{"Sphere":{"Tag":2,"O":[30,30,10],"R":"10"}}]}' --json '{"Optode":{"Source":{"Type":"fourier","Param1":[40,0,0,2]}}}' -w dpw $PARAM -n 1e4`
    haserror=`echo $temp | grep -o -E 'MCX[A-Z]* ERROR.-'`
    temp=`echo $temp | grep -o -E '=+\s+Invalid\s+'`
    if [ ! -z "$haserror" ] || [ ! -z "$temp" ]; then echo "fail to pass valgrind memory check"; fail=$((fail+1)); else echo "ok"; fi
fi

temp=`which cuda-memcheck 2> /dev/null`
if [ ! -z "$temp" ]; then
    echo "test gpu memory errors using cuda-memcheck ... "
    temp=`cuda-memcheck $MCX --bench cube60planar --shapes '{"Shapes":[{"Sphere":{"Tag":2,"O":[30,30,10],"R":"10"}}]}' -B 'ararar1' -w dpw $PARAM -n 1e5`
    haserror=`echo $temp | grep -o -E 'MCX[A-Z]* ERROR.-'`
    temp=`echo $temp | grep -o -E '=+\s+ERROR SUMMARY:\s+0\s+error'`
    if [ ! -z "$haserror" ] || [ -z "$temp" ]; then echo "fail to pass cuda memory check"; fail=$((fail+1)); else echo "ok"; fi
fi


if [ "$fail" -gt "0" ];  then
    echo "failed $fail tests";
    exit $fail;
else
    echo "passed all tests!";
fi
