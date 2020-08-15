#!/bin/sh

fail=0
MCX=../bin/mcx
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
if [ "$temp" -lt "40" ]; then echo "fail to print all command line flags"; fail=$((fail+1)); else echo "ok"; fi

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

echo "test exporting builtin volume with gzip compression ... "
temp=`$MCX --bench colin27 --dumpjson - --zip gzip | grep -o -E '"_ArrayZipData_":\s*"H4sIAAAAAAAA[EA]\+z'`
if [ -z "$temp" ]; then echo "fail to set gzip compression for volume exporting"; fail=$((fail+1)); else echo "ok"; fi

echo "test homogeneous domain simulation ... "
temp=`$MCX --bench cube60 $PARAM | grep -o -E 'absorbed:.*17\.[0-9]+%'`
if [ -z "$temp" ]; then echo "fail to run cube60 benchmark"; fail=$((fail+1)); else echo "ok"; fi

echo "test boundary reflection ... "
temp=`$MCX --bench cube60b $PARAM | grep -o -E 'absorbed:.*27\.[0-9]+%'`
if [ -z "$temp" ]; then echo "fail to run cube60b benchmark"; fail=$((fail+1)); else echo "ok"; fi

echo "test boundary reflection flag -b ... "
temp=`$MCX --bench cube60 -b 1 $PARAM | grep -o -E 'absorbed:.*27\.[0-9]+%'`
if [ -z "$temp" ]; then echo "fail to use -b 1 flag to enable reflection"; fail=$((fail+1)); else echo "ok"; fi

echo "test boundary condition flag -B ... "
temp=`$MCX --bench cube60 -b 0 -B aarraa $PARAM | grep -o -E 'absorbed:.*27\.[0-9]+%'`
if [ -z "$temp" ]; then echo "fail to use -B flag to set facet based boundary condition"; fail=$((fail+1)); else echo "ok"; fi

echo "test photon detection ... "
temp=`$MCX --bench cube60b $PARAM | grep -o -E 'detected.*4[0-9]+ photons'`
if [ -z "$temp" ]; then echo "fail to detect photons in the cube60b benchmark"; fail=$((fail+1)); else echo "ok"; fi

echo "test planary widefield source ... "
temp=`$MCX --bench cube60planar $PARAM | grep -o -E 'absorbed:.*25\.[0-9]+%'`
if [ -z "$temp" ]; then echo "fail to run cube60planar benchmark"; fail=$((fail+1)); else echo "ok"; fi

echo "test heterogeneous domain ... "
temp=`$MCX --bench spherebox $PARAM | grep -o -E 'absorbed:.*1[01]\.[0-9]+%'`
if [ -z "$temp" ]; then echo "fail to run spherebox benchmark"; fail=$((fail+1)); else echo "ok"; fi

echo "test progress bar ... "
temp=`$MCX --bench cube60 -D P  $PARAM | grep 'Progress: .* 100%'`
if [ -z "$temp" ]; then echo "fail to print progress bar"; fail=$((fail+1)); else echo "ok"; fi


if [ "$fail" -gt "0" ];  then
    echo "failed $fail tests";
    exit $fail;
else
    echo "passed all tests!";
fi
