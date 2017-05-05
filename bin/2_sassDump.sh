#!/bin/bash
app=$1
#echo $app

app_name=${app:2} # remove ./
#echo $app_name

out_file="$app_name.sm52.sass"

cuobjdump -sass  -arch=sm_52 $app > $out_file 2>&1

echo "SASS are dumped. => $out_file"
