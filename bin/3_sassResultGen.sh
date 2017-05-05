#!/bin/bash

## search all files with .sass suffix
target=`find . -iname "*.sass"`
#echo $target

#for f in $target;
#do
#	echo $f
#done


## extract filename, removing the leading symbol ./
filename=$(echo $target | sed -e "s/\.\///g")
#echo $filename

## 
for f in $filename;
do
	echo -e "\n-------\n$f\n-------"
	./_extractSassFromFile.sh $f
done

#
#
###  accumulate the results into one file
#cat ./*.result >> sass_list
