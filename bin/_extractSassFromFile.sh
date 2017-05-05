#!/bin/bash

filename=$1
echo "Processing" $filename 

## find the lines with SASS
sed -n '/\/\*/p' $filename > tmp 

## remove the leading space
#sed -i.bak 's/^[ \t]*//' $filename
sed -i 's/^[ \t]*//' tmp 

## delete predicates
sed -i 's/@!P[0-9]*//g' tmp 
sed -i 's/@P[0-9]*//g' tmp 

## remove lines with control code
sed -i '/^\/\* /d' tmp 

## remove lines with NOP and EXIT
sed -i '/NOP\|EXIT/d' tmp 

## delete { and } 
sed -i -e 's/{//g' tmp 
sed -i -e 's/}//g' tmp 

## awk to print SASS and reditect them to a file
awk -F" " '{print $2;}' tmp > "$filename".result

rm tmp

echo "Done! Check" "$filename".result
