#!/bin/sh

if [ ! -f /usr/bin/clang-format-3.5 ]
then
	echo "WARNING:"
	echo "\tThe clang-format-3.5 is missing in your system. In a debian-based\n"\
	     "\toperation system, run the following command to install the package.\n\n"\
	     "\t\tsudo apt-get install clang-format-3.5\n"
	exit 1
fi

# add file types 
filetype="**.cu"
for target in $filetype
do
	find . -name "$target" -print0 | xargs -0 clang-format-3.5 -i -style=Google	
done 
