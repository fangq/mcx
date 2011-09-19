#!/bin/sh
if [ ! -e semi8x8x8.bin ]; then
  dd if=/dev/zero of=semi8x8x8.bin bs=8 count=64
  perl -pi -e 's/\x0/\x1/g' semi8x8x8.bin
fi

../../bin/mcx -f mtest.inp -s mtest -M 1

