#!/bin/sh
grep '^simulat' speedtest.log | sed -e :a -e '/ms$/N; s/ms\n/ /; ta' -e 's/[a-zA-Z\.]//g' | awk '{print $4 "\t" $5 "\t" $6 "\t" $7}'
