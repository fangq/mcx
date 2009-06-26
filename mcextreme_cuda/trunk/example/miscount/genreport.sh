#!/bin/sh
grep '<mcx_se' miscnt.log > res1.txt
grep '^expected' miscnt.log | awk '{print $NF}' > res2.txt
paste res1.txt res2.txt
