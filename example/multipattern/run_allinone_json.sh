#!/bin/sh

../../bin/mcx -f multipattern.json --dumpjson > pattern_allinone.json
../../bin/mcx -f pattern_allinone.json $@
