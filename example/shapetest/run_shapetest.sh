#/bin/sh

../../bin/mcx -f jsonshape_allinone.json -M
../../bin/mcx -f jsonshape.json -M

hexdump jsonshape.mask > jsonshape.txt
hexdump jsonshape_one.mask > jsonshape_one.txt
diff jsonshape.txt jsonshape_one.txt

../../bin/mcx -f jsonshape_allinone.json
../../bin/mcx -f jsonshape.json
