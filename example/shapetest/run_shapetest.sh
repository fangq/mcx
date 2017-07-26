#/bin/sh

../../bin/mcx -f jsonshape_allinone.json -M
../../bin/mcx -f jsonshape.json -M
../../bin/mcx -f jsonshape.json -s "jsonshape_cmd" -P '{"Shapes":[{"ZLayers":[[1,10,1],[11,30,2],[31,50,3]]}]}' -M

hexdump jsonshape_vol.nii > jsonshape.txt
hexdump jsonshape_one_vol.nii > jsonshape_one.txt
diff jsonshape.txt jsonshape_one.txt

../../bin/mcx -f jsonshape_allinone.json
../../bin/mcx -f jsonshape.json

