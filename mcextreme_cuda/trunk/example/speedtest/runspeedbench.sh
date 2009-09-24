#!/bin/sh

RT=`pwd`

# generate a 60x60x60 homogeneous medium filled with index 1

dd if=/dev/zero of=seg60x60x60.bin bs=1000 count=216
perl -pi -e 's/\x0/\x1/g' seg60x60x60.bin

if [ $# = 0 ]; then
   options="mt fast log logfast mtatomic logatomic"
else
   options=$*
fi

for opt in $options
do
  echo "compile MCX with $opt"

  cd "$RT/../../src/"
  make $opt
  cd $RT

  echo 'run MCX with various threads and photon numbers'

  nthread="128 256 512 1024 1280 1536 1792"
#  nthread="512 640 1024 1280 1536 1792"
  nphoton="10000 100000 1000000"
#  nphoton="1000000"
  mcxbin="../../bin/mcextreme"

  for th in $nthread
  do 
    for np in $nphoton
    do
         echo "<mcx_session thread='$th' photon='$np'>"
         echo "<cmd>$mcxbin -t $th -n $np -g 10 -f benchcpeed.inp -s speed -a 0 -b 0 -p 1</cmd>"
         echo "<output>"
         $mcxbin -t $th -n $np -g 10 -f benchcpeed.inp -s speed -a 0 -b 0 -p 1
         echo "</output>"
         echo "</mcx_session>"
    done
  done
done
