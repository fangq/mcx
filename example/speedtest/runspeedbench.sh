#!/bin/sh

RT=`pwd`

if [ $# = 0 ]; then
   options="log"
else
   options=$*
fi

for opt in $options
do
  echo "compile MCX with $opt"

  cd "$RT/../../src/"
  rm -rf "$RT/../../bin/*"
  make clean $opt
  cd $RT

  echo 'run MCX with various threads and photon numbers'

  nthread="128 256 512 1024 1280 1536 1792"
  nphoton="10000 100000 1000000"

### use the following setting for dedicated GPU
#  nthread="1024 1792 2048 4096 5120 6144 7168 8192"
#  nphoton="1000000"

  mcxbin=`ls ../../bin/mcx*`

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
