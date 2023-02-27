#!/bin/bash

###############################################################################
#
#  MCX Nightly-Build Script
#
#  by Qianqian Fang <q.fang at neu.edu>
#
#  Format:
#     ./buildmcx.sh <releasetag> <branch>
#                   releasetag defaults to "nightly" if not given
#                   branch defaults to "master" if not given
#
#  Dependency:
#   - To compile mcx binary, mcxlab for octave, and mcxstudio
#
#     sudo apt-get install gcc nvidia-cuda-toolkit lazarus liboctave-dev
#
#   - To compile mcxlab for MATLAB, one must install MATLAB first, also search
#     and replace R20xx in this script to match your system's MATLAB version
#   - To compile mcxstudio, one must install lazarus with GLScene first
#   - For MacOS, first install macport and gcc-4.9
#   - For Windows, first install Cygwin64, and install x86_64-w64-mingw32-gcc
#   
###############################################################################

BUILD='nightly';
LAZMAC=

if [ ! -z "$1" ]
then
	BUILD=$1
fi
DATE=`date +'%Y%m%d'`
BUILDROOT=~/space/autobuild/$BUILD/mcx
OS=`uname -s`
MACHINE=`uname -m`

if [[ "$OS" == "Linux" ]]
then
    OS=linux
    source ~/.bashrc
elif [[ "$OS" == "Darwin" ]]; then
    OS=osx
    source ~/.bash_profile
    #LAZMAC='--ws=cocoa'
elif [[ "$OS" == CYGWIN* ]] || [[ "$OS" == MINGW* ]] || [[ "$OS" == MSYS* ]]; then
    OS=win
fi

TAG=${OS}-${MACHINE}-${BUILD}

SERVER=
REMOTEPATH=

if [ "$BUILD" == "nightly" ]
then
	TAG=${OS}-${MACHINE}-${BUILD}build
fi

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib
export PATH=.:/opt/local/bin:/usr/local/cuda/bin/:$PATH

mkdir -p $BUILDROOT
cd $BUILDROOT

rm -rf mcx
mkdir -p mcx/mcx
git clone --recurse-submodules https://github.com/fangq/mcx.git mcx/mcx

cat <<EOF >> mcx/mcx/.git/config
[filter "rcs-keywords"]
        clean  = .git_filters/rcs-keywords.clean
        smudge = .git_filters/rcs-keywords.smudge %f
EOF

cd mcx/mcx

if [ ! -z "$2" ]
then
      git checkout $2
fi

rm -rf *
git checkout .
git submodule update --init --remote

rm -rf .git
cd ..
zip -FSr $BUILDROOT/mcx-src-${BUILD}.zip mcx
if [ "$OS" == "linux" ] && [ ! -z "$SERVER" ]
then
	scp $BUILDROOT/mcx-src-${BUILD}.zip $SERVER:$REMOTEPATH/src
fi
cd mcx
cd src

rm -rf ../mcxlab/AUTO_BUILD_*
make clean
make mex &> ../mcxlab/AUTO_BUILD_${DATE}.log

if [ "$OS" == "linux" ]
then

g++ -c  -I/usr/local/cuda/include -I/usr/local/MATLAB/R2010b/extern/include \
-DMATLAB_MEX_FILE -ansi -D_GNU_SOURCE -fPIC -fno-omit-frame-pointer -pthread \
-DSAVE_DETECTORS -DUSE_CACHEBOX -DMCX_CONTAINER -fopenmp   -DMX_COMPAT_32 \
-O -DNDEBUG  "mcxlab.cpp" -o ../mcxlab/mcxlab.o

g++ -O -pthread -shared -Wl,--version-script,/usr/local/MATLAB/R2010b/extern/lib/glnxa64/mexFunction.map \
-Wl,--no-undefined -fopenmp  -o  "../mcxlab/mcx.mexa64"   "mcx_core.o"  "mcx_bench.o" \
"mcx_utils.o"  "mcx_shapes.o"  "mcx_tictoc.o" "mcx_mie.o" "mcx.o"  "cjson/cJSON.o" libzmat.a "ubj/ubjw.o" \
../mcxlab/mcxlab.o  -L/usr/local/cuda/lib64 -lcudadevrt -lcudart_static -ldl \
-lrt -Wl,-Bstatic -lm -static-libgcc -static-libstdc++ -Wl,-Bdynamic -lz \
-Wl,-rpath-link,/usr/local/MATLAB/R2010b/bin/glnxa64 \
-L/usr/local/MATLAB/R2010b/bin/glnxa64 -lmx -lmex -lmat \
-L/usr/local/MATLAB/R2010b/sys/os/glnxa64

elif [ "$OS" == "osx" ]; then

DYLD_LIBRARY_PATH="" g++ -c -I/usr/local/cuda/include -I/Applications/MATLAB_R2016a.app/extern/include \
-I/Applications/MATLAB_R2016a.app/simulink/include -DMATLAB_MEX_FILE -fno-common -fexceptions \
-m64 -isysroot /Developer/SDKs/MacOSX10.6.sdk -mmacosx-version-min=10.5 -I/usr/include \
-I/usr/clang-ide/lib/c++/v1 -DSAVE_DETECTORS -DUSE_CACHEBOX -DMCX_CONTAINER -DMX_COMPAT_32 \
-O2 -DNDEBUG "mcxlab.cpp" -o ../mcxlab/mcxlab.o

DYLD_LIBRARY_PATH="" g++ -static-libgcc -static-libstdc++ -O -Wl,-twolevel_namespace -undefined error -m64 \
-Wl,-syslibroot,/Developer/SDKs/MacOSX10.6.sdk -mmacosx-version-min=10.5 \
-bundle -Wl,-exported_symbols_list,/Applications/MATLAB_R2016a.app/extern/lib/maci64/mexFunction.map \
-L/Applications/MATLAB_R2016a.app/bin/maci64 -lmx -lmex -L/usr/lib -fopenmp \
-o "../mcxlab/mcx.mexmaci64" "mcx_core.o" "mcx_utils.o" "mcx_shapes.o" "mcx_tictoc.o" "mcx_mie.o" "mcx_bench.o" libzmat.a \
"mcx.o" "cjson/cJSON.o" "ubj/ubjw.o" ../mcxlab/mcxlab.o /opt/local/lib/gcc49/libgomp.a \
-L/usr/local/lib/ -L/usr/local/cuda/lib -lcudadevrt -lcudart_static -ldl -lm -lz \
-L/Applications/MATLAB_R2016a.app/bin/maci64 -lmx -lmex

elif [ "$OS" == "win" ]; then
    cmd /c mex mcx_core.obj mcx_utils.obj mcx_shapes.obj mcx_tictoc.obj mcx_mie.obj mcx_bench.obj cjson/cJSON.obj -output ../mcxlab/mcx -L"E:\Applications\CUDA7.5\CUDA7.5/lib/x64" -lcudadevrt -lcudart_static  CXXFLAGS='$CXXFLAGS -g -DSAVE_DETECTORS -DUSE_CACHEBOX -DMCX_CONTAINER /openmp  ' LDFLAGS='-L$TMW_ROOT$MATLABROOT/sys/os/$ARCH $LDFLAGS /openmp ' mcxlab.cpp -outdir ../mcxlab -I/usr/local/cuda/include -I"E:\Applications\CUDA7.5\CUDA7.5/lib/include" -DUSE_XORSHIFT128P_RAND -lzlib
    echo "Windows mcx build"
    cd ../mcxlab
    upx -9 mcx.mexw64
    cd ../src
fi

make clean
make oct  >>  ../mcxlab/AUTO_BUILD_${DATE}.log 2>&1

if [ "$OS" == "linux" ]
then
	make oct BACKEND=cudastatic >>  ../mcxlab/AUTO_BUILD_${DATE}.log 2>&1
fi

mexfile=(../mcxlab/mcx.mex*)

if [ -e "${mexfile[0]}" ]
then
        echo "Build Successfully" >> ../mcxlab/AUTO_BUILD_${DATE}.log
else
        echo "Build Failed" >> ../mcxlab/AUTO_BUILD_${DATE}.log
fi

if [ "$BUILD" != "nightly" ]
then
	rm -rf ../mcxlab/AUTO_BUILD_${DATE}.log
fi

rm -rf ../mcxlab/mcxlab.o ../mcxlab/mcxlab.obj

cd ../filter/src
make clean
make BACKEND=cudastatic

mexfile=(../bin/mcxfilter.mex*)

if [ -e "${mexfile[0]}" ]
then
        echo "Filter Build Successfully" >> ../mcxlab/AUTO_BUILD_${DATE}.log
	cp ../bin/mcxfilter.mex* ../../mcxlab/
	cp mcxfilter.m ../../mcxlab/
	mkdir ../../mcxlab/filter
	cp -a ../demos  ../../mcxlab/filter
	cp -a ../Wave3D ../../mcxlab/filter
else
        echo "Filter Build Failed" >> ../mcxlab/AUTO_BUILD_${DATE}.log
fi

cd ../

cp $BUILDROOT/dlls/zlib.dll ../mcxlab
cd ..
zip -FSr $BUILDROOT/mcxlab-${TAG}.zip mcxlab
cd src

[ ! -z "$SERVER" ] && scp $BUILDROOT/mcxlab-${TAG}.zip $SERVER:$REMOTEPATH/${OS}64/

make clean


if [ "$OS" == "linux" ]
then
    make AR=g++ BACKEND=cudastatic USERLINKOPT='libzmat.a -lz -Wl,-Bstatic -lgomp -Wl,-Bdynamic' &> $BUILDROOT/mcx_buildlog_${DATE}.log
elif [ "$OS" == "osx" ]; then
    make BACKEND=cudastatic  &> $BUILDROOT/mcx_buildlog_${DATE}.log
else
    make static &> $BUILDROOT/mcx_buildlog_${DATE}.log
fi

if [ -f "../bin/mcx" ]
then
	echo "Build Successfully" >> $BUILDROOT/mcx_buildlog_${DATE}.log
else
	echo "Build Failed" >> $BUILDROOT/mcx_buildlog_${DATE}.log
	exit 1;
fi

cd ../mcxstudio
lazbuild --build-mode=release ${LAZMAC} mcxshow.lpi
lazbuild --build-mode=release ${LAZMAC} mcxviewer.lpi
lazbuild --build-mode=release ${LAZMAC} mcxstudio.lpi
cp debug/mcxstudio ../bin
cp mcxshow ../bin
cp mcxviewer ../bin
cp README.txt ../inno/MCXStudio_README.txt

if [ "$OS" == "osx" ]
then
	cp -a debug/mcxstudio.app ../bin
	cp -a mcxshow.app   ../bin
	cp -a mcxviewer.app ../bin

	cat <<EOF > ../MAC_USER_PLEASE_RUN_THIS_FIRST.sh
#/bin/sh
xattr -dr com.apple.quarantine *
EOF
	chmod +x MAC_USER_PLEASE_RUN_THIS_FIRST.sh
fi

cd ../bin

cp $BUILDROOT/bindlls/*.dll .

if [ "$OS" == "win" ]
then
     upx -9 *.exe
     rm -rf mcx.exp mcx.lib
elif [ "$OS" == "linux" ]; then
     upx -9 mcx*
else
     echo "no compression on Mac"
fi

cd ../
rm -rf .git mcxlab vsproj nsight mcxstudio src Makefile package icons genlog.sh .git* deploy icons filter
if [ "$OS" != "win" ]
then
    rm -rf setup
else
    find . -type f -name "*.txt" -o -name "*.sh" -o -name "*.inp" -o -name "*.m" -o -name "*.json" | xargs unix2dos
fi
cd ../

mv $BUILDROOT/mcx_buildlog_${DATE}.log mcx/AUTO_BUILD_${DATE}.log

if [ "$BUILD" != "nightly" ]
then
	rm -rf mcx/AUTO_BUILD_${DATE}.log
fi


if [ "$OS" == "win" ]
then
   zip -FSr mcx-${TAG}.zip mcx
else
   zip -FSry mcx-${TAG}.zip mcx
fi

mv mcx-${TAG}.zip $BUILDROOT

cd $BUILDROOT

[ ! -z "$SERVER" ] && scp mcx-${TAG}.zip ${SERVER}:${REMOTEPATH}/${OS}64/
