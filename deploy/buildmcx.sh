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

## setting up environment

BUILD='nightly'
LAZMAC=

if [ ! -z "$1" ]; then
	BUILD=$1
fi
DATE=$(date +'%Y%m%d')
BUILDROOT=~/space/autobuild/$BUILD/mcx
OS=$(uname -s)
MACHINE=$(uname -m)

if [[ $OS == "Linux" ]]; then
	OS=linux
	source ~/.bashrc
elif [[ $OS == "Darwin" ]]; then
	OS=macos
	source ~/.bash_profile
	#LAZMAC='--ws=cocoa'
elif [[ $OS == CYGWIN* ]] || [[ $OS == MINGW* ]] || [[ $OS == MSYS* ]]; then
	OS=win
fi

TAG=${OS}-${MACHINE}-${BUILD}

if [ "$BUILD" == "nightly" ]; then
	TAG=${OS}-${MACHINE}-${BUILD}build
fi

## setting up upload server (blank if no need to upload)

SERVER=
REMOTEPATH=

## checking out latest github code

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib
export PATH=.:/opt/local/bin:/usr/local/cuda/bin/:$PATH

mkdir -p $BUILDROOT
cd $BUILDROOT

rm -rf mcx
mkdir -p mcx/mcx
git clone --recurse-submodules https://github.com/fangq/mcx.git mcx/mcx

## automatically update revision/version number

cat <<EOF >>mcx/mcx/.git/config
[filter "rcs-keywords"]
        clean  = .git_filters/rcs-keywords.clean
        smudge = .git_filters/rcs-keywords.smudge %f
EOF

cd mcx/mcx

if [ ! -z "$2" ]; then
	git checkout $2
fi

rm -rf *
git checkout .
git submodule update --init --remote

## zip and upload source code package

rm -rf .git
cd ..
zip -FSr $BUILDROOT/mcx-src-${BUILD}.zip mcx
if [ "$OS" == "linux" ] && [ ! -z "$SERVER" ]; then
	scp $BUILDROOT/mcx-src-${BUILD}.zip $SERVER:$REMOTEPATH/src
fi
cd mcx
cd src

## build matlab mex file

rm -rf ../mcxlab/AUTO_BUILD_*
make clean
make mex &>../mcxlab/AUTO_BUILD_${DATE}.log

if [ "$OS" == "linux" ]; then
	make mex MEXLINKOPT="-static-libstdc++ -static-libgcc" >>../mcxlab/AUTO_BUILD_${DATE}.log 2>&1
elif [ "$OS" == "macos" ]; then
	make mex MEXLINKOPT='/usr/local/lib/libomp.a' >>../mcxlab/AUTO_BUILD_${DATE}.log 2>&1
elif [ "$OS" == "win" ]; then
	cmd /c mex mcx_core.obj mcx_utils.obj mcx_shapes.obj mcx_tictoc.obj mcx_mie.obj mcx_bench.obj cjson/cJSON.obj -output ../mcxlab/mcx -L"E:\Applications\CUDA7.5\CUDA7.5/lib/x64" -lcudadevrt -lcudart_static CXXFLAGS='$CXXFLAGS -g -DSAVE_DETECTORS -DUSE_CACHEBOX -DMCX_CONTAINER /openmp  ' LDFLAGS='-L$TMW_ROOT$MATLABROOT/sys/os/$ARCH $LDFLAGS /openmp ' mcxlab.cpp -outdir ../mcxlab -I/usr/local/cuda/include -I"E:\Applications\CUDA7.5\CUDA7.5/lib/include" -DUSE_XORSHIFT128P_RAND
fi

## build octave mex file

make clean
make oct >>../mcxlab/AUTO_BUILD_${DATE}.log 2>&1

if [ "$OS" == "linux" ]; then
	make oct BACKEND=cudastatic >>../mcxlab/AUTO_BUILD_${DATE}.log 2>&1
fi

## test mex file dependencies

mexfile=(../mcxlab/mcx.mex*)

if [ -e "${mexfile[0]}" ]; then
	if [ "$OS" == "macos" ]; then
		otool -L ../mcxlab/mcx.mex* >>../mcxlab/AUTO_BUILD_${DATE}.log
	elif [ "$OS" == "win" ]; then
		objdump -p ../mcxlab/mcx.mex* | grep -H "DLL Name:" >>../mcxlab/AUTO_BUILD_${DATE}.log
	else
		ldd ../mcxlab/mcx.mex* >>../mcxlab/AUTO_BUILD_${DATE}.log
	fi
	echo "Build Successfully" >>../mcxlab/AUTO_BUILD_${DATE}.log
else
	echo "Build Failed" >>../mcxlab/AUTO_BUILD_${DATE}.log
fi

## compress mex files with upx

upx -9 ../mcxlab/mcx.mex* || true

## zip and upload mex package

if [ "$BUILD" != "nightly" ]; then
	rm -rf ../mcxlab/AUTO_BUILD_${DATE}.log
fi

rm -rf ../mcxlab/mcxlab.o ../mcxlab/mcxlab.obj

## compile denoising filter mex for matlab

cd ../filter/src
make clean


if [ "$OS" == "osx" ]
then
    make BACKEND=cudastatic GCC=cc
else
    make BACKEND=cudastatic
fi

## zip and upload mex package

mexfile=(../bin/mcxfilter.mex*)

if [ -e "${mexfile[0]}" ]; then
	echo "Filter Build Successfully" >>../../mcxlab/AUTO_BUILD_${DATE}.log
	cp ../bin/mcxfilter.mex* ../../mcxlab/
	cp mcxfilter.m ../../mcxlab/
	mkdir ../../mcxlab/filter
	cp -a ../demos ../../mcxlab/filter
	cp -a ../Wave3D ../../mcxlab/filter
else
	echo "Filter Build Failed" >>../../mcxlab/AUTO_BUILD_${DATE}.log
fi

cd ../

cd ..
zip -FSr $BUILDROOT/mcxlab-${TAG}.zip mcxlab
cd src

[ ! -z "$SERVER" ] && scp $BUILDROOT/mcxlab-${TAG}.zip $SERVER:$REMOTEPATH/${OS}64/

## compile standalone binary/executable

make clean

if [ "$OS" == "linux" ]; then
	make AR=g++ BACKEND=cudastatic USERLINKOPT='lib/libzmat.a -Wl,-Bstatic -lgomp -Wl,-Bdynamic' &>$BUILDROOT/mcx_buildlog_${DATE}.log
elif [ "$OS" == "macos" ]; then
	make BACKEND=cudastatic &>$BUILDROOT/mcx_buildlog_${DATE}.log
else
	make static &>$BUILDROOT/mcx_buildlog_${DATE}.log
fi

## test binary dependencies

if [ -f "../bin/mcx" ]; then
	if [ "$OS" == "macos" ]; then
		otool -L ../bin/mcx >>$BUILDROOT/mcx_buildlog_${DATE}.log
	elif [ "$OS" == "win" ]; then
		objdump -p ../bin/mcx.exe | grep "DLL Name:" >>$BUILDROOT/mcx_buildlog_${DATE}.log
	else
		ldd ../bin/mcx >>$BUILDROOT/mcx_buildlog_${DATE}.log
	fi
	echo "Build Successfully" >>$BUILDROOT/mcx_buildlog_${DATE}.log
else
	echo "Build Failed" >>$BUILDROOT/mcx_buildlog_${DATE}.log
	exit 1
fi

## build mcxstudio GUI with lazarus-ide

cd ../mcxstudio
lazbuild --build-mode=release ${LAZMAC} mcxshow.lpi
lazbuild --build-mode=release ${LAZMAC} mcxviewer.lpi
lazbuild --build-mode=release ${LAZMAC} mcxstudio.lpi
cp debug/mcxstudio ../bin
cp mcxshow ../bin
cp mcxviewer ../bin
cp README.txt ../inno/MCXStudio_README.txt

## copy MacOS app files

if [ "$OS" == "macos" ]; then
	cp -a debug/mcxstudio.app ../bin
	cp -a mcxshow.app ../bin
	cp -a mcxviewer.app ../bin

	cat <<EOF >../MAC_USER_PLEASE_RUN_THIS_FIRST.sh
#/bin/sh
xattr -dr com.apple.quarantine *
EOF
	chmod +x ../MAC_USER_PLEASE_RUN_THIS_FIRST.sh
fi

cd ../bin

cp $BUILDROOT/bindlls/*.dll .

## compress binary with upx

upx -9 mcx* || true

## zip and upload binary package

if [ "$OS" == "win" ]; then
	rm -rf mcx.exp mcx.lib
fi

cd ../
rm -rf .git* .travis* inno pymcx mcxlab vsproj nsight mcxstudio src Makefile package icons genlog.sh deploy icons filter winget
if [ "$OS" != "win" ]; then
	rm -rf setup
else
	find . -type f -name "*.txt" -o -name "*.sh" -o -name "*.inp" -o -name "*.m" -o -name "*.json" | xargs unix2dos
fi
cd ../

mv $BUILDROOT/mcx_buildlog_${DATE}.log mcx/AUTO_BUILD_${DATE}.log

if [ "$BUILD" != "nightly" ]; then
	rm -rf mcx/AUTO_BUILD_${DATE}.log
fi

if [ "$OS" == "win" ]; then
	zip -FSr mcx-${TAG}.zip mcx
else
	zip -FSry mcx-${TAG}.zip mcx
fi

mv mcx-${TAG}.zip $BUILDROOT

cd $BUILDROOT

[ ! -z "$SERVER" ] && scp mcx-${TAG}.zip ${SERVER}:${REMOTEPATH}/${OS}64/
