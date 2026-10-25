#!/bin/bash
# Builds the Linux release binaries of mcx inside the manylinux2014 builder:
#
#   docker run --rm -v $PWD:/src \
#       [-v $MATLAB_ROOT:/matlab:ro -e MATLAB_ROOT=/matlab] \
#       [-v /usr/include/octave-X.Y.Z:/octinc:ro -e OCTAVE_INCDIR=/octinc] \
#       mcx-linux-release-builder bash .github/linux_release_builder/build_mcx.sh
#
# MATLAB is the runner's (matlab-actions/setup-matlab), mounted read-only; the
# Octave headers are the runner's too.  Either may be left out to skip that
# mex file.  Every artifact is checked at the end: none may need a glibc
# symbol newer than 2.17, the oldest the image has, or the build fails.

set -e
cd /src

# git refuses a checkout owned by another user, which a mounted one is
git config --global --add safe.directory '*' 2>/dev/null || true

export PATH=/src/.github/linux_release_builder:$PATH   # the mkoctfile stand-in
chmod +x /src/.github/linux_release_builder/mkoctfile

# the check at the end must only ever see what this run built
rm -f bin/mcx bin/mcxstudio mcxlab/mcx.mex mcxlab/mcx.mexa64

cd src

if [ -n "$OCTAVE_INCDIR" ]; then
    echo "== mcxlab for Octave (headers: $OCTAVE_INCDIR)"
    make clean
    make oct BACKEND=cudastatic MEXLINKOPT="-static-libgcc -static-libstdc++"
fi

if [ -n "$MATLAB_ROOT" ]; then
    echo "== mcxlab for MATLAB ($MATLAB_ROOT)"
    make clean
    make mex MEX="$MATLAB_ROOT/bin/mex" MEXLINKOPT="-static-libstdc++ -static-libgcc -fopenmp"
    strip ../mcxlab/mcx.mex*
fi

echo "== mcx"
make clean
# --as-needed drops the shared libgomp that -fopenmp adds after the static one
make AR=g++ BACKEND=cudastatic USERLINKOPT='lib/libzmat.a -static-libstdc++ -static-libgcc -Wl,-Bstatic -lgomp -Wl,-Bdynamic -Wl,--as-needed'

if command -v lazbuild > /dev/null; then
    echo "== mcxstudio"
    (cd ../mcxstudio2 && make LAZDIR=/opt/lazarus && cp bin/mcxstudio ../bin/)
fi

cd /src
bash .github/linux_release_builder/check_glibc.sh bin/mcx bin/mcxstudio mcxlab/mcx.mex mcxlab/mcx.mexa64
