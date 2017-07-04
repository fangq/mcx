#!/bin/sh
############################################################
#   MCX Monte Carlo Photon Simulator Packaging System
#
#  mcxrpmcopy.sh: script to copy rpm source files to the 
#                 packaging folder
#
#  Author: Qianqian Fang <q.fang at neu.edu>
#
############################################################ 

if [ $# -ne 2 ]; then
     echo 1>&2 Usage: $0 package-name version
     exit 2
fi

PKGNAME=$1
VERSION=$2

BINDIR=rpmroot/${PKGNAME}/
SPECFILE=rpmroot/${PKGNAME}/${PKGNAME}.spec

if [ ! -d $BINDIR ]; then
     echo 1>&2 $BINDIR does not exist, please run mcxrpmmkdir.sh first
     exit 2
fi

cp -a src $BINDIR
cp -a example $BINDIR
cp -a doc $BINDIR
cp -a utils $BINDIR
cp -a AUTHORS* $BINDIR
cp -a ChangeLog* $BINDIR
cp -a INSTALL* $BINDIR
cp -a README* $BINDIR
cp -a Makefile $BINDIR
cp -a $PKGNAME.* $BINDIR
cp -a LICENSE* $BINDIR

[ -d package/rpmsrc ] && cp -a package/rpmsrc/${PKGNAME}.spec rpmroot/${PKGNAME}

awk '/I.  Introduction/{dop=1;} /^$/{if(dop>0) dop++;} /./{if(dop==2) print " " $0;}' README* >> debian/DEBIAN/control
cat package/rpmsrc/${PKGNAME}.spec | sed -e '/%INFO%/{r pkg.info' -e 'd;}' > $SPECFILE

sed -i "s/%NAME%/$PKGNAME/g"  $SPECFILE
sed -i "s/%VERSION%/$VERSION/g"  $SPECFILE

