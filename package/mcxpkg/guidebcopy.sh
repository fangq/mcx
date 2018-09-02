#!/bin/sh
############################################################
#   MCX Monte Carlo Photon Simulator Packaging System
#
#  mcxdebcopy.sh: script to copy files to the packaging folder structure
#
#  Author: Qianqian Fang <q.fang at neu.edu>
#
############################################################ 

if [ $# -lt 2 ]; then
     echo 1>&2 Usage: $0 package-name version
     exit 2
fi

PKGNAME=$1
VERSION=$2

#INFO=`awk '/I. About this font/{dop=1;} /^$/{if(dop>0) dop++;} /./{if(dop==2) print " " $0;}' README*`

BINDIR=debian/usr/lib/$PKGNAME
DOCDIR=debian/usr/share/doc/$PKGNAME
MENUDIR=debian/usr/share/applications
ICONDIR=debian/usr/share/pixmaps

if [ ! -d $BINDIR ]; then
     echo 1>&2 $BINDIR does not exist, please run mcxdebmkdir.sh first
     exit 2
fi

cp -a AUTHORS* $DOCDIR
cp -a ChangeLog* $DOCDIR
cp -a $PKGNAME/README* $DOCDIR
cp -a example/*.mcxp $DOCDIR

cp -a $PKGNAME/debug/$PKGNAME $BINDIR

for fn in LICENSE*; do
   cp -a LICENSE* $BINDIR; break;
done

#for fn in mcxstudio/*.desktop; do
#   cp -a mcxstudio/*.desktop $MENUDIR; break;
#done

if [ -d pixmap ]; then
    mkdir -p $ICONDIR
    cp -a pixmap/* $ICONDIR
fi

[ -d package/mexdebsrc ] && cp -a package/mexdebsrc/* debian/DEBIAN/
chmod g-s -R debian
sed -i "s/%NAME%/$PKGNAME/g"  debian/DEBIAN/*
sed -i "s/%VERSION%/$VERSION/g"  debian/DEBIAN/*

#sed -i "s/%INFO%/$INFO/g"  debian/DEBIAN/control
#sed -i "s/^&/\n/g"  debian/DEBIAN/control
awk '/I.  Introduction/{dop=1;} /^$/{if(dop>0) dop++;} /./{if(dop==2) print " " $0;}' README* >> debian/DEBIAN/control

# install .mo files
if [ -d i18n ]; then
    for lang in `ls -1 i18n | sed -e 's/^i18n\///g'`
    do
	if [ -f i18n/$lang/*.mo ]; then
		mkdir -p $I18NDIR/$lang/LC_MESSAGES/; 
		cp -a i18n/$lang/*.mo $I18NDIR/$lang/LC_MESSAGES/
	fi
    done
fi
