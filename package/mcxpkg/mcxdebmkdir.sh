#!/bin/sh
############################################################
#   MCX Monte Carlo Photon Simulator Packaging System
#
#  mcxdebmkdir.sh: script to create deb packaging folder structure
#
#  Author: Qianqian Fang <q.fang at neu.edu>
#
############################################################ 

if [ $# -ne 1 ]; then
     echo 1>&2 Usage: $0 package-name
     exit 2
fi

PKGNAME=$1

BINDIR=debian/usr/lib/$PKGNAME
DOCDIR=debian/usr/share/doc/$PKGNAME
MENUDIR=debian/usr/share/applications
ICONDIR=debian/usr/share/pixmaps
   
mkdir -p $BINDIR
mkdir -p $DOCDIR
mkdir -p debian/DEBIAN/
mkdir -p $BINDIR

for fn in *.desktop; do
   mkdir -p $MENUDIR; break;
done

if [ -d pixmap ]; then
   mkdir -p $ICONDIR
fi

