#!/bin/sh
############################################################
#   MCX Monte Carlo Photon Simulator Packaging System
#
#  mcxrpmmkdir.sh: script to create rpm packaging folder structure
#
#  Author: Qianqian Fang <q.fang at neu.edu>
#
############################################################ 

if [ $# -ne 1 ]; then
     echo 1>&2 Usage: $0 package-name
     exit 2
fi

PKGNAME=$1

mkdir -p rpmroot/rpm/BUILD/
mkdir -p rpmroot/rpm/RPMS/
mkdir -p rpmroot/rpm/SOURCES/
mkdir -p rpmroot/rpm/SPECS/
mkdir -p rpmroot/rpm/SRPMS/

mkdir -p rpmroot/${PKGNAME}
