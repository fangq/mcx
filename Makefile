############################################################
#   MCX Monte Carlo Photon Simulator Packaging System
#
#  Author: Qianqian Fang <q.fang at neu.edu>
############################################################

PKGNAME=mcx
VERSION=2018
SOURCE=src
GUI=mcxstudio

all: bin/$(PKGNAME) bin/$(GUI) mex deb rpm

bin/$(PKGNAME):
	-$(MAKE) -C $(SOURCE) static
bin/$(GUI):
	-$(MAKE) -C $(GUI)
	-$(COPY) -a $(GUI)/debug/$(GUI) bin
mex:
	-$(MAKE) -C $(SOURCE) mex
oct:
	-$(MAKE) -C $(SOURCE) oct
deb: bin/$(PKGNAME)
	-package/mcxpkg/mcxdebmkdir.sh $(PKGNAME)
	-package/mcxpkg/mcxdebcopy.sh  $(PKGNAME) $(VERSION)
	-dpkg -b debian $(PKGNAME)-$(VERSION).deb
rpm:
	-package/mcxpkg/mcxrpmmkdir.sh $(PKGNAME)
	-package/mcxpkg/mcxrpmcopy.sh  $(PKGNAME) $(VERSION)
	cd rpmroot && tar zcvf $(PKGNAME)-$(VERSION).tar.gz $(PKGNAME) ; \
	rpmbuild --define="_topdir rpmroot/rpm" -ta $(PKGNAME)-$(VERSION).tar.gz
clean:
	-$(MAKE) -C $(SOURCE) clean
	-$(MAKE) -C $(GUI) clean
	-rm -rf debian rpmroot pkg.info $(PKGNAME)-$(VERSION).deb $(PKGNAME)-$(VERSION)*.rpm

.DEFAULT_GOAL := bin/$(PKGNAME)

