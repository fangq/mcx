############################################################
#   MCX Monte Carlo Photon Simulator Packaging System
#
#  Author: Qianqian Fang <q.fang at neu.edu>
############################################################

PKGNAME=mcx
VERSION=1.9.7
SOURCE=src
GUI=mcxstudio

all: bin gui mex deb rpm

bin: bin/$(PKGNAME)
bin/$(PKGNAME):
	-$(MAKE) -C $(SOURCE) static
gui: bin/$(GUI)
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
mexdeb: oct
	-package/mcxpkg/mcxdebmkdir.sh $(PKGNAME)lab
	-package/mcxpkg/mexdebcopy.sh  $(PKGNAME)lab $(VERSION)
	-dpkg -b debian $(PKGNAME)lab-$(VERSION).deb
guideb: bin/$(GUI)
	-package/mcxpkg/mcxdebmkdir.sh $(PKGNAME)studio
	-package/mcxpkg/guidebcopy.sh  $(PKGNAME)studio $(VERSION)
	-dpkg -b debian $(PKGNAME)studio-$(VERSION).deb
rpm:
	-package/mcxpkg/mcxrpmmkdir.sh $(PKGNAME)
	-package/mcxpkg/mcxrpmcopy.sh  $(PKGNAME) $(VERSION)
	cd rpmroot && tar zcvf $(PKGNAME)-$(VERSION).tar.gz $(PKGNAME) ; \
	rpmbuild --define="_topdir rpmroot/rpm" -ta $(PKGNAME)-$(VERSION).tar.gz
clean:
	-$(MAKE) -C $(SOURCE) clean
	-$(MAKE) -C $(GUI) clean
	-rm -rf debian rpmroot pkg.info $(PKGNAME)*-$(VERSION).deb $(PKGNAME)*-$(VERSION)*.rpm

.DEFAULT_GOAL := bin/$(PKGNAME)
.PHONY: all mex deb rpm mexdeb guideb

