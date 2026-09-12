# mcxstudio2 - a graphical front end for MCX, MMC and MCX-CL.
#
# lazbuild wants to compile the Lazarus packages we depend on into their own
# install directories, which under a distro package live in /usr and are not
# writable.  Giving it a project-local primary config path redirects that
# output here instead, so a plain "make" works for an ordinary user and in CI
# without touching the system Lazarus or the developer's own IDE config.

BINARY    ?= mcxstudio2
WIDGETSET ?= gtk2
PCP       ?= .lazpcp
LAZBUILD  ?= lazbuild
LAZFLAGS   = --pcp=$(PCP) --ws=$(WIDGETSET)

.PHONY: all debug icons test clean distclean res

all:
	$(LAZBUILD) $(LAZFLAGS) --build-mode=Release $(BINARY).lpi

debug:
	$(LAZBUILD) $(LAZFLAGS) --build-mode=Debug $(BINARY).lpi

# Regenerates the Windows manifest resource.  Committed, so a fresh clone
# builds without fpcres on PATH.
res:
	cd packaging/windows && fpcres -of res -o ../../$(BINARY).res $(BINARY).rc

# A contact sheet of every toolbar glyph, for reviewing the artwork.
icons: all
	./bin/$(BINARY) --dump-icons icons.png

# The document tests link mcxdoc alone -- no LCL, no widgetset, no display --
# so they run anywhere, including CI without an X server.
test:
	fpc -Mobjfpc -Fu. -FEtests -otests/mcxdoctest tests/mcxdoctest.lpr
	./tests/mcxdoctest ../example

clean:
	rm -rf lib bin tests/*.o tests/*.ppu tests/mcxdoctest

distclean: clean
	rm -rf $(PCP) icons.png
