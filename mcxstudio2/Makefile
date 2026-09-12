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
test: test-doc test-bind

# Layer one links mcxdoc alone -- no LCL, no widgetset, no display -- so it
# runs anywhere, including CI without an X server.
test-doc:
	fpc -Mobjfpc -Fu. -FEtests -otests/mcxdoctest tests/mcxdoctest.lpr
	./tests/mcxdoctest ../example

# Layer two needs a widgetset, because it drives the real form: it proves the
# binding table and the designer agree, and that opening a file and writing it
# straight back changes nothing mcx would notice.
test-bind: all
	./bin/$(BINARY) --self-test ../example

# What mcx's parser actually reads, against what the form binds.  A key mcx has
# grown that nothing here reaches is how the old GUI fell years behind.
check-paths:
	@grep -oE 'FIND_JSON_(KEY|OBJ)\("[A-Za-z0-9_]+", *"[A-Za-z0-9_.]+"' \
	    ../src/mcx_utils.c | sed 's/.*, *"//;s/"$$//' | sort -u > /tmp/mcx-paths
	@grep -oE "Path:'[A-Za-z0-9_.@]+'" mcxdoc.pas \
	    | sed "s/Path:'//;s/'$$//" | grep -v '^@' | sort -u > /tmp/mcx-bound
	@echo "read by mcx but not bound by the form:"
	@comm -23 /tmp/mcx-paths /tmp/mcx-bound | sed 's/^/  /'

clean:
	rm -rf lib bin tests/*.o tests/*.ppu tests/mcxdoctest

distclean: clean
	rm -rf $(PCP) icons.png
