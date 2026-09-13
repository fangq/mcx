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

# Where lazbuild's own LCL lives.  It normally learns this from the config the
# distribution ships in /etc/lazarus, and a machine missing that file -- a
# partial install, or a Lazarus that has never been run -- gets
# "invalid Lazarus directory": the empty string, because there was nowhere to
# read one from.  Naming it here is what an IDE would have written into the
# config.  Left empty where no such directory is found, which is every
# platform that installs Lazarus somewhere else, and then lazbuild is asked
# nothing it was not asked before.
LAZDIR ?= $(firstword $(wildcard /usr/lib/lazarus/default /usr/share/lazarus \
                                 /usr/lib/lazarus/*))
LAZFLAGS = --pcp=$(PCP) --ws=$(WIDGETSET) \
           $(if $(LAZDIR),--lazarusdir=$(LAZDIR))

.PHONY: all debug forms icons sheet test clean distclean res install uninstall

all: forms
	$(LAZBUILD) $(LAZFLAGS) --build-mode=Release $(BINARY).lpi

debug: forms
	$(LAZBUILD) $(LAZFLAGS) --build-mode=Debug $(BINARY).lpi

# lazbuild decides what to recompile from the .pas timestamps alone, so a form
# edited on its own relinks the binary without rebuilding the resource that
# holds it -- the old layout ships and nothing says so.  Touching the unit that
# owns a newer .lfm is enough to make the dependency real.
forms:
	@for f in *.lfm; do \
	    u=$${f%.lfm}.pas; \
	    if [ -f "$$u" ] && [ "$$f" -nt "$$u" ]; then touch "$$u"; fi; \
	done

# Rebuilds the embedded resource: the manifest plus one PNG per icon.  Both
# icons/*.png and $(BINARY).res are committed, so a fresh clone builds with
# neither inkscape nor fpcres on PATH; this runs only when the artwork moves.
res:
	cd packaging/windows && fpcres -of res -o ../../$(BINARY).res $(BINARY).rc

# Rasterises icons/svg into icons/*.png and regenerates the resource script.
#
# The masters are 96 pixels rather than the 16 or 24 a toolbar asks for: the
# display scale is not known until the program runs, so mcxicons area-averages
# each one down to whatever size the image list turns out to want.
SVGDIR ?= ../icons/svg
SVGSIZE ?= 96
RASTER ?= inkscape

icons:
	@mkdir -p icons
	@sed 's/#.*//' packaging/icons.map | while read -r name svg; do \
	    [ -n "$$name" ] || continue; \
	    echo "  $$svg.svg -> icons/$$name.png"; \
	    $(RASTER) -w $(SVGSIZE) -h $(SVGSIZE) -o icons/$$name.png \
	        $(SVGDIR)/$$svg.svg >/dev/null 2>&1 || exit 1; \
	done
	@$(RASTER) -w 256 -h 256 -o icons/mcxstudio2.png ../mcxstudio/mcx.svg \
	    >/dev/null 2>&1 || true
	@{ echo '1 24 "mcxstudio2.manifest"'; \
	   echo 'MCXICONPNG 10 "../../icons/mcxstudio2.png"'; \
	   sed 's/#.*//' packaging/icons.map | while read -r name svg; do \
	       [ -n "$$name" ] || continue; \
	       echo "MCXICON_$$(echo $$name | tr a-z A-Z) 10 \"../../icons/$$name.png\""; \
	   done; } > packaging/windows/$(BINARY).rc
	@$(MAKE) res

# A contact sheet of every icon, for reviewing the artwork.
sheet: all
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

# Installs into a prefix, for a distribution package or a local ~/.local.
#
#     make install PREFIX=$HOME/.local
#
# The icon goes in as a PNG at the size it was rasterised, because the
# hicolor theme wants a size in the path and the SVG masters live in the
# source tree rather than in the package.
PREFIX  ?= /usr/local
DESTDIR ?=

install: all
	install -Dm755 bin/$(BINARY) $(DESTDIR)$(PREFIX)/bin/$(BINARY)
	install -Dm644 packaging/linux/$(BINARY).desktop \
	    $(DESTDIR)$(PREFIX)/share/applications/$(BINARY).desktop
	install -Dm644 icons/$(BINARY).png \
	    $(DESTDIR)$(PREFIX)/share/icons/hicolor/256x256/apps/$(BINARY).png
	install -Dm755 ../utils/mcxp2json.py \
	    $(DESTDIR)$(PREFIX)/bin/mcxp2json

uninstall:
	rm -f $(DESTDIR)$(PREFIX)/bin/$(BINARY)
	rm -f $(DESTDIR)$(PREFIX)/bin/mcxp2json
	rm -f $(DESTDIR)$(PREFIX)/share/applications/$(BINARY).desktop
	rm -f $(DESTDIR)$(PREFIX)/share/icons/hicolor/256x256/apps/$(BINARY).png
