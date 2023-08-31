BINARY=mcxstudio

$(BINARY):
	lazbuild --build-mode=release $(BINARY).lpi
mcxshow:
	lazbuild --build-mode=release mcxshow.lpi
mcxviewer:
	lazbuild --build-mode=release mcxviewer.lpi
clean:
	rm -rf debug/$(BINARY)

