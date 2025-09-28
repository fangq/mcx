all:
	find . -name '*.po' -exec sh -c 'msgfmt "$$0" -o ../locale/`basename $$0 .po`.mo' '{}' \;
