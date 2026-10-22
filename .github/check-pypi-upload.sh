#!/bin/bash
# set perform_pypi_upload=1 only when PyPI answers and lacks at least one of the files in dist/;
# if PyPI cannot be queried, do not upload. Files are compared, not the version: the platform
# workflows finish at different times, and the first upload of a new version must not make the
# others skip theirs (the publish step uses skip_existing, so re-sent files are ignored)
PKG=pmcx
BUILD_VERSION=$(ls dist/ | head -1 | awk -F"-" '{ print $2 }')
UPLOAD_TO_PYPI=0
if [ -z "$BUILD_VERSION" ]; then
  echo "No package found in dist/, not uploading."
elif RELEASES=$(curl -sfL --retry 3 "https://pypi.org/pypi/$PKG/json"); then
  for f in dist/*; do
    if grep -q "\"filename\": *\"$(basename "$f")\"" <<< "$RELEASES"; then
      echo "$(basename "$f") is already on PyPI."
    else
      echo "$(basename "$f") is not on PyPI yet."
      UPLOAD_TO_PYPI=1
    fi
  done
else
  echo "Could not query PyPI for $PKG, not uploading."
fi
echo "perform_pypi_upload=$UPLOAD_TO_PYPI" >> $GITHUB_OUTPUT
