#!/bin/bash
# set perform_pypi_upload=1 only when PyPI answers and does not yet list the version of the
# package in dist/; if PyPI cannot be queried, do not upload (pip index was unreliable on some
# runners and its empty output used to be taken as "not published yet")
PKG=pmcx
BUILD_VERSION=$(ls dist/ | head -1 | awk -F"-" '{ print $2 }')
UPLOAD_TO_PYPI=0
if [ -z "$BUILD_VERSION" ]; then
  echo "No package found in dist/, not uploading."
elif RELEASES=$(curl -sfL --retry 3 "https://pypi.org/pypi/$PKG/json"); then
  # release versions are keys of the "releases" object, e.g. "0.7.3":[
  if grep -q "\"$BUILD_VERSION\": *\[" <<< "$RELEASES"; then
    echo "$PKG $BUILD_VERSION was found on PyPI, not uploading."
  else
    echo "$PKG $BUILD_VERSION wasn't found on PyPI."
    UPLOAD_TO_PYPI=1
  fi
else
  echo "Could not query PyPI for $PKG, not uploading."
fi
echo "perform_pypi_upload=$UPLOAD_TO_PYPI" >> $GITHUB_OUTPUT
