#!/bin/bash

for PYBIN in /opt/python/*/bin/; do
    "${PYBIN}/pip" wheel . -w wheels/
done
for WHEEL in wheels/*; do
    auditwheel repair ${WHEEL} -w wheelhouse/
done
rm -rf wheels/
