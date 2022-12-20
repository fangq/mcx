#!/bin/bash
cd pmcx/
for PYBIN in /opt/python/*/bin/; do
    "${PYBIN}/pip" wheel . -w wheels/
done
rm -rf dist/
for WHEEL in wheels/*; do
    auditwheel repair ${WHEEL} -w dist/
done
rm -rf wheels/
