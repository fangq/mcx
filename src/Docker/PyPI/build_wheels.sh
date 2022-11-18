#!/bin/bash

for PYBIN in /opt/python/*/bin/; do
    "${PYBIN}/pip" wheel . -w wheels/
    auditwheel repair wheels/pymcx*whl -w wheelhouse/
    rm -rf wheels/
done
