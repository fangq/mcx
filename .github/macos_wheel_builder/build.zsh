#!/bin/zsh

CPYTHON_VERSIONS=("3.7" "3.8" "3.9" "3.10" "3.11")
cd mcx/pmcx/
for PY_VER in ${CPYTHON_VERSIONS[@]}; do
	/usr/local/opt/python@$PY_VER/bin/pip$PY_VER wheel . -w dist/
done
/usr/local/opt/pypy3/bin/pip_pypy3 wheel . -w dist/
exit
