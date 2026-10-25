#!/bin/bash
# check_glibc.sh file...: prints the newest GLIBC and GLIBCXX symbol versions
# each file needs, and fails if any needs a glibc newer than MAX_GLIBC (2.17,
# manylinux2014), any libstdc++ symbol at all, or any of the GCC runtime
# libraries (libstdc++, libgcc_s, libgomp) as a shared library: those are meant
# to be linked statically, as older systems may lack them or have older ones.
# A mex file (a shared object with no program interpreter) may use libgomp.so:
# the static libgomp.a is not position-independent, and MATLAB and Octave both
# load a libgomp of their own.
# Files that do not exist are skipped, so one list serves every build.

MAX_GLIBC=${MAX_GLIBC:-2.17}
fail=0

for f in "$@"; do
    [ -f "$f" ] || continue
    syms=$(objdump -T "$f" 2>/dev/null)
    glibc=$(echo "$syms" | grep -oE 'GLIBC_[0-9.]+' | sed 's/GLIBC_//' | sort -uV | tail -1)
    glibcxx=$(echo "$syms" | grep -oE 'GLIBCXX_[0-9.]+' | sed 's/GLIBCXX_//' | sort -uV | tail -1)
    needed=$(objdump -p "$f" 2>/dev/null | awk '/NEEDED/{print $2}' | tr '\n' ' ')
    status=ok
    if [ -n "$glibc" ] && [ "$(printf '%s\n%s\n' "$glibc" "$MAX_GLIBC" | sort -V | tail -1)" != "$MAX_GLIBC" ]; then
        status="FAIL: needs GLIBC_$glibc"
        fail=1
    fi
    if [ -n "$glibcxx" ]; then
        status="$status; FAIL: needs libstdc++ GLIBCXX_$glibcxx (not linked statically)"
        fail=1
    fi
    runtime='libstdc\+\+|libgcc_s|libgomp'
    readelf -l "$f" 2>/dev/null | grep -q INTERP || runtime='libstdc\+\+|libgcc_s'
    if echo "$needed" | grep -qE "$runtime"; then
        status="$status; FAIL: links a GCC runtime library dynamically"
        fail=1
    fi
    printf '%-40s GLIBC_%-6s %s\n    needs: %s\n' "$f" "${glibc:--}" "$status" "$needed"
done

exit $fail
