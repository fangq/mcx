#!/bin/bash
#
# testbuiltin.sh - run the example simulations exported under test/builtin/ and compare
#                  their absorbed fraction against the golden values
#
# usage: ./testbuiltin.sh [-u] [extra simulator parameters, such as -G 1]
#
# test/builtin/<folder>_<example>[_k].json are self-contained inputs (volume or mesh embedded)
# exported from the example scripts and the mcxlab/mmclab demos. Each is run with a fixed seed
# and NPHOTON photons; the absorbed fraction(s) printed by the simulator are compared with
# test/builtin/golden.txt within 5 binomial standard errors plus a relative tolerance RELTOL,
# which also covers float differences between devices and between mcx and mcx-cl. -u writes
# golden.txt from the current binary instead of comparing.
#
# environment variables:
#    BIN:      simulator binary (default ../bin/mcx; ../bin/mcxcl or ../bin/mmc also work)
#    BUILTIN:  folder of the exported inputs (default ./builtin)
#    GOLDEN:   golden value file (default $BUILTIN/golden.txt)
#    NPHOTON:  photon number of each run (default 1e6)
#    SEED:     RNG seed (default 1648335518)
#    RELTOL:   relative tolerance of the absorbed fraction (default 0.02)
#    TIMEOUT:  seconds allowed for each run (default 300)

TESTDIR=$(cd "$(dirname "$0")" && pwd)
BIN=${BIN:-$TESTDIR/../bin/mcx}
BUILTIN=${BUILTIN:-$TESTDIR/builtin}
GOLDEN=${GOLDEN:-$BUILTIN/golden.txt}
NPHOTON=${NPHOTON:-1e6}
SEED=${SEED:-1648335518}
RELTOL=${RELTOL:-0.02}
TIMEOUT=${TIMEOUT:-300}
UPDATE=0
if [ "$1" = "-u" ]; then
    UPDATE=1
    shift
fi
PARAM="$@"

case "$BIN" in
    */*) BIN=$(cd "$(dirname "$BIN")" && pwd)/$(basename "$BIN") ;;
esac

if [ ! -x "$BIN" ]; then
    echo "simulator binary $BIN is not found"
    exit 1
fi

if ! ls "$BUILTIN"/*.json > /dev/null 2>&1; then
    echo "no exported inputs under $BUILTIN, skip the built-in example tests"
    exit 0
fi

WORKDIR=$(mktemp -d)
trap 'rm -rf "$WORKDIR"' EXIT

if [ "$UPDATE" = "1" ]; then
    printf "# name nphoton absorbed(%%)[,absorbed per source/run...] detected\n" > "$GOLDEN"
fi

fail=0
pass=0
skip=0

for json in "$BUILTIN"/*.json; do
    name=$(basename "$json" .json)
    log=$(cd "$WORKDIR" && timeout "$TIMEOUT" "$BIN" -f "$json" -n "$NPHOTON" -E "$SEED" -s "$name" $PARAM 2>&1 | sed 's/\x1b\[[0-9;]*m//g')
    absorbed=$(echo "$log" | grep -o -E "absorbed: *[0-9.]+%" | grep -o -E "[0-9.]+" | paste -sd, -)
    detected=$(echo "$log" | grep -o -E "detected [0-9]+ photons" | head -1 | grep -o -E "[0-9]+")
    detected=${detected:-0}
    rm -f "$WORKDIR"/*

    if [ -z "$absorbed" ]; then
        reason=$(echo "$log" | grep -E "ERROR|error" | head -1)
        echo "$name: no absorbed fraction reported ${reason:+($reason)}"
        if [ "$UPDATE" = "1" ]; then
            printf "%s %s none %s\n" "$name" "$NPHOTON" "$detected" >> "$GOLDEN"
        else
            fail=$((fail + 1))
        fi
        continue
    fi

    if [ "$UPDATE" = "1" ]; then
        printf "%s %s %s %s\n" "$name" "$NPHOTON" "$absorbed" "$detected" >> "$GOLDEN"
        echo "$name: absorbed $absorbed%, detected $detected"
        continue
    fi

    ref=$(awk -v n="$name" '$1==n {print $3; exit}' "$GOLDEN")
    if [ -z "$ref" ] || [ "$ref" = "none" ]; then
        echo "$name: absorbed $absorbed% (no golden value, skipped)"
        skip=$((skip + 1))
        continue
    fi

    verdict=$(python3 -c "
import math, sys
cur = [float(x) for x in sys.argv[1].split(',')]
ref = [float(x) for x in sys.argv[2].split(',')]
n, rtol = float(sys.argv[3]), float(sys.argv[4])
if len(cur) != len(ref):
    print('FAIL %d vs %d absorbed values' % (len(cur), len(ref))); sys.exit()
bad = [(c, r) for c, r in zip(cur, ref)
       if abs(c - r) > 5 * math.sqrt(max(r, 1e-3) * max(100 - r, 1e-3) / n) + rtol * r]
print('FAIL ' + ' '.join('%g vs %g' % b for b in bad) if bad else 'ok')
" "$absorbed" "$ref" "$NPHOTON" "$RELTOL")

    if [ "$verdict" = "ok" ]; then
        echo "$name: absorbed $absorbed% vs golden $ref% ok"
        pass=$((pass + 1))
    else
        echo "$name: absorbed $absorbed% vs golden $ref% $verdict"
        fail=$((fail + 1))
    fi
done

if [ "$UPDATE" = "1" ]; then
    echo "golden values written to $GOLDEN"
    exit 0
fi

echo "passed $pass, failed $fail, skipped $skip built-in example tests"
[ "$fail" -eq 0 ]
