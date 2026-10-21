#!/bin/sh
#
# testtraj.sh - step-by-step photon trajectory tests
#
# usage: ./testtraj.sh [-u] [extra mcx parameters]
#    -u: regenerate the golden trajectories under golden/
#
# each case under traj_cases/ is simulated with a fixed seed and one photon
# per GPU thread (-A 0 -t <photons> -T 1), so that every photon draws from its
# own reproducible random number stream and the saved trajectory (-D M) is
# deterministic; it is then checked against physical invariants and compared
# to the solver-neutral golden file golden/<case>.json. MCX-CL derives the
# per-thread seeds the same way, so the same golden data apply to both.
#
# environment variables:
#    MCX:    path to the mcx binary (default ../bin/mcx)
#    PYTHON: python interpreter with numpy (default python3)
#    REF:    an extra golden folder from another solver (mcxcl, mmc) to
#            compare against, using REFTOL as the position tolerance and
#            REFMATCH as the minimum fraction of matching steps

MCX=${MCX:-$(dirname "$0")/../bin/mcx}
PYTHON=${PYTHON:-python3}
REFTOL=${REFTOL:-1e-3}
REFMATCH=${REFMATCH:-0.8}
UPDATE=0

if [ "$1" = "-u" ]; then
    UPDATE=1
    shift
fi

PARAM="$@"
TESTDIR=$(cd "$(dirname "$0")" && pwd)

case "$MCX" in
    */*) MCX=$(cd "$(dirname "$MCX")" && pwd)/$(basename "$MCX") ;;
esac
WORKDIR=$(mktemp -d)
trap 'rm -rf "$WORKDIR"' EXIT

if ! "$MCX" -L 2>&1 | grep -q 'Device [0-9]'; then
    echo "no GPU device is found, skip trajectory tests"
    exit 0
fi

fail=0

for casefile in "$TESTDIR"/traj_cases/*.json; do
    case=$(basename "$casefile" .json)
    echo "test trajectory of case '$case' ... "

    nphoton=$("$PYTHON" -c "import json,sys; print(int(json.load(open(sys.argv[1]))['Session']['Photons']))" "$casefile")
    (cd "$WORKDIR" && "$MCX" -f "$casefile" -A 0 -t "$nphoton" -T 1 -D M -F jnii -s "$case" $PARAM > "$case.log" 2>&1)

    if [ ! -f "$WORKDIR/${case}_traj.jdt" ]; then
        echo "fail to save trajectory, see log below"
        tail -5 "$WORKDIR/$case.log"
        fail=$((fail + 1))
        continue
    fi

    golden="$TESTDIR/golden/$case.json"

    if [ "$UPDATE" = "1" ] || [ ! -f "$golden" ]; then
        mkdir -p "$TESTDIR/golden"
        "$PYTHON" "$TESTDIR/trajtest.py" "$WORKDIR/${case}_traj.jdt" --case "$casefile" --save "$golden" --solver mcx || fail=$((fail + 1))
    else
        "$PYTHON" "$TESTDIR/trajtest.py" "$WORKDIR/${case}_traj.jdt" --case "$casefile" --ref "$golden" || fail=$((fail + 1))
    fi

    if [ -n "$REF" ] && [ -f "$REF/$case.json" ]; then
        "$PYTHON" "$TESTDIR/trajtest.py" "$WORKDIR/${case}_traj.jdt" --case "$casefile" --ref "$REF/$case.json" --tol "$REFTOL" --wtol "$REFTOL" --min-match "$REFMATCH" || fail=$((fail + 1))
    fi
done

if [ "$fail" -gt 0 ]; then
    echo "failed $fail trajectory tests"
    exit $fail
fi

echo "passed all trajectory tests!"
