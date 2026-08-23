#!/bin/bash
# MCX Cloud v2 redbird worker entrypoint (CPU-only FEM diffusion forward solve).
#
# Same HTTP contract as mcx-run.sh: fetch input from the API, run, push results back.
# Differences from the MC engines:
#   - no GPU: docker.js claims CPU cores (--reserve-cpu/--limit-cpu) instead of an
#     NVIDIA_GPU generic resource, so there is no CUDA_VISIBLE_DEVICES / JIT cache here
#   - the solver is a Python script, not a binary. redbirdpy is installed in the image
#     (Dockerfile.redbird), but OUR driver is not -- the scheduler substitutes the driver
#     source into the placeholder token inside the heredoc below. Keeping the driver a
#     separate .py file means it stays runnable/testable outside the container.
#     NOTE: that token must occur EXACTLY ONCE in this file (the substitution replaces the
#     first match), so do not repeat it in comments.
#
# Env (set by the scheduler): API_URL, JOB_ID, WORKER_SECRET, ENGINE
set -uo pipefail

: "${API_URL:?}"; : "${JOB_ID:?}"; : "${WORKER_SECRET:?}"
H="x-worker-secret: ${WORKER_SECRET}"
BASE="${API_URL}/jobs/${JOB_ID}"
cd /tmp || exit 1
start=$(date +%s)

wget_get()  { wget -q -O "$2" --header="$H" "$1"; }
wget_send() { # method url file content-type
  wget -q -O /dev/null --method="$1" --body-file="$3" \
    --header="$H" --header="content-type: $4" "$2"
}

fail() {
  wget_send POST "${BASE}/complete?error=1" output.log 'text/plain' 2>/dev/null || true
  exit 1
}

# 1) fetch the (reassembled) input
if ! wget_get "${BASE}/input" input.json; then
  echo 'failed to fetch input' > output.log
  fail
fi

# 2) materialize the forward-solve driver. Quoted heredoc: no shell expansion, so the
# Python source is written verbatim.
cat > redbird_fwd.py <<'REDBIRD_FWD_PY_EOF'
@REDBIRD_FWD_PY@
REDBIRD_FWD_PY_EOF

# 3) solve
if ! python3 redbird_fwd.py > output.log 2>&1; then
  fail
fi
# belt-and-suspenders: require the output file as the real success signal
if [ ! -s output.jmsh ]; then
  echo 'redbird produced no output.jmsh' >> output.log
  fail
fi

# 4) push outputs (raw bytes; already JData/JNIfTI JSON)
wget_send PUT "${BASE}/output" output.jmsh 'application/octet-stream' || fail
if [ -s output_detp.jdat ]; then
  wget_send PUT "${BASE}/detphoton" output_detp.jdat 'application/octet-stream' || true
fi

# 5) finalize (body = log, so the client sees the solver log)
runtime=$(( $(date +%s) - start ))
wget_send POST "${BASE}/complete?runtime=${runtime}" output.log 'text/plain' || true
