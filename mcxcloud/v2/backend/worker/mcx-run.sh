#!/bin/bash
# MCX Cloud v2 worker entrypoint. Runs inside the mcx container as the swarm service
# command (passed via `docker service create ... /bin/bash -c "<this>"`). It fetches the
# input from the API, runs mcx, and pushes results back over HTTP — so the manager never
# has to discover output via NFS (eliminates the v1 ~60 s attribute-cache lag).
#
# Uses `wget` (GNU wget, present in fangqq/mcx) rather than curl, so no augmented worker
# image is required. GNU wget's --method/--body-file give the PUT uploads the API expects.
#
# Env (set by the scheduler): API_URL, JOB_ID, WORKER_SECRET, SEEDFLAG.
# CUDA_VISIBLE_DEVICES is taken from the swarm-assigned generic resource.
set -uo pipefail

: "${API_URL:?}"; : "${JOB_ID:?}"; : "${WORKER_SECRET:?}"
H="x-worker-secret: ${WORKER_SECRET}"
BASE="${API_URL}/jobs/${JOB_ID}"
cd /tmp || exit 1
start=$(date +%s)

# wget helpers. GET saves to a file; PUT/POST send a file body with an explicit
# content-type. wget exits non-zero on HTTP error responses (>=4xx), which drives fail().
wget_get()  { wget -q -O "$2" --header="$H" "$1"; }
wget_send() { # method url file content-type
  wget -q -O /dev/null --method="$1" --body-file="$3" \
    --header="$H" --header="content-type: $4" "$2"
}

fail() {
  wget_send POST "${BASE}/complete?error=1" output.log 'text/plain' 2>/dev/null || true
  exit 1
}

# 1) fetch the (reassembled) MCX input
if ! wget_get "${BASE}/input" input.json; then
  echo 'failed to fetch input' > output.log
  fail
fi

# 2) run the simulation on the swarm-assigned GPU
export CUDA_VISIBLE_DEVICES="${DOCKER_RESOURCE_NVIDIA_GPU:-0}"
if ! mcx -f input.json -s output -F jnii --log ${SEEDFLAG:-} > output.log 2>&1; then
  fail
fi

# 3) push outputs (raw bytes; already JData/JNIfTI JSON). Routes are PUT.
wget_send PUT "${BASE}/output" output.jnii 'application/octet-stream' || fail

detp="$(ls output_detp.jdt output_detp.jdat output.jdt 2>/dev/null | head -1 || true)"
if [ -n "${detp}" ] && [ -f "${detp}" ]; then
  wget_send PUT "${BASE}/detphoton" "${detp}" 'application/octet-stream' || true
fi

# 4) finalize (body = log, so the client sees the mcx run log)
runtime=$(( $(date +%s) - start ))
wget_send POST "${BASE}/complete?runtime=${runtime}" output.log 'text/plain' || true
