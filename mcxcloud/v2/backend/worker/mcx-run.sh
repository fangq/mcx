#!/bin/bash
# MCX Cloud v2 worker entrypoint. Runs inside the mcx container as the swarm service
# command (passed via `docker service create ... /bin/bash -c "<this>"`). It fetches the
# input from the API, runs mcx, and pushes results back over HTTP — so the manager never
# has to discover output via NFS (eliminates the v1 ~60 s attribute-cache lag).
#
# Env (set by the scheduler): API_URL, JOB_ID, WORKER_SECRET, SEEDFLAG.
# CUDA_VISIBLE_DEVICES is taken from the swarm-assigned generic resource.
set -uo pipefail

: "${API_URL:?}"; : "${JOB_ID:?}"; : "${WORKER_SECRET:?}"
H="x-worker-secret: ${WORKER_SECRET}"
BASE="${API_URL}/jobs/${JOB_ID}"
cd /tmp || exit 1
start=$(date +%s)

fail() {
  curl -fsS -X POST -H "$H" -H 'content-type: text/plain' \
    --data-binary @output.log "${BASE}/complete?error=1" 2>/dev/null || true
  exit 1
}

# 1) fetch the (reassembled) MCX input
if ! curl -fsS -H "$H" "${BASE}/input" -o input.json; then
  echo 'failed to fetch input' > output.log
  fail
fi

# 2) run the simulation on the swarm-assigned GPU
export CUDA_VISIBLE_DEVICES="${DOCKER_RESOURCE_NVIDIA_GPU:-0}"
if ! mcx -f input.json -s output -F jnii --log ${SEEDFLAG:-} > output.log 2>&1; then
  fail
fi

# 3) push outputs (raw bytes; already JData/JNIfTI JSON)
curl -fsS -H "$H" -H 'content-type: application/octet-stream' \
  --data-binary @output.jnii "${BASE}/output" || fail

detp="$(ls output_detp.jdt output_detp.jdat output.jdt 2>/dev/null | head -1 || true)"
if [ -n "${detp}" ] && [ -f "${detp}" ]; then
  curl -fsS -H "$H" -H 'content-type: application/octet-stream' \
    --data-binary "@${detp}" "${BASE}/detphoton" || true
fi

# 4) finalize (body = log, so the client sees the mcx run log)
runtime=$(( $(date +%s) - start ))
curl -fsS -X POST -H "$H" -H 'content-type: text/plain' \
  --data-binary @output.log "${BASE}/complete?runtime=${runtime}" || true
