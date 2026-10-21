# MCX Cloud v2

Modernization of MCX Cloud. Design rationale and decisions: [`V2_DESIGN.md`](V2_DESIGN.md).
Deployment / private-cloud setup / cutover: [`DEPLOY.md`](DEPLOY.md).

v2 runs in parallel with the untouched v1 (`../v1/frontend`, `../v1/backend`) until cutover.

## Stack (decided)

- **DB:** PostgreSQL — JSONB metadata + a `blobs` `bytea` content-addressed table.
- **Backend:** Node.js (≥20 LTS) + Fastify, **plain ESM JS + JSDoc, no build**;
  queue = **pg-boss** (in Postgres, no Redis); transport = **REST + SSE**.
- **Frontend:** no-build native ES modules + import maps; three.js ported.
- **Orchestration:** Docker Swarm kept, driven by an event queue (no 20 s cron).

## Layout

```
v2/
  V2_DESIGN.md            # full design & decisions
  README.md
  schema/                 # shared by frontend editor + backend validator
    mcx-input.v1.json     # authoritative MCX input JSON Schema
    README.md
  contracts/              # Phase 1 — the data/API contracts everything else depends on
    openapi.yaml          # REST API (OpenAPI 3.1)
    sse-events.md         # SSE job-progress message contract
    normalization.md      # _DataLink_ content-addressed normalization & reassembly rules
  backend/                # Node + Fastify API, pg-boss scheduler, docker helpers, worker script, SQL migrations
    scripts/              #   migrate-v1-library.js (SQLite mcxpub -> Postgres library)
  frontend/               # no-build ESM app (index.html + js/), see V2_DESIGN.md §8
  deploy/                 # docker-compose.yml (local bring-up) + Dockerfile.worker
  DEPLOY.md               # private-cloud setup, v1 import, smoke test, cutover
```

`schema/` and `contracts/` sit at the top level because both `frontend/` and `backend/`
consume them. The v2 `frontend/`+`backend/` split mirrors `../v1/`.

## Phase status

- [x] **Phase 1 — Contracts**: schema extracted & versioned; REST (OpenAPI) + SSE + the
  `_DataLink_` normalization spec defined.
- [x] **Phase 2 — DB + Fastify API** (`backend/`): Postgres schema (`blobs`/`jobs`/`library`),
  content-addressed normalization/reassembly (verified), Ajv validation + preview limits,
  capability tokens, SSE plumbing, worker completion callback. Not yet run on the Node 22+
  target (dev sandbox is Node 12); SQLite→Postgres data migration still to do.
- [x] **Phase 3 — Event scheduler + worker protocol** (`backend/scheduler.js`, `backend/docker.js`,
  `backend/worker/mcx-run.sh`): pg-boss consumer claims queued jobs → `docker service create
  --generic-resource NVIDIA_GPU=1` → holds slot until done (caps at GPU count) → kill on
  timeout → cleanup. Worker fetches input + pushes output/detphoton over HTTP (no NFS).
  Not yet run against a live swarm (dev sandbox has no Docker/GPU/Node 20).
- [x] **Phase 4 — Frontend rebuild** (`frontend/`): no-build native-ESM app + import map
  (three, @json-editor, pako). Reactive store, fetch + SSE client, JSON-Editor from the
  API schema, Browse/Share library, and a three.js volume raycaster ported to modern
  three (GLSL3 + `Data3DTexture`) with a self-contained pako volume decode (no jdata/numjs).
  Parses clean; not yet run in a browser against a live API.
- [x] **Phase 5 — Cutover + migration + docs**: `backend/scripts/migrate-v1-library.js`
  (SQLite `mcxpub` → Postgres `library`, normalizing each doc), `deploy/docker-compose.yml`
  (local GPU-less bring-up), `deploy/Dockerfile.worker`, and `DEPLOY.md` (private-cloud
  setup, GPU/swarm config, v1 import, smoke test, cutover + rollback).

## Status

All five phases are written. **Not yet run** on a real target — the next step is a live
integration pass on Node 20 + Postgres (+ a browser, + a GPU swarm for full dispatch);
see the "Known validation gaps" in `DEPLOY.md`.
