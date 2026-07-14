# MCX Cloud v2 — backend

Fastify API implementing [`../contracts/openapi.yaml`](../contracts/openapi.yaml).
**Plain ESM JavaScript + JSDoc `// @ts-check`** — no build step, no transpile, and still
type-checkable with `tsc`. See [`../V2_DESIGN.md`](../V2_DESIGN.md) §5–§7.5.

## Requirements

- **Node.js ≥ 20 LTS**
- **PostgreSQL 13+**

## Setup

```sh
cp .env.example .env          # edit DATABASE_URL, CORS_ORIGIN, WORKER_SECRET
createdb mcxcloud
npm install
npm run migrate               # node src/migrate.js → applies db/migrations/*.sql
npm start                     # node src/server.js  → http://localhost:8080
npm run typecheck             # optional: tsc --noEmit over the @ts-check'd sources (no build)
```

Load env vars however you prefer, e.g. `node --env-file=.env src/server.js`.

## Layout

```
db/migrations/001_init.sql    blobs · blob_refs · jobs · library
src/
  server.js     Fastify app + route registration + health
  config.js     env config
  db.js         pg pool + withTx()
  hash.js       sha256 + stableStringify (canonicalization)
  jdata.js      normalize() / reassemble()  (contracts/normalization.md)
  blobs.js      content-addressed put/get + refcount/GC
  tokens.js     per-job capability tokens
  schema.js     Ajv validator (../schema/mcx-input.v1.json) + preview-limit checks
  sse.js        in-process job event bus
  queue.js      pg-boss enqueue (consumer/scheduler = Phase 3)
  migrate.js    migration runner
  routes/       schema · jobs · library · blobs
```

Every source file starts with `// @ts-check` and uses JSDoc types, so the editor and
`npm run typecheck` catch type errors without a build. Dependencies (Fastify 5, pg-boss 10)
target Node 20+.

## Implemented (Phase 2)

- `POST /jobs` — validate → preview-limit check → **normalize** (extract heavy JData
  arrays to content-addressed `blobs`) → whole-record cache hit reuse → insert + enqueue.
- `GET /jobs/:id`, `DELETE /jobs/:id` (token), `GET /jobs/:id/stream` (SSE),
  `GET /jobs/:id/output|detphoton` (token), `POST /jobs/:id/complete` (worker callback).
- `GET/POST /library`, `GET /library/:id` (reassembled), `GET /blobs/sha256/:hex`,
  `GET /schema/mcx-input.v1`, `GET /health`.

## Deferred to Phase 3

- pg-boss **consumer/scheduler** that dispatches queued jobs to Docker Swarm
  (`docker service create --generic-resource NVIDIA_GPU=1 …`) and the mcx container's
  `POST /jobs/:id/complete` callback wiring; kill/cleanup as queued timers.

## Verification status

The core dedup logic (normalize / reassemble / canonical hashing) is verified against
schema-shaped samples: a cloned simulation dedups to a single `blobs` row with an
identical `doc_hash`, and `reassemble()` is an exact inverse of `normalize()`. Full server
boot + Postgres integration + `npm run typecheck` should be run on the Node ≥ 20 target
(the dev sandbox had Node 12 and no Postgres).
