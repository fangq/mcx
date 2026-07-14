# MCX Cloud v2 — backend

Fastify API implementing [`../contracts/openapi.yaml`](../contracts/openapi.yaml). Node.js +
TypeScript, **no build** (run `.ts` directly via Node's native type-stripping). See
[`../V2_DESIGN.md`](../V2_DESIGN.md) §5–§7.5.

## Requirements

- **Node.js ≥ 22.18** (runs `.ts` files directly; on 22.6–22.17 use `--experimental-strip-types`)
- **PostgreSQL 13+**

## Setup

```sh
cp .env.example .env          # edit DATABASE_URL, CORS_ORIGIN, WORKER_SECRET
createdb mcxcloud
npm install
npm run migrate               # applies db/migrations/*.sql
npm start                     # node src/server.ts  → http://localhost:8080
npm run typecheck             # optional: tsc --noEmit (CI gate, not a build)
```

Load env vars however you prefer, e.g. `node --env-file=.env src/server.ts`.

## Layout

```
db/migrations/001_init.sql    blobs · blob_refs · jobs · library
src/
  server.ts     Fastify app + route registration + health
  config.ts     env config
  db.ts         pg pool + withTx()
  hash.ts       sha256 + stableStringify (canonicalization)
  jdata.ts      normalize() / reassemble()  (contracts/normalization.md)
  blobs.ts      content-addressed put/get + refcount/GC
  tokens.ts     per-job capability tokens
  schema.ts     Ajv validator (../schema/mcx-input.v1.json) + preview-limit checks
  sse.ts        in-process job event bus
  queue.ts      pg-boss enqueue (consumer/scheduler = Phase 3)
  migrate.ts    migration runner
  routes/       schema · jobs · library · blobs
```

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

The core dedup logic (normalize / reassemble / canonical hashing) was verified against
sample inputs mirroring the real schema: a cloned simulation dedups to a single `blobs`
row with an identical `doc_hash`, and `reassemble()` is an exact inverse of `normalize()`.
Full server boot + Postgres integration + `npm run typecheck` must be run on the
Node ≥ 22.18 target (the dev sandbox had Node 12 and no Postgres, so it could not execute
Fastify v5 or `tsc` 5.8).
