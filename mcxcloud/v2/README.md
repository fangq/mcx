# MCX Cloud v2

Modernization of MCX Cloud. Design rationale and decisions: [`V2_DESIGN.md`](V2_DESIGN.md).

v2 runs in parallel with the untouched v1 (`../v1/frontend`, `../v1/backend`) until cutover.

## Stack (decided)

- **DB:** PostgreSQL — JSONB metadata + a `blobs` `bytea` content-addressed table.
- **Backend:** Node.js + TypeScript + Fastify, **no build** (native TS type-strip);
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
  backend/                # Phases 2-3 (planned) — Node/TS + Fastify API, pg-boss scheduler, SQL migrations
  frontend/               # Phase 4 (planned) — no-build ESM app (index.html + js/ + vendor/), see V2_DESIGN.md §8
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
- [ ] Phase 3 — pg-boss event scheduler replacing `mcxcloudd` (Swarm dispatch unchanged)
- [ ] Phase 4 — Frontend rebuild (ESM modules, reactive store, three.js port, SSE client)
- [ ] Phase 5 — Cutover + "private MCX cloud" docs
