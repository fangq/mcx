# Deploying MCX Cloud v2 (a "private MCX cloud")

Architecture recap (see [`V2_DESIGN.md`](V2_DESIGN.md)): a **manager node** runs the
Fastify API + pg-boss scheduler + PostgreSQL; **worker nodes** contribute GPUs to a Docker
Swarm; the **static frontend** is served from anywhere and calls the API over HTTPS/SSE.

```
static frontend (any host) ──REST/SSE──▶ API+scheduler+Postgres (manager) ──▶ Docker Swarm (GPUs)
```

## 0. Quick local test (no GPUs)

Exercises the whole submit → validate → normalize → dedup → SSE → library/preview flow
with the scheduler off (nothing is dispatched):

```sh
cd v2/deploy
docker compose up -d
docker compose run --rm api npm run migrate     # create tables
# edit v2/frontend/index.html: window.MCX_API_BASE = 'http://localhost:8080'
open http://localhost:8000
```

## 1. Prerequisites (manager)

- Node.js **≥ 20 LTS**, PostgreSQL **13+**, Docker with Swarm mode (`docker swarm init`).
- Worker nodes joined to the swarm, each with NVIDIA drivers + `nvidia-container-toolkit`.

## 2. Advertise GPUs to the swarm

On **each GPU node**, expose GPUs as a generic resource named `NVIDIA_GPU` (this is what
`--generic-resource NVIDIA_GPU=1` schedules against). In `/etc/docker/daemon.json`:

```json
{
  "default-runtime": "nvidia",
  "node-generic-resources": ["NVIDIA_GPU=GPU-<uuid-1>", "NVIDIA_GPU=GPU-<uuid-2>"]
}
```

Get UUIDs with `nvidia-smi -a | grep UUID`. Also uncomment `swarm-resource = "DOCKER_RESOURCE_NVIDIA_GPU"`
in `/etc/nvidia-container-runtime/config.toml`, then `systemctl restart docker`. Verify:
`docker node inspect <node> --format '{{.Description.Resources.GenericResources}}'`.

## 3. Build & publish the worker image

The scheduler runs `worker/mcx-run.sh` inside the container, which needs `bash` + `curl`:

```sh
cd v2/deploy
docker build -f Dockerfile.worker -t <registry>/mcx-worker:v2024.2 .
docker push <registry>/mcx-worker:v2024.2
```

## 4. Database + config

```sh
createdb mcxcloud
cd v2/backend
cp .env.example .env       # set DATABASE_URL, CORS_ORIGIN, WORKER_SECRET, WORKER_API_URL,
                           # WORKER_IMAGE=<registry>/mcx-worker:v2024.2
npm install
npm run migrate            # applies db/migrations/*.sql
```

- `WORKER_API_URL` must be the manager address **reachable from inside a swarm container**
  (its LAN IP/hostname, not `localhost`).
- `CORS_ORIGIN` must be the exact origin serving the frontend (e.g. `https://mcx.space`).

## 5. Import the v1 shared library (optional)

```sh
sqlite3 -json /var/lib/mcxcloud/db/mcxcloud.db \
  "select time,title,comment,license,name,inst,email,netname,json,thumbnail,\
   upvote,downvote,readcount,runcount,createtime from mcxpub" > mcxpub.json
node scripts/migrate-v1-library.js mcxpub.json
```

Each entry is normalized (heavy Shapes → content-addressed `blobs`), so duplicated volumes
across entries are stored once — the fix for the v1 >1 GB bloat.

## 6. Run the API + scheduler (manager)

```sh
node --env-file=.env src/server.js
```

Run it under a supervisor (systemd/pm2). For a multi-node manager you can run API-only
replicas with `RUN_SCHEDULER=0` and a single node with the scheduler enabled.

## 7. Deploy the frontend

Copy `v2/frontend/` to the static host (IONOS/mcx.space). Set `window.MCX_API_BASE` in
`index.html` to the API origin. Must be served over http(s) (native ESM won't load from
`file://`).

## 8. Smoke test (end to end)

1. `curl https://<api>/health` → `{"status":"ok"}`.
2. `curl https://<api>/schema/mcx-input.v1` → the schema JSON.
3. In the browser: **Create** builds a form; **JSON** shows valid; **Run** → submit →
   the SSE log streams status/log and ends in `completed`; **Preview → Draw Output**
   renders the volume; **Download output** works.
4. **Share** a simulation, then **Browse** finds it; **Load** repopulates the form.
5. Re-submit the same input → returns instantly as `cached` (whole-record dedup).
6. `select count(*), sum(size) from blobs;` — cloning a sim adds no new blob rows.

## 9. Cutover from v1

- v1 (`../v1`) and v2 run in parallel; they share nothing (separate DB, separate API host).
- Point a test subdomain / `MCX_API_BASE` at the v2 API and validate with §8.
- Flip mcx.space to serve `v2/frontend` (or switch the API origin) once green.
- **Rollback:** repoint the frontend/API origin back to v1; no data migration is destructive
  (the import only reads the v1 SQLite file).

## Known validation gaps (run these on the target first)

- pg-boss v10 `work()` options in `src/scheduler.js` / `src/queue.js` (batch/handler shape).
- `countGpus()` parsing of `docker node inspect` output format in `src/docker.js`.
- three.js volume **axis order** in `frontend/js/preview.js` (v1 transposed; this port
  does not) and `@json-editor` ESM interop.
