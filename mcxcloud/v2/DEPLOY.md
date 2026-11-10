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

## 3. Worker image

The scheduler runs `worker/mcx-run.sh` inside the container. It needs only `bash` + a
HTTP client, and uses **`wget`** (GNU wget) — present in the stock `fangqq/mcx` image — so
**no augmented image is required**; set `WORKER_IMAGE` to your mcx image (e.g.
`fangqq/mcx:v2025.10`). Verified on `fangqq/mcx:v2025.10`: bash + GNU wget 1.17.1, no curl.

Only if your mcx image lacks both curl and wget, build the optional augmented image:

```sh
cd v2/deploy
docker build -f Dockerfile.worker -t <registry>/mcx-worker:<tag> .
docker push <registry>/mcx-worker:<tag>
```

## 4. Database + config

The schema is portable to **Postgres 9.5+** (UUIDs are minted in Node; the native
`SKIP LOCKED` queue needs no extension and no queue library). Use the system Postgres, or a
persistent container if you can't provision a system role:

```sh
# option A — system Postgres (needs a superuser to create the role/db once):
sudo -u postgres createuser mcxcloud --pwprompt
sudo -u postgres createdb -O mcxcloud mcxcloud

# option B — persistent container (named volume survives restarts/reboots):
docker volume create mcxcloud_pgdata
docker run -d --name mcxcloud-db --restart unless-stopped \
  -e POSTGRES_USER=mcxcloud -e POSTGRES_PASSWORD="$(openssl rand -hex 24)" \
  -e POSTGRES_DB=mcxcloud -v mcxcloud_pgdata:/var/lib/postgresql/data \
  -p 127.0.0.1:5433:5432 postgres:12          # bind localhost only
```

```sh
cd v2/backend
cp .env.example .env
chmod 600 .env             # it holds DATABASE_URL (with password) + WORKER_SECRET
# set: DATABASE_URL, CORS_ORIGIN=<frontend origin>, WORKER_API_URL, WORKER_IMAGE,
#      and a STRONG secret:  WORKER_SECRET=$(openssl rand -hex 32)
npm install                # no build; installs fastify, pg, ajv, @fastify/cors
npm run migrate            # applies db/migrations/*.sql
```

- `WORKER_API_URL` must be **reachable from inside a swarm container** (never `localhost`,
  which resolves to the container itself). **Recommended:** the public HTTPS proxy endpoint
  (`https://<manager-fqdn>/api`, §7) — then the API can stay bound to `127.0.0.1` (`HOST`)
  and the swarm nodes reach it over TLS via Apache; no need to expose port 8080 on the LAN.
  *Alternative:* `http://<manager-LAN-IP>:8080` with `HOST=0.0.0.0`, but then firewall 8080
  to the cluster subnet. (A `localhost`/localhost-bound mismatch makes every worker callback
  fail instantly → jobs run until the `MAX_RUNTIME_MS` kill with no output.)
- `CORS_ORIGIN` must be the exact origin serving the frontend (e.g. `https://mcx.space`).
- `WORKER_SECRET` must be a strong random value (the worker presents it to push results);
  keep port 8080 **internal only** — never expose it publicly.

## 5. Import the v1 shared library (optional)

```sh
sqlite3 -json /var/lib/mcxcloud/db/mcxcloud.db \
  "select time,title,comment,license,name,inst,email,netname,json,thumbnail,\
   upvote,downvote,readcount,runcount,createtime from mcxpub" > mcxpub.json
node scripts/migrate-v1-library.js mcxpub.json
```

Each entry is normalized (heavy Shapes → content-addressed `blobs`), so duplicated volumes
across entries are stored once — the fix for the v1 >1 GB bloat.

### Seed the curated builtin examples

Maintainer-curated demos live in `db/seeds/*.json` (e.g. the multi-source examples
`multisrc`/`eachsrc`). Seeding is idempotent (existing `doc_hash` entries are skipped) and
entries are inserted pre-approved:

```sh
node --env-file=.env scripts/seed-library.js db/seeds/*.json
```

## 6. Run the API + scheduler (manager)

For a quick foreground run: `node --env-file=.env src/server.js`. For production, use the
provided **systemd** unit so it restarts on crash/reboot (the scheduler needs the running
user to be in the `docker` group):

```sh
sudo cp v2/deploy/mcxcloud-api.service /etc/systemd/system/   # adjust User/paths inside
sudo systemctl daemon-reload
sudo systemctl enable --now mcxcloud-api
journalctl -u mcxcloud-api -f
```

Set `RUN_SCHEDULER=1` in `.env` to dispatch (0 = API-only). For a multi-node manager, run
API-only replicas with `RUN_SCHEDULER=0` and a single node with the scheduler enabled.
`MAX_CONCURRENT=0` auto-detects the swarm GPU count; `WORKER_NODE_CONSTRAINT` (e.g.
`node.hostname==neza`) optionally pins dispatch to specific nodes.

### Optional: prompt-to-JSON (the LLM tab)

The **LLM** tab turns a plain-language description into an MCX input document. Conversion
runs server-side via MCXLLM (an LLM + a pydantic/langchain post-processor), which lives in
its own project and its own Python environment — `backend/llm/mcxllm_bridge.py` imports it
rather than vendoring a copy. The API spawns one bridge process per request (~2 s for the
langchain import, plus the model call). The method is described in
<https://www.biorxiv.org/content/10.64898/2026.07.20.738933v1>, which the tab asks users to
cite.

```sh
# in backend/.env
LLM_ENABLED=1
LLM_PYTHON=/path/to/miniconda3/envs/LLM_mcx/bin/python   # has langchain + openai + pydantic
LLM_MCX_DIR=/path/to/LLM_mcx/.../performance_testing     # holds LLMMCX_gpt4o.py
LLM_MIN_GAP_MS=10000                                     # each call costs money or a GPU slot

# self-hosted backend — the FULL OpenAI-style chat URL of an ollama server or proxy
LLM_OLLAMA_URL=https://neurojson.io/api/v1/ollama-public/chat
LLM_OLLAMA_MODELS=qwen3.8:27b        # the neurojson proxy serves this one model, whatever is asked
# or a raw ollama host:  LLM_OLLAMA_URL=http://shui:11434/v1   (/chat/completions is appended)
#                        LLM_OLLAMA_MODELS=qwen3.8:27b   (comma-separate to offer several)
# hosted backend — bring-your-own-key; '' hides it. No server-side key is needed or used.
LLM_OPENAI_MODEL=gpt-4o
# which (backend, model) is preselected
LLM_BACKEND=ollama
LLM_DEFAULT_MODEL=qwen3.8:27b
```

### Choosing a backend

Both backends run the **same** MCXLLM code — the pydantic schema, prompt template and
post-processor are model-agnostic, and both speak the OpenAI chat-completions format, so
`mcxllm_bridge.py` only swaps how a completion is fetched (the OpenAI SDK for `openai`, a
direct POST to `LLM_OLLAMA_URL` for `ollama` — the SDK can't target a proxy whose path
isn't `/chat/completions`, such as neurojson.io's `/ollama-public/chat`). Every
configured `(backend, model)` pair is offered in a dropdown in the LLM tab and chosen per
request, which makes the UI usable for side-by-side comparison. `GET /llm/health` returns
the list, so a frontend never offers a backend the server cannot serve.

Measured on one prompt ("a 100x100x100 domain, two spheres, 1e7 photons, energy deposit"),
qwen3.8:27b on a LAN ollama host vs gpt-4o:

| backend | latency | result |
|---|---|---|
| `gpt-4o` | ~6 s | schema-valid, but `Domain.Dim` stayed `[60,60,60]` against a 100³ prompt |
| `qwen3.8:27b` | ~12-24 s | schema-valid, `Domain.Dim` correct |

So the hosted model is faster and the local one was more accurate on that prompt — one
sample, not a benchmark, which is the reason the choice is exposed at runtime instead of
baked in. Note `LLM_TIMEOUT_MS` must accommodate the slowest model plus a cold model load
(the first request to an unloaded ollama model pays ~10 s extra).

**OpenAI runs on the user's key, never the server's.** Selecting `gpt-4o` shows an API-key
field; the key is sent with that one request (`apiKey` in the `POST /llm/generate` body),
handed to the bridge on stdin — not argv or the environment — and redacted from any error
text before it is logged or returned. The API strips every `OPENAI_*` variable from the
bridge's environment, so an `OPENAI_API_KEY` that happens to be set on the server can never
be billed for a user's request. Nothing stores the key: the browser keeps it only in the
input box. With
`LLM_ENABLED` unset the route answers 503 and the frontend **hides the tab** (it probes
`GET /llm/health` at boot), so the same static frontend can serve an API without the Python
side installed.

Two behaviours worth knowing, both in `src/llm.js`:

- **Schema defaults are filled in.** MCXLLM's `reorder_keys()` predates this schema and
  omits `Session.DoAutoThread` and `Domain.OriginType`, which the editor marks required —
  without the fill every generated input shows as invalid and Run refuses it. Only these
  documented defaults are supplied.
- **Contradictions are reported, not repaired.** MCXLLM sets `Shapes[].Grid.Size` from the
  prompt but can leave `Domain.Dim` at its default `[60,60,60]`, so a prompt asking for a
  100³ domain may yield a 60³ volume that clips the shapes. Both values validate on their
  own, so the status box warns instead — quietly rewriting the model's answer would hide
  the errors most worth seeing.

## 7. Expose the API over HTTPS (reverse proxy + TLS)

The frontend runs from `https://mcx.space`, so the browser will **only** call the API over
HTTPS (mixed-content rule). Keep the Node API bound to `127.0.0.1:8080` and put it behind
the manager's existing TLS vhost. A ready-to-install Apache config is in
[`deploy/apache-mcxcloud.conf`](deploy/apache-mcxcloud.conf) — it proxies `/api/` to the
local API:

```sh
sudo a2enmod proxy proxy_http headers            # (already enabled on zodiac)
# add the <Location "/api/"> block to the :443 vhost, or Include the file from it
sudo apache2ctl configtest && sudo systemctl reload apache2
curl https://<manager-fqdn>/api/health           # -> {"status":"ok"}
```

Notes: the API sets its own CORS headers (don't add them in Apache too); SSE
(`GET /api/jobs/:id/stream`) is a long-lived stream — the config sets `timeout=3600` and
disables buffering; verify events aren't delayed once live.

## 8. Deploy the frontend

Copy `v2/frontend/` to the static host (IONOS/mcx.space). Set `window.MCX_API_BASE` in
`index.html` to the **HTTPS** API URL from §7 (e.g. `https://<manager-fqdn>/api`); it must
match `CORS_ORIGIN`. Serve over http(s) (native ESM won't load from `file://`).

## 9. Smoke test (end to end)

1. `curl https://<api>/health` → `{"status":"ok"}`.
2. `curl https://<api>/schema/mcx-input.v1` → the schema JSON.
3. In the browser: **Create** builds a form; **JSON** shows valid; **Run** → submit →
   the SSE log streams status/log and ends in `completed`; **Preview → Draw Output**
   renders the volume; **Download output** works.
4. **Share** a simulation, then **Browse** finds it; **Load** repopulates the form.
5. Re-submit the same input → returns instantly as `cached` (whole-record dedup).
6. `select count(*), sum(size) from blobs;` — cloning a sim adds no new blob rows.

## 10. Cutover from v1

- v1 (`../v1`) and v2 run in parallel; they share nothing (separate DB, separate API host).
- Point a test subdomain / `MCX_API_BASE` at the v2 API and validate with §9.
- Flip mcx.space to serve `v2/frontend` (or switch the API origin) once green.
- **Rollback:** repoint the frontend/API origin back to v1; no data migration is destructive
  (the import only reads the v1 SQLite file).

## Validation status (verified on zodiac, Postgres 12)

Backend + a real GPU run are validated end-to-end (submit → validate → normalize → dedup →
native `SKIP LOCKED` dispatch → `docker service` on a GPU node → real mcx → wget push →
SSE `completed`; whole-record cache confirmed; `countGpus()` correct). Remaining to check:

- **three.js volume axis order** in `frontend/js/preview.js` (v1 transposed; this port does
  not) and `@json-editor` ESM interop — needs a browser against a live API (§9).
- `jobs.node` / `jobs.gpu` columns are not populated by the worker/complete path (cosmetic).
- SSE latency through the Apache proxy under real load (§7 note).
