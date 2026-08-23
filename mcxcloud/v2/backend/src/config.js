// @ts-check

/**
 * @typedef {Object} Config
 * @property {number} port
 * @property {string} host
 * @property {string} databaseUrl
 * @property {string} corsOrigin
 * @property {number} threshold
 * @property {string} workerSecret
 * @property {boolean} runScheduler
 * @property {string} workerApiUrl     API base URL reachable from inside the swarm container
 * @property {string} workerImage      docker image for the mcx worker
 * @property {string} workerImageMmc   docker image for mesh (mmc) jobs; future engines (redbird) get their own image + env
 * @property {string} workerImageRedbird  docker image for redbird (FEM diffusion) jobs: a
 *   Python base + `pip install redbirdpy`, no GPU needed. NOT fangqq/mcxstudio — that image
 *   does ship redbird-matlab, but its Octave is 4.0.0, which has no containers.Map at all,
 *   and redbird's FEM path uses containers.Map unconditionally (rbrunforward.m), so it
 *   cannot run there; its bundled redbird is also an older snapshot missing rbgetdetdir.m,
 *   and its zmat is incomplete so loadjson could not even unpack zlib-packed mesh arrays.
 * @property {string} workerConstraintRedbird  placement constraint for redbird jobs — CPU-only, FEM assembly/solve benefits from many cores, so pin to high-core-count nodes via a shared label (e.g. 'node.labels.cpu-heavy==1'; Swarm ANDs multiple --constraint flags, so pinning to two specific hosts needs a label shared by both, not two constraints). '' = any node
 * @property {number} redbirdCpuReserve  swarm --reserve-cpu (core count) per redbird job
 * @property {number} redbirdCpuLimit    swarm --limit-cpu (core count) per redbird job
 * @property {boolean} redbirdEnabled  master switch for the redbird engine (default OFF).
 *   Requires, on every eligible node: the worker image actually carrying octave +
 *   redbird-matlab + jsonlab/iso2mesh on the load path, and the placement label that
 *   workerConstraintRedbird selects. Until those are verified, submissions naming
 *   Session.Engine=redbird are rejected up front rather than dispatched and left to fail.
 * @property {number} redbirdMaxRuntimeMs  per-attempt runtime cap for redbird jobs. Much
 *   larger than the MC engines' maxRuntimeMs: a direct sparse FEM solve is far slower than
 *   a GPU MC run (a 6.5k-node CW slab measured ~20 s locally: ~8.5 s mesh prep + ~11.6 s
 *   solve), and the complex/RF path is slower still.
 * @property {number} redbirdMaxNodes  mesh node cap for redbird. Far below mmc's 300k: FEM
 *   assembly + direct solve grow superlinearly in time AND memory, so a mesh mmc handles
 *   fine would hang or OOM the FEM path.
 * @property {number} maxRuntimeMs      hard cap per simulation ATTEMPT before kill
 * @property {number} restartMaxAttempts  swarm --restart-max-attempts (1 = no retry, matches
 *   the old --restart-condition none behavior); container failures (a GPU that's down, a
 *   phantom/stale generic-resource UUID, a transient driver hiccup) get retried as a FRESH
 *   scheduling decision — swarmkit gives the replacement task an empty NodeID and re-runs
 *   full node selection, and a node's claimed GPUs are FIFO (claim from the front, release
 *   to the back on task removal), so a single broken GPU does not get deterministically
 *   re-selected either on the same node or (moreso) across a retry that lands elsewhere
 * @property {number} restartDelayMs    swarm --restart-delay between attempts
 * @property {number} maxConcurrent     0 = auto-detect from swarm GPU count
 * @property {string} workerConstraint  optional swarm placement constraint (e.g. 'node.hostname==neza'); '' = any node
 * @property {string} workerConstraintMmc  placement constraint for mesh (mmc) jobs only — mmc-OpenCL kernel JIT is driver-sensitive (e.g. fails on 530.30, flaky on 470.182), so pin mmc to verified nodes via e.g. 'node.labels.mmc==1'; '' = fall back to workerConstraint
 * @property {number} jobTtlMs          purge non-library jobs older than this (ms)
 * @property {number} minSubmitGapMs    minimum gap between submissions from one client (ms)
 * @property {string} nvCachePath       host directory bind-mounted into every GPU worker as CUDA_CACHE_PATH,
 *   so the driver's PTX JIT cache survives across ephemeral job containers (mcx/mmc ship
 *   SASS for a handful of archs; a GPU outside that set forces a JIT recompile on every
 *   *fresh* container — this was measured to cost ~10s per job vs ~0.3s once cached).
 *   Must pre-exist and be writable by the container's runtime UID on every GPU node.
 *   '' disables the mount (each job reverts to paying the JIT cost every time).
 * @property {string} adminSecret       shared secret for the library review/admin API ('' disables it)
 */

/** @type {Config} */
export const config = {
  port: Number(process.env.PORT ?? 8080),
  host: process.env.HOST ?? '0.0.0.0',
  databaseUrl: process.env.DATABASE_URL ?? 'postgres://mcxcloud@localhost/mcxcloud',
  corsOrigin: process.env.CORS_ORIGIN ?? 'https://mcx.space',
  threshold: Number(process.env.BLOB_THRESHOLD ?? 4096),
  workerSecret: process.env.WORKER_SECRET ?? 'change-me',
  runScheduler: process.env.RUN_SCHEDULER !== '0',
  workerApiUrl: process.env.WORKER_API_URL ?? 'http://localhost:8080',
  workerImage: process.env.WORKER_IMAGE ?? 'fangqq/mcx:v2024.2',
  workerImageMmc: process.env.WORKER_IMAGE_MMC ?? 'fangqq/mmc:v2025.10',
  workerImageRedbird: process.env.WORKER_IMAGE_REDBIRD ?? 'fangqq/redbird:v0.4.1',
  workerConstraintRedbird: process.env.WORKER_NODE_CONSTRAINT_REDBIRD ?? '',
  redbirdCpuReserve: Number(process.env.REDBIRD_CPU_RESERVE ?? 8),
  redbirdCpuLimit: Number(process.env.REDBIRD_CPU_LIMIT ?? 16),
  redbirdEnabled: process.env.REDBIRD_ENABLED === '1',
  redbirdMaxRuntimeMs: Number(process.env.REDBIRD_MAX_RUNTIME_MS ?? 600000), // 10 min
  redbirdMaxNodes: Number(process.env.REDBIRD_MAX_NODES ?? 50000),
  maxRuntimeMs: Number(process.env.MAX_RUNTIME_MS ?? 60000),
  restartMaxAttempts: Number(process.env.RESTART_MAX_ATTEMPTS ?? 2),
  restartDelayMs: Number(process.env.RESTART_DELAY_MS ?? 2000),
  maxConcurrent: Number(process.env.MAX_CONCURRENT ?? 0),
  workerConstraint: process.env.WORKER_NODE_CONSTRAINT ?? '',
  workerConstraintMmc: process.env.WORKER_NODE_CONSTRAINT_MMC ?? '',
  nvCachePath: process.env.NV_CACHE_PATH ?? '/var/lib/mcxcloud/nvcache',
  jobTtlMs: Number(process.env.JOB_TTL_MS ?? 3600000), // 1 hour
  minSubmitGapMs: Number(process.env.MIN_SUBMIT_GAP_MS ?? 5000), // 5 s
  adminSecret: process.env.ADMIN_SECRET ?? '', // '' disables the library review/admin API
};
