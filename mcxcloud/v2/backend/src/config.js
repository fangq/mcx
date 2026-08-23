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
