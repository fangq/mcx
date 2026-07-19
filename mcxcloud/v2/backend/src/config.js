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
 * @property {number} maxRuntimeMs      hard cap per simulation before kill
 * @property {number} maxConcurrent     0 = auto-detect from swarm GPU count
 * @property {string} workerConstraint  optional swarm placement constraint (e.g. 'node.hostname==neza'); '' = any node
 * @property {number} jobTtlMs          purge non-library jobs older than this (ms)
 * @property {number} minSubmitGapMs    minimum gap between submissions from one client (ms)
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
  maxConcurrent: Number(process.env.MAX_CONCURRENT ?? 0),
  workerConstraint: process.env.WORKER_NODE_CONSTRAINT ?? '',
  jobTtlMs: Number(process.env.JOB_TTL_MS ?? 3600000), // 1 hour
  minSubmitGapMs: Number(process.env.MIN_SUBMIT_GAP_MS ?? 5000), // 5 s
  adminSecret: process.env.ADMIN_SECRET ?? '', // '' disables the library review/admin API
};
