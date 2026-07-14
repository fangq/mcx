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
 * @property {number} maxRuntimeMs      hard cap per simulation before kill
 * @property {number} maxConcurrent     0 = auto-detect from swarm GPU count
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
  maxRuntimeMs: Number(process.env.MAX_RUNTIME_MS ?? 60000),
  maxConcurrent: Number(process.env.MAX_CONCURRENT ?? 0),
};
