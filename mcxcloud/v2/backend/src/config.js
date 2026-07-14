// @ts-check

/**
 * @typedef {Object} Config
 * @property {number} port
 * @property {string} host
 * @property {string} databaseUrl
 * @property {string} corsOrigin
 * @property {number} threshold
 * @property {string} workerSecret
 */

/** @type {Config} */
export const config = {
  port: Number(process.env.PORT ?? 8080),
  host: process.env.HOST ?? '0.0.0.0',
  databaseUrl: process.env.DATABASE_URL ?? 'postgres://mcxcloud@localhost/mcxcloud',
  corsOrigin: process.env.CORS_ORIGIN ?? 'https://mcx.space',
  threshold: Number(process.env.BLOB_THRESHOLD ?? 4096),
  workerSecret: process.env.WORKER_SECRET ?? 'change-me',
};
