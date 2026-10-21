// @ts-check
import Fastify from 'fastify';
import cors from '@fastify/cors';
import { config } from './config.js';
import { initQueue } from './queue.js';
import { initScheduler } from './scheduler.js';
import { purgeOldJobs } from './purge.js';
import { schemaRoutes } from './routes/schema.js';
import { jobRoutes } from './routes/jobs.js';
import { libraryRoutes } from './routes/library.js';
import { blobRoutes } from './routes/blobs.js';
import { adminRoutes } from './routes/admin.js';

const app = Fastify({ logger: true, bodyLimit: 64 * 1024 * 1024 });

// Raw parsers for the worker artifact uploads / log (bypass JSON parsing).
app.addContentTypeParser('application/octet-stream', { parseAs: 'buffer' }, (_req, body, done) => done(null, body));
app.addContentTypeParser('text/plain', { parseAs: 'string' }, (_req, body, done) => done(null, body));

await app.register(cors, {
  origin: config.corsOrigin,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
});

app.get('/health', async () => ({ status: 'ok' }));

await app.register(schemaRoutes);
await app.register(jobRoutes);
await app.register(libraryRoutes);
await app.register(blobRoutes);
await app.register(adminRoutes);
if (!config.adminSecret) app.log.warn('ADMIN_SECRET unset — library review/admin API is disabled');

// Queue + scheduler: submit enqueues; the scheduler consumes and dispatches to Swarm.
try {
  await initQueue();
  app.log.info('job queue ready (native Postgres SKIP LOCKED + LISTEN/NOTIFY)');
  if (config.runScheduler) {
    const capacity = await initScheduler();
    app.log.info(`scheduler running (capacity=${capacity} GPU slot(s))`);
  }
} catch (err) {
  app.log.warn(`queue/scheduler init failed — submit still works: ${(/** @type {Error} */ (err)).message}`);
}

// Periodically purge expired non-library jobs (keeps library results cached forever).
const runPurge = () =>
  purgeOldJobs(config.jobTtlMs)
    .then((n) => { if (n) app.log.info(`purged ${n} expired job(s)`); })
    .catch((e) => app.log.warn(`purge failed: ${(/** @type {Error} */ (e)).message}`));
const purgeTimer = setInterval(runPurge, 10 * 60 * 1000);
if (typeof purgeTimer.unref === 'function') purgeTimer.unref();
runPurge(); // once at startup

await app.listen({ port: config.port, host: config.host });
