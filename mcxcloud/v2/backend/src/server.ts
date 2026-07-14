import Fastify from 'fastify';
import cors from '@fastify/cors';
import { config } from './config.ts';
import { initQueue } from './queue.ts';
import { schemaRoutes } from './routes/schema.ts';
import { jobRoutes } from './routes/jobs.ts';
import { libraryRoutes } from './routes/library.ts';
import { blobRoutes } from './routes/blobs.ts';

const app = Fastify({ logger: true, bodyLimit: 64 * 1024 * 1024 });

await app.register(cors, {
  origin: config.corsOrigin,
  methods: ['GET', 'POST', 'DELETE', 'OPTIONS'],
});

app.get('/health', async () => ({ status: 'ok' }));

await app.register(schemaRoutes);
await app.register(jobRoutes);
await app.register(libraryRoutes);
await app.register(blobRoutes);

// pg-boss enqueue path; the consumer/scheduler that dispatches to Docker Swarm is Phase 3.
try {
  await initQueue();
  app.log.info('job queue ready (pg-boss)');
} catch (err) {
  app.log.warn(`queue init failed — submit still works, dispatch is Phase 3: ${(err as Error).message}`);
}

await app.listen({ port: config.port, host: config.host });
