import type { FastifyInstance } from 'fastify';
import { mcxSchema } from '../schema.ts';

export async function schemaRoutes(app: FastifyInstance): Promise<void> {
  // The single source of truth for the input format; the frontend editor fetches it here.
  app.get('/schema/mcx-input.v1', async (_req, reply) => {
    reply.type('application/schema+json').send(mcxSchema);
  });
}
