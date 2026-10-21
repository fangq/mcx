// @ts-check
import { mcxSchema } from '../schema.js';

/** @param {import('fastify').FastifyInstance} app */
export async function schemaRoutes(app) {
  // The single source of truth for the input format; the frontend editor fetches it here.
  app.get('/schema/mcx-input.v1', async (_req, reply) => {
    reply.type('application/schema+json').send(mcxSchema);
  });
}
