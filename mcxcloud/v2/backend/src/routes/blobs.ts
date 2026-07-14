import type { FastifyInstance } from 'fastify';
import { pool } from '../db.ts';

export async function blobRoutes(app: FastifyInstance): Promise<void> {
  // Resolve a content-addressed blob. Target of exported `_DataLink_` refs so external
  // JData tools can fetch the referenced array over HTTPS.
  app.get('/blobs/sha256/:hex', async (req, reply) => {
    const { hex } = req.params as { hex: string };
    if (!/^[0-9a-f]{64}$/.test(hex)) {
      return reply.code(400).send({ status: 'error', message: 'bad hash' });
    }
    const r = await pool.query('select data from blobs where hash = $1', ['sha256/' + hex]);
    if (r.rowCount === 0) return reply.code(404).send({ status: 'error', message: 'not found' });
    return reply.type('application/json').send((r.rows[0].data as Buffer).toString('utf8'));
  });
}
