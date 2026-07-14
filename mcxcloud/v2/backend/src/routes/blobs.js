// @ts-check
import { pool } from '../db.js';

/** @param {import('fastify').FastifyInstance} app */
export async function blobRoutes(app) {
  // Resolve a content-addressed blob. Target of exported `_DataLink_` refs so external
  // JData tools can fetch the referenced array over HTTPS.
  app.get('/blobs/sha256/:hex', async (req, reply) => {
    const { hex } = /** @type {{ hex: string }} */ (req.params);
    if (!/^[0-9a-f]{64}$/.test(hex)) {
      return reply.code(400).send({ status: 'error', message: 'bad hash' });
    }
    const r = await pool.query('select data from blobs where hash = $1', ['sha256/' + hex]);
    if (r.rowCount === 0) return reply.code(404).send({ status: 'error', message: 'not found' });
    return reply.type('application/json').send((/** @type {Buffer} */ (r.rows[0].data)).toString('utf8'));
  });
}
