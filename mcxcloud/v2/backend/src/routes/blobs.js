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
    const buf = /** @type {Buffer} */ (r.rows[0].data);
    // content-addressed blobs are immutable -> cache hard
    reply.header('cache-control', 'public, max-age=31536000, immutable');
    // thumbnails are stored as `data:<type>;base64,...` URIs; decode to real image bytes
    // so an <img src> pointing here renders (rather than getting the URI as text).
    if (buf.slice(0, 5).toString('latin1') === 'data:') {
      const text = buf.toString('utf8');
      const m = /^data:([\w.+-]+\/[\w.+-]+)?(;base64)?,/.exec(text);
      if (m) {
        const payload = text.slice(m[0].length);
        const body = m[2] ? Buffer.from(payload, 'base64') : Buffer.from(decodeURIComponent(payload), 'utf8');
        return reply.type(m[1] || 'application/octet-stream').send(body);
      }
    }
    // otherwise it's JData/JSON array content (the target of a `_DataLink_` ref)
    return reply.type('application/json').send(buf.toString('utf8'));
  });
}
