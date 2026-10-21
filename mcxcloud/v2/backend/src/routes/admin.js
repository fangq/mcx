// @ts-check
import { config } from '../config.js';
import { pool, withTx } from '../db.js';
import { attachRefs, detachOwner, getBlob, putBlob } from '../blobs.js';
import { normalize, reassemble } from '../jdata.js';
import { validateInput } from '../schema.js';
import { checkAdminSecret, mintAdminToken, verifyAdminToken, SESSION_TTL_MS } from '../admin.js';

const LICENSES = new Set(['CC0', 'CC-BY', 'CC-BY-SA']);
/** @param {string | null} h */
const blobUrl = (h) => (h ? '/blobs/' + h : null);

/** last X-Forwarded-For entry = the client Apache actually saw (see routes/jobs.js). */
function clientIp(/** @type {import('fastify').FastifyRequest} */ req) {
  const xff = String(req.headers['x-forwarded-for'] || '');
  const parts = xff.split(',').map((s) => s.trim()).filter(Boolean);
  return parts[parts.length - 1] || req.ip || 'unknown';
}

/** failed-login throttle: ip -> { fails, until } */
const loginThrottle = new Map();

/** @param {import('fastify').FastifyInstance} app */
export async function adminRoutes(app) {
  /** Gate: require a valid admin session token; on failure send 403 and return false. */
  const requireAdmin = (/** @type {import('fastify').FastifyRequest} */ req, /** @type {import('fastify').FastifyReply} */ reply) => {
    const auth = String(req.headers['authorization'] || '');
    const token = auth.startsWith('Bearer ') ? auth.slice(7) : String(req.headers['x-admin-token'] || '');
    if (!verifyAdminToken(token)) {
      reply.code(403).send({ status: 'error', message: 'forbidden' });
      return false;
    }
    return true;
  };

  // ---- login: exchange ADMIN_SECRET for a short-lived session token ---------------
  app.post('/admin/login', async (req, reply) => {
    if (!config.adminSecret) return reply.code(404).send({ status: 'error', message: 'admin disabled' });
    const ip = clientIp(req);
    const t = loginThrottle.get(ip);
    if (t && t.until > Date.now()) {
      return reply.code(429).send({ status: 'error', message: 'too many attempts — try again later' });
    }
    const body = /** @type {{ secret?: string }} */ (req.body || {});
    const secret = String(req.headers['x-admin-secret'] || body.secret || '');
    if (!checkAdminSecret(secret)) {
      const fails = (t?.fails || 0) + 1;
      loginThrottle.set(ip, { fails, until: fails >= 5 ? Date.now() + 5 * 60 * 1000 : 0 });
      await new Promise((r) => setTimeout(r, 300)); // slow brute force a little
      return reply.code(403).send({ status: 'error', message: 'invalid secret' });
    }
    loginThrottle.delete(ip);
    return reply.send({ token: mintAdminToken(), expiresInMs: SESSION_TTL_MS });
  });

  // ---- list submissions by status (default: pending review) ----------------------
  app.get('/admin/library', async (req, reply) => {
    if (!requireAdmin(req, reply)) return;
    const status = String(/** @type {{status?:string}} */ (req.query).status || 'pending');
    const r = await pool.query(
      `select id, title, description, license, thumbnail_hash, run_count, submitter, status, created_at
       from library where status = $1 order by created_at desc limit 300`,
      [status],
    );
    return reply.send(
      r.rows.map((row) => ({
        id: row.id,
        title: row.title,
        description: row.description,
        license: row.license,
        thumbnail: blobUrl(row.thumbnail_hash),
        runCount: row.run_count,
        submitter: row.submitter,
        status: row.status,
        createdAt: row.created_at,
      })),
    );
  });

  // ---- full record (reassembled doc) for review / load-and-run -------------------
  app.get('/admin/library/:id', async (req, reply) => {
    if (!requireAdmin(req, reply)) return;
    const { id } = /** @type {{ id: string }} */ (req.params);
    const r = await pool.query(
      'select id, title, description, license, submitter, status, input_doc from library where id = $1',
      [id],
    );
    if (r.rowCount === 0) return reply.code(404).send({ status: 'error', message: 'not found' });
    const row = r.rows[0];
    const client = await pool.connect();
    try {
      const doc = await reassemble(row.input_doc, (h) => getBlob(client, h));
      return reply.send({
        id: row.id,
        title: row.title,
        description: row.description,
        license: row.license,
        submitter: row.submitter,
        status: row.status,
        doc,
      });
    } finally {
      client.release();
    }
  });

  // ---- approve -------------------------------------------------------------------
  app.post('/admin/library/:id/approve', async (req, reply) => {
    if (!requireAdmin(req, reply)) return;
    const { id } = /** @type {{ id: string }} */ (req.params);
    const r = await pool.query("update library set status = 'approved' where id = $1 returning id", [id]);
    if (r.rowCount === 0) return reply.code(404).send({ status: 'error', message: 'not found' });
    return reply.send({ id, status: 'approved' });
  });

  // ---- reject (delete the submission and GC its blobs) ---------------------------
  app.delete('/admin/library/:id', async (req, reply) => {
    if (!requireAdmin(req, reply)) return;
    const { id } = /** @type {{ id: string }} */ (req.params);
    const n = await withTx(async (client) => {
      // Delete the row FIRST so its thumbnail_hash FK is released, THEN detach + GC the blobs
      // (otherwise the thumbnail blob is still referenced -> library_thumbnail_hash_fkey).
      const d = await client.query('delete from library where id = $1', [id]);
      if (d.rowCount === 0) return 0;
      await detachOwner(client, 'library', id);
      return d.rowCount;
    });
    if (!n) return reply.code(404).send({ status: 'error', message: 'not found' });
    return reply.send({ id, status: 'rejected' });
  });

  // ---- edit / replace (admin-curated) --------------------------------------------
  // Replaces the submission's content in place (same id, original submitter preserved) and
  // keeps its current status, so the admin re-reviews the edited version before approving.
  app.put('/admin/library/:id', async (req, reply) => {
    if (!requireAdmin(req, reply)) return;
    const { id } = /** @type {{ id: string }} */ (req.params);
    const body = /** @type {{ title?: string, description?: string, license?: string, thumbnail?: string, doc?: unknown }} */ (
      req.body || {}
    );
    if (!body.title || !body.description || !body.license || !body.doc) {
      return reply.code(422).send({ status: 'invalid', message: 'missing required fields' });
    }
    if (!LICENSES.has(body.license)) return reply.code(422).send({ status: 'invalid', message: 'unknown license' });
    if (!validateInput(body.doc)) {
      return reply
        .code(422)
        .send({ status: 'invalid', message: 'schema validation failed', errors: validateInput.errors });
    }
    const n = await withTx(async (client) => {
      const cur = await client.query('select thumbnail_hash from library where id = $1', [id]);
      if (cur.rowCount === 0) return 0;
      const { doc, docHash, refs } = await normalize(
        body.doc,
        (canon, enc, size) => putBlob(client, canon, enc, size),
        config.threshold,
      );
      // Keep the existing thumbnail unless the admin captured a new one (editing only the
      // description shouldn't drop the image).
      let thumbHash = /** @type {string | null} */ (cur.rows[0].thumbnail_hash);
      if (body.thumbnail) {
        thumbHash = await putBlob(client, body.thumbnail, null, Buffer.byteLength(body.thumbnail, 'utf8'));
      }
      const owned = [...refs];
      if (thumbHash) owned.push(thumbHash);
      // Update the row FIRST (so thumbnail_hash points at a kept blob), then reconcile edges:
      // attach the new owned set, drop the stale edges, decrement + GC anything orphaned.
      await client.query(
        `update library set title = $2, description = $3, license = $4, input_doc = $5, doc_hash = $6, thumbnail_hash = $7
         where id = $1`,
        [id, body.title, body.description, body.license, doc, docHash, thumbHash],
      );
      await attachRefs(client, owned, 'library', id);
      const stale = await client.query(
        `delete from blob_refs where owner_kind = 'library' and owner_id = $1 and not (hash = any($2::text[])) returning hash`,
        [id, owned],
      );
      for (const row of stale.rows) {
        await client.query('update blobs set refcount = refcount - 1 where hash = $1', [row.hash]);
      }
      await client.query('delete from blobs where refcount <= 0');
      return 1;
    });
    if (!n) return reply.code(404).send({ status: 'error', message: 'not found' });
    return reply.send({ id, status: 'updated' });
  });
}
