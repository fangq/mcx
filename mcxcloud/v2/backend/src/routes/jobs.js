// @ts-check
import { config } from '../config.js';
import { pool, withTx } from '../db.js';
import { attachRefs, getBlob, putBlob, putBlobRaw } from '../blobs.js';
import { normalize, reassemble } from '../jdata.js';
import { mintToken, tokenHash } from '../tokens.js';
import { validateInput, checkLimits } from '../schema.js';
import { enqueueJob } from '../queue.js';
import { publish, subscribe } from '../sse.js';

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

/**
 * @typedef {Object} JobRow
 * @property {string} id
 * @property {string} status
 * @property {string} token_hash
 * @property {string | null} output_hash
 * @property {string | null} detp_hash
 * @property {string | null} error
 */

/**
 * Returns { row } on match, { code: 404|403 } otherwise.
 * @param {string} id
 * @param {string | undefined} token
 * @returns {Promise<{ row?: JobRow, code?: number }>}
 */
async function authJob(id, token) {
  if (!UUID.test(id)) return { code: 404 };
  const r = await pool.query(
    'select id, status, token_hash, output_hash, detp_hash, error from jobs where id = $1',
    [id],
  );
  if (r.rowCount === 0) return { code: 404 };
  const row = /** @type {JobRow} */ (r.rows[0]);
  if (!token || row.token_hash !== tokenHash(token)) return { code: 403 };
  return { row };
}

/** @param {import('fastify').FastifyInstance} app */
export async function jobRoutes(app) {
  // ---- submit -----------------------------------------------------------------
  app.post('/jobs', async (req, reply) => {
    const body = /** @type {{ doc?: unknown, user?: Record<string, unknown> }} */ (req.body);
    if (!body?.doc || typeof body.doc !== 'object') {
      return reply.code(422).send({ status: 'invalid', message: 'missing doc' });
    }
    if (!validateInput(body.doc)) {
      return reply
        .code(422)
        .send({ status: 'invalid', message: 'schema validation failed', errors: validateInput.errors });
    }
    const limit = checkLimits(/** @type {Record<string, any>} */ (body.doc));
    if (limit) return reply.code(422).send({ status: 'invalid', message: limit });

    const { token, tokenHash: th } = mintToken();

    const result = await withTx(async (client) => {
      const { doc, docHash, refs } = await normalize(
        body.doc,
        (canon, enc, size) => putBlob(client, canon, enc, size),
        config.threshold,
      );

      // whole-record cache: reuse a prior completed result for the identical doc
      const cached = await client.query(
        `select output_hash, detp_hash from jobs
         where doc_hash = $1 and status in ('completed','cached') and output_hash is not null
         limit 1`,
        [docHash],
      );

      if (cached.rowCount && cached.rows[0].output_hash) {
        const ins = await client.query(
          `insert into jobs (input_doc, doc_hash, status, submitter, token_hash, output_hash, detp_hash, ended_at)
           values ($1,$2,'cached',$3,$4,$5,$6, now()) returning id`,
          [doc, docHash, body.user ?? null, th, cached.rows[0].output_hash, cached.rows[0].detp_hash],
        );
        const id = /** @type {string} */ (ins.rows[0].id);
        const owned = [...refs, cached.rows[0].output_hash];
        if (cached.rows[0].detp_hash) owned.push(cached.rows[0].detp_hash);
        await attachRefs(client, owned, 'job', id);
        return { id, status: 'cached', cached: true };
      }

      const ins = await client.query(
        `insert into jobs (input_doc, doc_hash, status, submitter, token_hash)
         values ($1,$2,'queued',$3,$4) returning id`,
        [doc, docHash, body.user ?? null, th],
      );
      const id = /** @type {string} */ (ins.rows[0].id);
      await attachRefs(client, refs, 'job', id);
      return { id, status: 'queued', cached: false };
    });

    if (!result.cached) {
      await enqueueJob(result.id, 50);
      publish(result.id, 'status', { status: 'queued' });
    }
    return reply.code(201).send({ id: result.id, token, status: result.status, cached: result.cached });
  });

  // ---- status -----------------------------------------------------------------
  app.get('/jobs/:id', async (req, reply) => {
    const { id } = /** @type {{ id: string }} */ (req.params);
    if (!UUID.test(id)) return reply.code(404).send({ status: 'error', message: 'not found' });
    const r = await pool.query('select id, status, error, priority, created_at from jobs where id = $1', [id]);
    if (r.rowCount === 0) return reply.code(404).send({ status: 'error', message: 'not found' });
    const job = r.rows[0];
    /** @type {number | null} */
    let queuePos = null;
    if (job.status === 'queued') {
      const q = await pool.query(
        `select count(*)::int as n from jobs
         where status = 'queued' and (priority > $1 or (priority = $1 and created_at < $2))`,
        [job.priority, job.created_at],
      );
      queuePos = q.rows[0].n;
    }
    return reply.send({ id: job.id, status: job.status, error: job.error, queuePos });
  });

  // ---- cancel -----------------------------------------------------------------
  app.delete('/jobs/:id', async (req, reply) => {
    const { id } = /** @type {{ id: string }} */ (req.params);
    const { token } = /** @type {{ token?: string }} */ (req.query);
    const auth = await authJob(id, token);
    if (auth.code) return reply.code(auth.code).send({ status: 'error', message: auth.code === 404 ? 'not found' : 'forbidden' });
    await pool.query(
      `update jobs set status = 'cancelled', ended_at = now()
       where id = $1 and status not in ('completed','cached','failed','killed')`,
      [id],
    );
    publish(id, 'status', { status: 'cancelled' });
    return reply.send({ id, status: 'cancelled' });
  });

  // ---- live SSE stream --------------------------------------------------------
  app.get('/jobs/:id/stream', async (req, reply) => {
    const { id } = /** @type {{ id: string }} */ (req.params);
    const { token } = /** @type {{ token?: string }} */ (req.query);
    const auth = await authJob(id, token);
    if (auth.code) return reply.code(auth.code).send({ status: 'error', message: auth.code === 404 ? 'not found' : 'forbidden' });

    reply.hijack();
    const raw = reply.raw;
    raw.writeHead(200, {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      Connection: 'keep-alive',
      'Access-Control-Allow-Origin': config.corsOrigin,
    });
    raw.write(': connected\n\n');
    /** @param {string} event @param {Record<string, unknown>} data */
    const write = (event, data) => raw.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);

    // initial snapshot so a late/reconnecting subscriber gets terminal state immediately
    const job = /** @type {JobRow} */ (auth.row);
    write('status', { jobid: id, status: job.status });
    if ((job.status === 'completed' || job.status === 'cached') && job.output_hash) {
      write('complete', { jobid: id, outputHash: job.output_hash, hasDetphoton: !!job.detp_hash });
    }

    const unsub = subscribe(id, (e) => write(e.event, e.data));
    const hb = setInterval(() => raw.write(': keep-alive\n\n'), 15000);
    const close = () => {
      clearInterval(hb);
      unsub();
      raw.end();
    };
    req.raw.on('close', close);
  });

  // ---- output / detected photons ---------------------------------------------
  /**
   * @param {import('fastify').FastifyRequest} req
   * @param {import('fastify').FastifyReply} reply
   * @param {'output_hash' | 'detp_hash'} column
   */
  const serveBlob = async (req, reply, column) => {
    const { id } = /** @type {{ id: string }} */ (req.params);
    const { token } = /** @type {{ token?: string }} */ (req.query);
    const auth = await authJob(id, token);
    if (auth.code) return reply.code(auth.code).send({ status: 'error', message: auth.code === 404 ? 'not found' : 'forbidden' });
    const hash = (/** @type {JobRow} */ (auth.row))[column];
    if (!hash) return reply.code(404).send({ status: 'error', message: 'not available' });
    const client = await pool.connect();
    try {
      const data = await getBlob(client, hash);
      return reply.type('application/json').send(data);
    } finally {
      client.release();
    }
  };
  app.get('/jobs/:id/output', (req, reply) => serveBlob(req, reply, 'output_hash'));
  app.get('/jobs/:id/detphoton', (req, reply) => serveBlob(req, reply, 'detp_hash'));

  // ---- worker protocol (internal, x-worker-secret) ----------------------------
  // The mcx container: GET input -> run -> PUT output [+ detphoton] -> POST complete.
  // This keeps NFS out of the loop (results are pushed, not discovered).

  /** @param {import('fastify').FastifyRequest} req @returns {boolean} */
  const isWorker = (req) => req.headers['x-worker-secret'] === config.workerSecret;

  // fetch the fully-reassembled MCX input
  app.get('/jobs/:id/input', async (req, reply) => {
    if (!isWorker(req)) return reply.code(403).send({ status: 'error', message: 'forbidden' });
    const { id } = /** @type {{ id: string }} */ (req.params);
    if (!UUID.test(id)) return reply.code(404).send({ status: 'error', message: 'not found' });
    const r = await pool.query('select input_doc from jobs where id = $1', [id]);
    if (r.rowCount === 0) return reply.code(404).send({ status: 'error', message: 'not found' });
    const client = await pool.connect();
    try {
      const doc = await reassemble(r.rows[0].input_doc, (h) => getBlob(client, h));
      return reply.type('application/json').send(doc);
    } finally {
      client.release();
    }
  });

  /**
   * @param {import('fastify').FastifyRequest} req
   * @param {import('fastify').FastifyReply} reply
   * @param {'output_hash' | 'detp_hash'} column
   */
  const uploadArtifact = async (req, reply, column) => {
    if (!isWorker(req)) return reply.code(403).send({ status: 'error', message: 'forbidden' });
    const { id } = /** @type {{ id: string }} */ (req.params);
    if (!UUID.test(id)) return reply.code(404).send({ status: 'error', message: 'not found' });
    const buf = req.body;
    if (!Buffer.isBuffer(buf) || buf.length === 0) {
      return reply.code(400).send({ status: 'error', message: 'empty body' });
    }
    await withTx(async (client) => {
      const hash = await putBlobRaw(client, buf);
      await client.query(`update jobs set ${column} = $2 where id = $1`, [id, hash]);
      await attachRefs(client, [hash], 'job', id);
    });
    return reply.code(204).send();
  };
  // raw bytes (application/octet-stream); output.jnii / detphoton are already JData JSON
  app.put('/jobs/:id/output', (req, reply) => uploadArtifact(req, reply, 'output_hash'));
  app.put('/jobs/:id/detphoton', (req, reply) => uploadArtifact(req, reply, 'detp_hash'));

  // finalize: ?error=1 -> failed; else mark completed (requires output already uploaded).
  // request body (text/plain) is the mcx log.
  app.post('/jobs/:id/complete', async (req, reply) => {
    if (!isWorker(req)) return reply.code(403).send({ status: 'error', message: 'forbidden' });
    const { id } = /** @type {{ id: string }} */ (req.params);
    if (!UUID.test(id)) return reply.code(404).send({ status: 'error', message: 'not found' });
    const q = /** @type {{ error?: string, runtime?: string }} */ (req.query);
    const log = typeof req.body === 'string' ? req.body : Buffer.isBuffer(req.body) ? req.body.toString('utf8') : null;
    const runtime = q.runtime ? Number(q.runtime) : null;

    if (q.error) {
      await pool.query(
        `update jobs set status = 'failed', error = $2, log = $3, ended_at = now() where id = $1`,
        [id, (log || 'simulation error').slice(0, 4000), log],
      );
      publish(id, 'error', { status: 'failed', message: 'simulation error' });
      return reply.code(204).send();
    }

    const r = await pool.query(
      `update jobs set status = 'completed', log = $2, runtime = $3, ended_at = now()
       where id = $1 and output_hash is not null returning output_hash, detp_hash`,
      [id, log, runtime],
    );
    if (r.rowCount === 0) return reply.code(409).send({ status: 'error', message: 'no output uploaded' });
    publish(id, 'complete', {
      outputHash: r.rows[0].output_hash,
      hasDetphoton: !!r.rows[0].detp_hash,
      runtime,
    });
    return reply.code(204).send();
  });
}
