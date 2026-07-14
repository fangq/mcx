import type { FastifyInstance, FastifyReply, FastifyRequest } from 'fastify';
import { config } from '../config.ts';
import { pool, withTx } from '../db.ts';
import { attachRefs, getBlob, putBlob } from '../blobs.ts';
import { normalize } from '../jdata.ts';
import { stableStringify } from '../hash.ts';
import { mintToken, tokenHash } from '../tokens.ts';
import { validateInput, checkLimits } from '../schema.ts';
import { enqueueJob } from '../queue.ts';
import { publish, subscribe } from '../sse.ts';

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
const TERMINAL = new Set(['completed', 'cached', 'failed', 'cancelled', 'killed', 'invalid']);

interface JobRow {
  id: string;
  status: string;
  token_hash: string;
  output_hash: string | null;
  detp_hash: string | null;
  error: string | null;
}

/** Returns {row} on match, {status:404|403} otherwise. */
async function authJob(id: string, token: string | undefined): Promise<{ row?: JobRow; code?: number }> {
  if (!UUID.test(id)) return { code: 404 };
  const r = await pool.query(
    'select id, status, token_hash, output_hash, detp_hash, error from jobs where id = $1',
    [id],
  );
  if (r.rowCount === 0) return { code: 404 };
  const row = r.rows[0] as JobRow;
  if (!token || row.token_hash !== tokenHash(token)) return { code: 403 };
  return { row };
}

export async function jobRoutes(app: FastifyInstance): Promise<void> {
  // ---- submit -----------------------------------------------------------------
  app.post('/jobs', async (req, reply) => {
    const body = req.body as { doc?: unknown; user?: Record<string, unknown> };
    if (!body?.doc || typeof body.doc !== 'object') {
      return reply.code(422).send({ status: 'invalid', message: 'missing doc' });
    }
    if (!validateInput(body.doc)) {
      return reply
        .code(422)
        .send({ status: 'invalid', message: 'schema validation failed', errors: validateInput.errors });
    }
    const limit = checkLimits(body.doc as Record<string, unknown>);
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
        const id = ins.rows[0].id as string;
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
      const id = ins.rows[0].id as string;
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
    const { id } = req.params as { id: string };
    if (!UUID.test(id)) return reply.code(404).send({ status: 'error', message: 'not found' });
    const r = await pool.query('select id, status, error, priority, created_at from jobs where id = $1', [id]);
    if (r.rowCount === 0) return reply.code(404).send({ status: 'error', message: 'not found' });
    const job = r.rows[0];
    let queuePos: number | null = null;
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
    const { id } = req.params as { id: string };
    const { token } = req.query as { token?: string };
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
  app.get('/jobs/:id/stream', async (req: FastifyRequest, reply: FastifyReply) => {
    const { id } = req.params as { id: string };
    const { token } = req.query as { token?: string };
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
    const write = (event: string, data: Record<string, unknown>) =>
      raw.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);

    // initial snapshot so a late/reconnecting subscriber gets terminal state immediately
    const job = auth.row!;
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
  const serveBlob = async (
    req: FastifyRequest,
    reply: FastifyReply,
    column: 'output_hash' | 'detp_hash',
  ) => {
    const { id } = req.params as { id: string };
    const { token } = req.query as { token?: string };
    const auth = await authJob(id, token);
    if (auth.code) return reply.code(auth.code).send({ status: 'error', message: auth.code === 404 ? 'not found' : 'forbidden' });
    const hash = auth.row![column];
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

  // ---- worker completion callback (internal) ----------------------------------
  app.post('/jobs/:id/complete', async (req, reply) => {
    if (req.headers['x-worker-secret'] !== config.workerSecret) {
      return reply.code(403).send({ status: 'error', message: 'forbidden' });
    }
    const { id } = req.params as { id: string };
    if (!UUID.test(id)) return reply.code(404).send({ status: 'error', message: 'not found' });
    const body = req.body as {
      output?: unknown;
      detphoton?: unknown;
      log?: string;
      runtime?: number;
      error?: string | null;
    };

    if (body.error) {
      await pool.query(
        `update jobs set status = 'failed', error = $2, log = $3, ended_at = now() where id = $1`,
        [id, body.error, body.log ?? null],
      );
      publish(id, 'error', { status: 'failed', message: body.error });
      return reply.code(204).send();
    }

    await withTx(async (client) => {
      const outCanon = stableStringify(body.output);
      const outHash = await putBlob(client, outCanon, null, Buffer.byteLength(outCanon, 'utf8'));
      const owned = [outHash];
      let detpHash: string | null = null;
      if (body.detphoton !== undefined && body.detphoton !== null) {
        const dCanon = stableStringify(body.detphoton);
        detpHash = await putBlob(client, dCanon, null, Buffer.byteLength(dCanon, 'utf8'));
        owned.push(detpHash);
      }
      await client.query(
        `update jobs set status = 'completed', output_hash = $2, detp_hash = $3,
         log = $4, runtime = $5, ended_at = now() where id = $1`,
        [id, outHash, detpHash, body.log ?? null, body.runtime ?? null],
      );
      await attachRefs(client, owned, 'job', id);
      publish(id, 'complete', { outputHash: outHash, hasDetphoton: !!detpHash, runtime: body.runtime ?? null });
    });
    return reply.code(204).send();
  });
}
