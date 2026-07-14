import type { FastifyInstance } from 'fastify';
import { config } from '../config.ts';
import { pool, withTx } from '../db.ts';
import { attachRefs, getBlob, putBlob } from '../blobs.ts';
import { normalize, reassemble } from '../jdata.ts';
import { validateInput } from '../schema.ts';

const LICENSES = new Set(['CC0', 'CC-BY', 'CC-BY-SA']);
const blobUrl = (hash: string | null): string | null =>
  hash ? '/blobs/' + hash : null; // hash is 'sha256/<hex>'

export async function libraryRoutes(app: FastifyInstance): Promise<void> {
  // ---- browse / search --------------------------------------------------------
  app.get('/library', async (req, reply) => {
    const { q, limit, offset } = req.query as { q?: string; limit?: string; offset?: string };
    const lim = Math.min(Number(limit ?? 20) || 20, 100);
    const off = Number(offset ?? 0) || 0;
    const rows = q
      ? await pool.query(
          `select id, title, description, license, thumbnail_hash, run_count from library
           where to_tsvector('english', coalesce(title,'') || ' ' || coalesce(description,''))
                 @@ plainto_tsquery('english', $1)
           order by run_count desc limit $2 offset $3`,
          [q, lim, off],
        )
      : await pool.query(
          `select id, title, description, license, thumbnail_hash, run_count from library
           order by run_count desc limit $1 offset $2`,
          [lim, off],
        );
    return reply.send(
      rows.rows.map((r) => ({
        id: r.id,
        title: r.title,
        description: r.description,
        license: r.license,
        thumbnail: blobUrl(r.thumbnail_hash),
        runCount: r.run_count,
      })),
    );
  });

  // ---- load one ---------------------------------------------------------------
  app.get('/library/:id', async (req, reply) => {
    const { id } = req.params as { id: string };
    const r = await pool.query(
      'select id, title, description, license, input_doc from library where id = $1',
      [id],
    );
    if (r.rowCount === 0) return reply.code(404).send({ status: 'error', message: 'not found' });
    const row = r.rows[0];
    const client = await pool.connect();
    let doc: unknown;
    try {
      doc = await reassemble(row.input_doc, (h) => getBlob(client, h));
    } finally {
      client.release();
    }
    await pool.query('update library set read_count = read_count + 1 where id = $1', [id]);
    return reply.send({ id: row.id, title: row.title, description: row.description, license: row.license, doc });
  });

  // ---- share ------------------------------------------------------------------
  app.post('/library', async (req, reply) => {
    const body = req.body as {
      title?: string;
      description?: string;
      license?: string;
      thumbnail?: string;
      doc?: unknown;
      user?: Record<string, unknown>;
    };
    if (!body?.title || !body?.description || !body?.license || !body?.doc) {
      return reply.code(422).send({ status: 'invalid', message: 'missing required fields' });
    }
    if (!LICENSES.has(body.license)) {
      return reply.code(422).send({ status: 'invalid', message: 'unknown license' });
    }
    if (!validateInput(body.doc)) {
      return reply
        .code(422)
        .send({ status: 'invalid', message: 'schema validation failed', errors: validateInput.errors });
    }

    const out = await withTx(async (client) => {
      const { doc, docHash, refs } = await normalize(
        body.doc,
        (canon, enc, size) => putBlob(client, canon, enc, size),
        config.threshold,
      );
      const owned = [...refs];
      let thumbHash: string | null = null;
      if (body.thumbnail) {
        thumbHash = await putBlob(client, body.thumbnail, null, Buffer.byteLength(body.thumbnail, 'utf8'));
        owned.push(thumbHash);
      }
      const ins = await client.query(
        `insert into library (title, description, license, submitter, input_doc, doc_hash, thumbnail_hash)
         values ($1,$2,$3,$4,$5,$6,$7) returning id`,
        [body.title, body.description, body.license, body.user ?? null, doc, docHash, thumbHash],
      );
      const id = ins.rows[0].id as string;
      await attachRefs(client, owned, 'library', id);
      return { id, hash: docHash };
    });
    return reply.code(201).send(out);
  });
}
