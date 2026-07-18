// @ts-check
import { withTx } from './db.js';

/**
 * Purge terminal, non-library jobs older than maxAgeMs and GC any blobs left with no
 * references. A job is kept (permanent cache) if its doc_hash matches a `library` entry —
 * so running a shared library simulation caches its result forever, while one-off jobs are
 * cleaned after the TTL. (Blobs still referenced by a surviving job/library are never
 * deleted, thanks to the refcount edges in blob_refs.)
 * @param {number} maxAgeMs
 * @returns {Promise<number>} number of jobs purged
 */
export async function purgeOldJobs(maxAgeMs) {
  const cutoff = new Date(Date.now() - maxAgeMs);
  return withTx(async (client) => {
    const sel = await client.query(
      `select id from jobs
       where status in ('completed','cached','failed','killed','cancelled')
         and coalesce(ended_at, created_at) < $1
         and not exists (select 1 from library l where l.doc_hash = jobs.doc_hash)`,
      [cutoff],
    );
    if (sel.rowCount === 0) return 0;
    const ids = sel.rows.map((r) => r.id);
    // 1. drop these jobs' blob edges and decrement the referenced blobs' refcounts
    await client.query(
      `with removed as (
         delete from blob_refs where owner_kind='job' and owner_id = any($1::uuid[]) returning hash
       ), counts as (
         select hash, count(*)::int as n from removed group by hash
       )
       update blobs b set refcount = b.refcount - c.n from counts c where b.hash = c.hash`,
      [ids],
    );
    // 2. delete the jobs FIRST (releases the output_hash/detp_hash foreign keys) ...
    await client.query('delete from jobs where id = any($1::uuid[])', [ids]);
    // 3. ... then GC any blobs now unreferenced
    await client.query('delete from blobs where refcount <= 0');
    return ids.length;
  });
}
