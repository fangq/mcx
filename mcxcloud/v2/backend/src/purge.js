// @ts-check
import { withTx } from './db.js';

/**
 * Purge terminal, non-library jobs older than maxAgeMs: drop their heavy/content columns
 * (input doc, hashes, log) and GC any blobs left with no references, but KEEP the row —
 * id, submitter, engine, status, error, runtime, and timestamps survive forever as a usage
 * record (e.g. for funding-agency reporting: who ran how many jobs, from which institution,
 * over time) without retaining the simulation content itself. A job is exempt (kept in
 * full, permanently) if its doc_hash matches a `library` entry — running a shared library
 * simulation caches its result forever, while one-off jobs are blanked after the TTL.
 * (Blobs still referenced by a surviving job/library are never deleted, thanks to the
 * refcount edges in blob_refs.) `input_doc is not null` in the selection makes this
 * idempotent — an already-blanked row is never reselected on a later run.
 * @param {number} maxAgeMs
 * @returns {Promise<number>} number of jobs blanked
 */
export async function purgeOldJobs(maxAgeMs) {
  const cutoff = new Date(Date.now() - maxAgeMs);
  return withTx(async (client) => {
    const sel = await client.query(
      `select id from jobs
       where status in ('completed','cached','failed','killed','cancelled')
         and coalesce(ended_at, created_at) < $1
         and input_doc is not null
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
    // 2. blank the content columns FIRST (releases the output_hash/detp_hash foreign
    // keys) ... token_hash is cleared too: with no output left to protect, an old token
    // should no longer resolve to anything (authJob compares against token_hash, and a
    // provided token's hash can never equal null)
    await client.query(
      `update jobs set input_doc = null, doc_hash = null, output_hash = null,
         detp_hash = null, log = null, token_hash = null
       where id = any($1::uuid[])`,
      [ids],
    );
    // 3. ... then GC any blobs now unreferenced
    await client.query('delete from blobs where refcount <= 0');
    return ids.length;
  });
}
