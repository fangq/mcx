// @ts-check
// One-time import of the v1 shared-simulation library (SQLite `mcxpub` table) into the
// v2 Postgres `library` table, normalizing each doc (heavy Shapes -> content-addressed
// blobs). The transient v1 job queue (`mcxcloud`) is NOT migrated.
//
// Usage:
//   sqlite3 -json /var/lib/mcxcloud/db/mcxcloud.db \
//     "select time,title,comment,license,name,inst,email,netname,json,thumbnail,\
//      upvote,downvote,readcount,runcount,createtime from mcxpub" > mcxpub.json
//   node scripts/migrate-v1-library.js mcxpub.json
//
// Idempotent-ish: skips rows whose doc_hash already exists in `library`.

import { readFileSync } from 'node:fs';
import { pool, withTx } from '../src/db.js';
import { normalize } from '../src/jdata.js';
import { putBlob, putBlobRaw, attachRefs } from '../src/blobs.js';
import { config } from '../src/config.js';

const file = process.argv[2];
if (!file) {
  console.error('usage: node scripts/migrate-v1-library.js <mcxpub.json>');
  process.exit(1);
}

/** @type {any[]} */
const rows = JSON.parse(readFileSync(file, 'utf8'));
let imported = 0, skipped = 0, failed = 0;

for (const row of rows) {
  if (!row.json || String(row.json).trim().length < 3) { skipped++; continue; } // dedup placeholder
  let doc;
  try {
    doc = JSON.parse(row.json);
  } catch {
    failed++;
    console.warn(`skip (bad json): ${row.title}`);
    continue;
  }

  try {
    await withTx(async (client) => {
      const norm = await normalize(doc, (canon, enc, size) => putBlob(client, canon, enc, size), config.threshold);

      const dup = await client.query('select 1 from library where doc_hash = $1 limit 1', [norm.docHash]);
      if (dup.rowCount) { skipped++; return; }

      const owned = [...norm.refs];
      let thumbHash = null;
      if (row.thumbnail) {
        thumbHash = await putBlobRaw(client, Buffer.from(String(row.thumbnail), 'utf8'));
        owned.push(thumbHash);
      }
      const submitter = { fullname: row.name, email: row.email, inst: row.inst, netname: row.netname };
      const created = row.createtime ? new Date(Number(row.createtime) * 1000) : new Date();
      const ins = await client.query(
        `insert into library
           (title, description, license, submitter, input_doc, doc_hash, thumbnail_hash,
            upvotes, downvotes, read_count, run_count, created_at)
         values ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12) returning id`,
        [
          row.title || '(untitled)', row.comment || '', row.license || 'CC0', submitter,
          norm.doc, norm.docHash, thumbHash,
          row.upvote || 0, row.downvote || 0, row.readcount || 0, row.runcount || 0, created,
        ],
      );
      await attachRefs(client, owned, 'library', ins.rows[0].id);
      imported++;
    });
  } catch (err) {
    failed++;
    console.warn(`failed: ${row.title} — ${(/** @type {Error} */ (err)).message}`);
  }
}

await pool.end();
console.log(`done: imported=${imported} skipped=${skipped} failed=${failed} of ${rows.length}`);
