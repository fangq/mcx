// @ts-check
// Seed curated library entries (builtin examples) from JSON files. Each file holds
// { title, description, license, submitter, doc } where doc is a complete MCX input.
// Entries are inserted pre-approved (they are maintainer-curated, unlike user shares
// which start 'pending'). Idempotent: a doc whose doc_hash already exists is skipped.
//
// Usage:
//   node --env-file=.env scripts/seed-library.js db/seeds/*.json

import { readFileSync } from 'node:fs';
import { randomUUID } from 'node:crypto';
import { pool, withTx } from '../src/db.js';
import { normalize } from '../src/jdata.js';
import { putBlob, attachRefs } from '../src/blobs.js';
import { config } from '../src/config.js';
import { validateInput } from '../src/schema.js';

const files = process.argv.slice(2);
if (!files.length) {
  console.error('usage: node scripts/seed-library.js <seed.json> [...]');
  process.exit(1);
}

let imported = 0, skipped = 0, failed = 0;

for (const file of files) {
  /** @type {{ title?: string, description?: string, license?: string, submitter?: object, doc?: object }} */
  const entry = JSON.parse(readFileSync(file, 'utf8'));
  if (!entry.title || !entry.description || !entry.license || !entry.doc) {
    failed++;
    console.warn(`skip (missing title/description/license/doc): ${file}`);
    continue;
  }
  if (!validateInput(entry.doc)) {
    failed++;
    console.warn(`skip (schema validation failed): ${file}`, validateInput.errors);
    continue;
  }

  try {
    await withTx(async (client) => {
      const norm = await normalize(entry.doc, (canon, enc, size) => putBlob(client, canon, enc, size), config.threshold);

      const dup = await client.query('select 1 from library where doc_hash = $1 limit 1', [norm.docHash]);
      if (dup.rowCount) { skipped++; console.log(`skip (already in library): ${entry.title}`); return; }

      const ins = await client.query(
        `insert into library
           (id, title, description, license, submitter, input_doc, doc_hash, status)
         values ($1,$2,$3,$4,$5,$6,$7,'approved') returning id`,
        [randomUUID(), entry.title, entry.description, entry.license, entry.submitter ?? null, norm.doc, norm.docHash],
      );
      await attachRefs(client, [...norm.refs], 'library', ins.rows[0].id);
      imported++;
      console.log(`imported: ${entry.title} (${ins.rows[0].id})`);
    });
  } catch (err) {
    failed++;
    console.warn(`failed: ${entry.title} — ${(/** @type {Error} */ (err)).message}`);
  }
}

await pool.end();
console.log(`done: imported=${imported} skipped=${skipped} failed=${failed} of ${files.length}`);
