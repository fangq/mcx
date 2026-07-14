// @ts-check
import { readdirSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { pool } from './db.js';

// Apply db/migrations/*.sql in filename order. Idempotent (migrations use IF NOT EXISTS).
const dir = fileURLToPath(new URL('../db/migrations/', import.meta.url));
const files = readdirSync(dir).filter((f) => f.endsWith('.sql')).sort();

for (const f of files) {
  process.stdout.write(`applying ${f} ... `);
  await pool.query(readFileSync(dir + f, 'utf8'));
  process.stdout.write('ok\n');
}
await pool.end();
console.log('migrations done');
