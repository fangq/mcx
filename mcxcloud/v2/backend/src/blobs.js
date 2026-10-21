// @ts-check
import { sha256hex } from './hash.js';

/**
 * Store a canonical blob (idempotent by content hash). Returns 'sha256/<hex>'.
 * @param {import('pg').PoolClient} client
 * @param {string} canon
 * @param {string | null} encoding
 * @param {number} size
 * @returns {Promise<string>}
 */
export async function putBlob(client, canon, encoding, size) {
  const hash = 'sha256/' + sha256hex(canon);
  await client.query(
    `insert into blobs (hash, size, encoding, refcount, data)
     values ($1, $2, $3, 0, $4)
     on conflict (hash) do nothing`,
    [hash, size, encoding, Buffer.from(canon, 'utf8')],
  );
  return hash;
}

/**
 * Store raw bytes as a blob (idempotent by content hash). Used for worker-uploaded
 * outputs, which are already-serialized JData/JNIfTI and need no canonicalization.
 * @param {import('pg').PoolClient} client
 * @param {Buffer} buf
 * @returns {Promise<string>}
 */
export async function putBlobRaw(client, buf) {
  const hash = 'sha256/' + sha256hex(buf);
  await client.query(
    `insert into blobs (hash, size, encoding, refcount, data)
     values ($1, $2, null, 0, $3)
     on conflict (hash) do nothing`,
    [hash, buf.length, buf],
  );
  return hash;
}

/**
 * @param {import('pg').PoolClient} client
 * @param {string} hash
 * @returns {Promise<string>}
 */
export async function getBlob(client, hash) {
  const r = await client.query('select data from blobs where hash = $1', [hash]);
  if (r.rowCount === 0) throw new Error('missing blob ' + hash);
  return (/** @type {Buffer} */ (r.rows[0].data)).toString('utf8');
}

/**
 * Attach blob hashes to an owner; bump refcount only for edges that are new.
 * @param {import('pg').PoolClient} client
 * @param {string[]} hashes
 * @param {'job' | 'library'} ownerKind
 * @param {string} ownerId
 * @returns {Promise<void>}
 */
export async function attachRefs(client, hashes, ownerKind, ownerId) {
  for (const hash of hashes) {
    const r = await client.query(
      `insert into blob_refs (hash, owner_kind, owner_id) values ($1, $2, $3)
       on conflict do nothing`,
      [hash, ownerKind, ownerId],
    );
    if (r.rowCount === 1) {
      await client.query('update blobs set refcount = refcount + 1 where hash = $1', [hash]);
    }
  }
}

/**
 * Drop all edges from an owner, decrement refcounts, and GC orphaned blobs.
 * @param {import('pg').PoolClient} client
 * @param {'job' | 'library'} ownerKind
 * @param {string} ownerId
 * @returns {Promise<void>}
 */
export async function detachOwner(client, ownerKind, ownerId) {
  const r = await client.query(
    'delete from blob_refs where owner_kind = $1 and owner_id = $2 returning hash',
    [ownerKind, ownerId],
  );
  for (const row of r.rows) {
    await client.query('update blobs set refcount = refcount - 1 where hash = $1', [row.hash]);
  }
  await client.query('delete from blobs where refcount <= 0');
}
