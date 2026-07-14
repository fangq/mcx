import type { PoolClient } from 'pg';
import { sha256hex } from './hash.ts';

/** Store a canonical blob (idempotent by content hash). Returns 'sha256/<hex>'. */
export async function putBlob(
  client: PoolClient,
  canon: string,
  encoding: string | null,
  size: number,
): Promise<string> {
  const hash = 'sha256/' + sha256hex(canon);
  await client.query(
    `insert into blobs (hash, size, encoding, refcount, data)
     values ($1, $2, $3, 0, $4)
     on conflict (hash) do nothing`,
    [hash, size, encoding, Buffer.from(canon, 'utf8')],
  );
  return hash;
}

export async function getBlob(client: PoolClient, hash: string): Promise<string> {
  const r = await client.query('select data from blobs where hash = $1', [hash]);
  if (r.rowCount === 0) throw new Error('missing blob ' + hash);
  return (r.rows[0].data as Buffer).toString('utf8');
}

/** Attach blob hashes to an owner; bump refcount only for edges that are new. */
export async function attachRefs(
  client: PoolClient,
  hashes: string[],
  ownerKind: 'job' | 'library',
  ownerId: string,
): Promise<void> {
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

/** Drop all edges from an owner, decrement refcounts, and GC orphaned blobs. */
export async function detachOwner(
  client: PoolClient,
  ownerKind: 'job' | 'library',
  ownerId: string,
): Promise<void> {
  const r = await client.query(
    'delete from blob_refs where owner_kind = $1 and owner_id = $2 returning hash',
    [ownerKind, ownerId],
  );
  for (const row of r.rows) {
    await client.query('update blobs set refcount = refcount - 1 where hash = $1', [row.hash]);
  }
  await client.query('delete from blobs where refcount <= 0');
}
