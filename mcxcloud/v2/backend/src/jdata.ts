import { sha256hex, stableStringify } from './hash.ts';

// Content-addressed normalization & reassembly — see contracts/normalization.md.

export const DEFAULT_THRESHOLD = 4096;
const CAS = 'cas:'; // internal content-addressed-store URI scheme

/** A JData packed/annotated N-D array node (the heavy fields we extract). */
export function isDataNode(node: unknown): node is Record<string, unknown> {
  if (typeof node !== 'object' || node === null || Array.isArray(node)) return false;
  return '_ArrayZipData_' in node || '_ArrayData_' in node;
}

/** If node is exactly a `_DataLink_` reference in our `cas:` scheme, return the blob hash. */
function linkHash(node: unknown): string | null {
  if (typeof node !== 'object' || node === null || Array.isArray(node)) return null;
  const keys = Object.keys(node);
  if (keys.length !== 1 || keys[0] !== '_DataLink_') return null;
  const v = (node as Record<string, unknown>)['_DataLink_'];
  if (typeof v === 'string' && v.startsWith(CAS)) return v.slice(CAS.length); // 'sha256/<hex>'
  return null;
}

export interface NormalizeResult {
  doc: unknown;
  docHash: string; // 'sha256/<hex>' over the normalized doc
  refs: string[]; // unique blob hashes referenced
}

/**
 * Extract every packed-array node ≥ threshold into blobs (via putBlob) and replace it
 * with a `_DataLink_` ref. putBlob returns the blob hash 'sha256/<hex>'.
 */
export async function normalize(
  input: unknown,
  putBlob: (canon: string, encoding: string | null, size: number) => Promise<string>,
  threshold: number = DEFAULT_THRESHOLD,
): Promise<NormalizeResult> {
  const refs: string[] = [];

  async function walk(node: unknown): Promise<unknown> {
    if (Array.isArray(node)) {
      const out: unknown[] = [];
      for (const item of node) out.push(await walk(item));
      return out;
    }
    if (node !== null && typeof node === 'object') {
      if (isDataNode(node)) {
        const canon = stableStringify(node);
        const size = Buffer.byteLength(canon, 'utf8');
        if (size >= threshold) {
          const enc = (node as Record<string, unknown>)['_ArrayZipType_'];
          const hash = await putBlob(canon, typeof enc === 'string' ? enc : null, size);
          refs.push(hash);
          return { _DataLink_: CAS + hash };
        }
        return node;
      }
      const out: Record<string, unknown> = {};
      for (const [k, v] of Object.entries(node)) out[k] = await walk(v);
      return out;
    }
    return node;
  }

  const doc = await walk(input);
  const docHash = 'sha256/' + sha256hex(stableStringify(doc));
  return { doc, docHash, refs: [...new Set(refs)] };
}

/** Inverse of normalize: resolve every `_DataLink_` back to its stored node. */
export async function reassemble(
  input: unknown,
  getBlob: (hash: string) => Promise<string>,
): Promise<unknown> {
  async function walk(node: unknown): Promise<unknown> {
    const h = linkHash(node);
    if (h !== null) return JSON.parse(await getBlob(h));
    if (Array.isArray(node)) {
      const out: unknown[] = [];
      for (const item of node) out.push(await walk(item));
      return out;
    }
    if (node !== null && typeof node === 'object') {
      const out: Record<string, unknown> = {};
      for (const [k, v] of Object.entries(node)) out[k] = await walk(v);
      return out;
    }
    return node;
  }
  return walk(input);
}
