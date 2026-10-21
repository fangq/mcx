// @ts-check
import { sha256hex, stableStringify } from './hash.js';

// Content-addressed normalization & reassembly — see contracts/normalization.md.

export const DEFAULT_THRESHOLD = 4096;
const CAS = 'cas:'; // internal content-addressed-store URI scheme

/**
 * A JData packed/annotated N-D array node (the heavy fields we extract).
 * @param {unknown} node
 * @returns {boolean}
 */
export function isDataNode(node) {
  if (typeof node !== 'object' || node === null || Array.isArray(node)) return false;
  return '_ArrayZipData_' in node || '_ArrayData_' in node;
}

/**
 * If node is exactly a `_DataLink_` reference in our `cas:` scheme, return the blob hash.
 * @param {unknown} node
 * @returns {string | null}
 */
function linkHash(node) {
  if (typeof node !== 'object' || node === null || Array.isArray(node)) return null;
  const keys = Object.keys(node);
  if (keys.length !== 1 || keys[0] !== '_DataLink_') return null;
  const v = /** @type {Record<string, unknown>} */ (node)['_DataLink_'];
  if (typeof v === 'string' && v.startsWith(CAS)) return v.slice(CAS.length); // 'sha256/<hex>'
  return null;
}

/**
 * @typedef {Object} NormalizeResult
 * @property {unknown} doc            normalized document
 * @property {string} docHash         'sha256/<hex>' over the normalized doc
 * @property {string[]} refs          unique blob hashes referenced
 */

/**
 * Extract every packed-array node >= threshold into blobs (via putBlob) and replace it
 * with a `_DataLink_` ref. putBlob returns the blob hash 'sha256/<hex>'.
 * @param {unknown} input
 * @param {(canon: string, encoding: string | null, size: number) => Promise<string>} putBlob
 * @param {number} [threshold]
 * @returns {Promise<NormalizeResult>}
 */
export async function normalize(input, putBlob, threshold = DEFAULT_THRESHOLD) {
  /** @type {string[]} */
  const refs = [];

  /** @param {unknown} node @returns {Promise<unknown>} */
  async function walk(node) {
    if (Array.isArray(node)) {
      const out = [];
      for (const item of node) out.push(await walk(item));
      return out;
    }
    if (node !== null && typeof node === 'object') {
      if (isDataNode(node)) {
        const canon = stableStringify(node);
        const size = Buffer.byteLength(canon, 'utf8');
        if (size >= threshold) {
          const enc = /** @type {Record<string, unknown>} */ (node)['_ArrayZipType_'];
          const hash = await putBlob(canon, typeof enc === 'string' ? enc : null, size);
          refs.push(hash);
          return { _DataLink_: CAS + hash };
        }
        return node;
      }
      /** @type {Record<string, unknown>} */
      const out = {};
      for (const [k, v] of Object.entries(node)) out[k] = await walk(v);
      return out;
    }
    return node;
  }

  const doc = await walk(input);
  const docHash = 'sha256/' + sha256hex(stableStringify(doc));
  return { doc, docHash, refs: [...new Set(refs)] };
}

/**
 * Inverse of normalize: resolve every `_DataLink_` back to its stored node.
 * @param {unknown} input
 * @param {(hash: string) => Promise<string>} getBlob
 * @returns {Promise<unknown>}
 */
export async function reassemble(input, getBlob) {
  /** @param {unknown} node @returns {Promise<unknown>} */
  async function walk(node) {
    const h = linkHash(node);
    if (h !== null) return JSON.parse(await getBlob(h));
    if (Array.isArray(node)) {
      const out = [];
      for (const item of node) out.push(await walk(item));
      return out;
    }
    if (node !== null && typeof node === 'object') {
      /** @type {Record<string, unknown>} */
      const out = {};
      for (const [k, v] of Object.entries(node)) out[k] = await walk(v);
      return out;
    }
    return node;
  }
  return walk(input);
}
