// @ts-check
import { createHash } from 'node:crypto';

/**
 * sha256 hex digest of a string or buffer.
 * @param {string | Buffer} input
 * @returns {string}
 */
export function sha256hex(input) {
  return createHash('sha256').update(input).digest('hex');
}

/**
 * Deterministic JSON serialization: object keys sorted lexicographically, no
 * insignificant whitespace. Identical content -> identical bytes -> identical hash,
 * regardless of incoming key order (see contracts/normalization.md §2).
 * @param {unknown} value
 * @returns {string}
 */
export function stableStringify(value) {
  if (value === null || typeof value !== 'object') return JSON.stringify(value) ?? 'null';
  if (Array.isArray(value)) return '[' + value.map(stableStringify).join(',') + ']';
  const obj = /** @type {Record<string, unknown>} */ (value);
  const keys = Object.keys(obj).sort();
  return '{' + keys.map((k) => JSON.stringify(k) + ':' + stableStringify(obj[k])).join(',') + '}';
}
