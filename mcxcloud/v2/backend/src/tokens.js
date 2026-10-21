// @ts-check
import { randomBytes } from 'node:crypto';
import { sha256hex } from './hash.js';

/**
 * Mint a per-job capability token; store only its hash.
 * @returns {{ token: string, tokenHash: string }}
 */
export function mintToken() {
  const token = randomBytes(24).toString('hex');
  return { token, tokenHash: sha256hex(token) };
}

/** @param {string} token @returns {string} */
export function tokenHash(token) {
  return sha256hex(token);
}
