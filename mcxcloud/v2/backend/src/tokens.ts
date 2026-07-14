import { randomBytes } from 'node:crypto';
import { sha256hex } from './hash.ts';

/** Mint a per-job capability token; store only its hash. */
export function mintToken(): { token: string; tokenHash: string } {
  const token = randomBytes(24).toString('hex');
  return { token, tokenHash: sha256hex(token) };
}

export function tokenHash(token: string): string {
  return sha256hex(token);
}
