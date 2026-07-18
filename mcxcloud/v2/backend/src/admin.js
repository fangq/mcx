// @ts-check
// Admin (library-review) authentication. The workflow is deliberately two-step so the raw
// secret is transmitted only once: the admin exchanges ADMIN_SECRET for a short-lived signed
// session token (mintAdminToken), and every subsequent admin call carries that token instead.
// Nothing here is stored on disk or shipped to the frontend; ADMIN_SECRET lives only in the
// server .env. An empty ADMIN_SECRET disables the whole admin API (safe default).
import { createHash, createHmac, timingSafeEqual } from 'node:crypto';
import { config } from './config.js';

const SESSION_TTL_MS = 60 * 60 * 1000; // admin session token lifetime (1 h)

/**
 * Constant-time check of a presented secret against ADMIN_SECRET. Both sides are hashed to a
 * fixed 32-byte digest first, so neither the comparison nor the length leaks timing.
 * @param {unknown} secret
 * @returns {boolean}
 */
export function checkAdminSecret(secret) {
  if (!config.adminSecret) return false;
  const h = (/** @type {unknown} */ s) => createHash('sha256').update(String(s ?? '')).digest();
  return timingSafeEqual(h(secret), h(config.adminSecret));
}

/**
 * Mint a short-lived signed session token: base64url(payload).base64url(hmac).
 * @returns {string}
 */
export function mintAdminToken() {
  const payload = Buffer.from(JSON.stringify({ exp: Date.now() + SESSION_TTL_MS })).toString('base64url');
  const sig = createHmac('sha256', config.adminSecret).update(payload).digest('base64url');
  return `${payload}.${sig}`;
}

/**
 * Verify a session token's HMAC signature and expiry.
 * @param {unknown} token
 * @returns {boolean}
 */
export function verifyAdminToken(token) {
  if (!config.adminSecret) return false;
  const [payload, sig] = String(token ?? '').split('.');
  if (!payload || !sig) return false;
  const expected = createHmac('sha256', config.adminSecret).update(payload).digest('base64url');
  const a = Buffer.from(sig);
  const b = Buffer.from(expected);
  if (a.length !== b.length || !timingSafeEqual(a, b)) return false;
  try {
    const p = JSON.parse(Buffer.from(payload, 'base64url').toString('utf8'));
    return typeof p.exp === 'number' && p.exp > Date.now();
  } catch {
    return false;
  }
}

export { SESSION_TTL_MS };
