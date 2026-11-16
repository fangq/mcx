// @ts-check

/**
 * The connecting client's IP, used to throttle the endpoints that cost real resources.
 *
 * Apache appends the connecting client's IP to X-Forwarded-For, so the LAST entry is the
 * real client (a client-supplied X-Forwarded-For only adds spoofed entries before it).
 * @param {import('fastify').FastifyRequest} req
 * @returns {string}
 */
export function clientIp(req) {
  const xff = /** @type {string} */ (req.headers['x-forwarded-for'] || '');
  const parts = xff.split(',').map((s) => s.trim()).filter(Boolean);
  return parts[parts.length - 1] || req.ip || 'unknown';
}
