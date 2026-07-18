// @ts-check
// All server communication: fetch (REST) + EventSource (SSE). No JSONP, no polling.

const API = /** @type {any} */ (window).MCX_API_BASE || '/api';

/** @param {string} path @param {RequestInit} [opts] */
async function req(path, opts) {
  const r = await fetch(API + path, opts);
  if (!r.ok) {
    let msg = `${r.status}`;
    try {
      const j = await r.json();
      msg = j.message || msg;
    } catch { /* non-json error */ }
    throw new Error(msg);
  }
  return r;
}

/** The authoritative MCX input schema (drives the editor). @returns {Promise<object>} */
export async function fetchSchema() {
  return (await req('/schema/mcx-input.v1')).json();
}

/**
 * Submit a simulation.
 * @param {object} doc @param {object} user
 * @returns {Promise<{ id: string, token: string, status: string, cached: boolean }>}
 */
export async function submitJob(doc, user) {
  return (
    await req('/jobs', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ doc, user }),
    })
  ).json();
}

/** @param {string} id @param {string} token */
export async function cancelJob(id, token) {
  await req(`/jobs/${id}?token=${encodeURIComponent(token)}`, { method: 'DELETE' });
}

/** @param {string} id @param {string} token @returns {Promise<object>} */
export async function fetchOutput(id, token) {
  return (await req(`/jobs/${id}/output?token=${encodeURIComponent(token)}`)).json();
}

/** @param {string} id @param {string} token @returns {Promise<object>} */
export async function fetchDetphoton(id, token) {
  return (await req(`/jobs/${id}/detphoton?token=${encodeURIComponent(token)}`)).json();
}

/**
 * Open the live job stream. Returns a close() function.
 * @param {string} id @param {string} token
 * @param {(event: string, data: any) => void} onEvent
 * @returns {() => void}
 */
export function streamJob(id, token, onEvent) {
  const es = new EventSource(`${API}/jobs/${id}/stream?token=${encodeURIComponent(token)}`);
  for (const type of ['status', 'log', 'progress', 'complete', 'error']) {
    es.addEventListener(type, (/** @type {MessageEvent} */ e) => {
      let data = {};
      try { data = JSON.parse(e.data); } catch { /* ignore */ }
      onEvent(type, data);
      if (type === 'complete' || type === 'error') es.close();
    });
  }
  es.onerror = () => {}; // browser auto-reconnects; terminal events already close it
  return () => es.close();
}

/**
 * @param {string} [q] @param {number} [limit] @param {number} [offset]
 * @returns {Promise<Array<object>>}
 */
export async function searchLibrary(q, limit = 20, offset = 0) {
  const qs = new URLSearchParams({ limit: String(limit), offset: String(offset) });
  if (q) qs.set('q', q);
  return (await req('/library?' + qs.toString())).json();
}

/** @param {string} id @returns {Promise<{ id: string, title: string, description: string, license: string, doc: object }>} */
export async function loadLibraryEntry(id) {
  return (await req('/library/' + id)).json();
}

/** @param {object} entry @returns {Promise<{ id: string, hash: string }>} */
export async function shareLibrary(entry) {
  return (
    await req('/library', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(entry),
    })
  ).json();
}

/** absolute URL for a /blobs or thumbnail path returned by the API */
export function assetUrl(path) {
  if (!path) return '';
  return /^https?:|^data:/.test(path) ? path : API + path;
}

// ---- admin / library review -------------------------------------------------------------
// The admin session token lives ONLY in this module variable (memory) — never localStorage,
// a cookie, or the URL. It is gone on reload. The raw ADMIN_SECRET is sent exactly once, to
// /admin/login, in exchange for this short-lived token.
let adminToken = '';

/** @returns {boolean} whether an admin session is active */
export function isAdmin() {
  return !!adminToken;
}

/** Drop the in-memory admin session. */
export function adminLogout() {
  adminToken = '';
}

/** Exchange the admin secret for a short-lived session token. @param {string} secret */
export async function adminLogin(secret) {
  const r = await req('/admin/login', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ secret }),
  });
  const j = await r.json();
  adminToken = j.token;
  return j;
}

/** @param {'pending'|'approved'|'rejected'} [status] @returns {Promise<Array<object>>} */
export async function adminListLibrary(status = 'pending') {
  return (await req('/admin/library?status=' + encodeURIComponent(status), { headers: { 'x-admin-token': adminToken } })).json();
}

/** Full record incl. reassembled doc, for review / load-and-run. @param {string} id */
export async function adminLoadEntry(id) {
  return (await req('/admin/library/' + id, { headers: { 'x-admin-token': adminToken } })).json();
}

/** @param {string} id */
export async function adminApprove(id) {
  // bodyless POST — no content-type, or Fastify rejects the empty JSON body
  return (await req('/admin/library/' + id + '/approve', { method: 'POST', headers: { 'x-admin-token': adminToken } })).json();
}

/** @param {string} id */
export async function adminReject(id) {
  return (await req('/admin/library/' + id, { method: 'DELETE', headers: { 'x-admin-token': adminToken } })).json();
}

/** Replace a submission's content in place (admin-curated edit). @param {string} id @param {object} entry */
export async function adminUpdate(id, entry) {
  return (
    await req('/admin/library/' + id, {
      method: 'PUT',
      headers: { 'content-type': 'application/json', 'x-admin-token': adminToken },
      body: JSON.stringify(entry),
    })
  ).json();
}
