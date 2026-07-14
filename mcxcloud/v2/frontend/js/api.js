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
