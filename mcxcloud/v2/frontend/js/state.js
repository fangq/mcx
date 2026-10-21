// @ts-check
// Tiny reactive store — the single source of truth (replaces v1's globals + hidden
// inputs + textarea string-flags). Views subscribe and react to changes.

/** @type {Set<(key: string, value: any) => void>} */
const listeners = new Set();

/**
 * @typedef {Object} AppState
 * @property {object|null} schema      MCX input JSON Schema (fetched from the API)
 * @property {object|null} doc         current MCX input document (from the editor)
 * @property {boolean} valid           does `doc` validate?
 * @property {string|null} jobId
 * @property {string|null} token
 * @property {string} status
 * @property {string} log
 * @property {object|null} output      last output (parsed JNIfTI) for preview/download
 * @property {boolean} hasDetphoton
 * @property {object} user             { fullname, email, inst, netname }
 */

/** @type {AppState} */
const initial = {
  schema: null,
  doc: null,
  valid: false,
  jobId: null,
  token: null,
  status: 'idle',
  log: '',
  output: null,
  hasDetphoton: false,
  user: {},
};

export const state = new Proxy(initial, {
  set(obj, key, value) {
    obj[/** @type {string} */ (key)] = value;
    for (const fn of listeners) fn(/** @type {string} */ (key), value);
    return true;
  },
});

/**
 * Subscribe to state changes.
 * @param {(key: string, value: any) => void} fn
 * @returns {() => void} unsubscribe
 */
export function subscribe(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}
