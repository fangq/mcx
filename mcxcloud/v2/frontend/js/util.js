// @ts-check
import pako from 'pako';

/** @param {string} sel @returns {HTMLElement} */
export const $ = (sel) => /** @type {HTMLElement} */ (document.querySelector(sel));
/** @param {string} sel @returns {NodeListOf<HTMLElement>} */
export const $$ = (sel) => document.querySelectorAll(sel);

/** base64 -> Uint8Array */
export function b64ToBytes(b64) {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

const TYPED = {
  uint8: Uint8Array, int8: Int8Array,
  uint16: Uint16Array, int16: Int16Array,
  uint32: Uint32Array, int32: Int32Array,
  single: Float32Array, float32: Float32Array,
  double: Float64Array, float64: Float64Array,
};

/**
 * Decode a JData packed/annotated N-D array node into a flat typed array + shape.
 * Handles the common `_ArrayZipData_` (zlib/gzip + base64) form and plain `_ArrayData_`.
 * Self-contained (pako only) — no jdata/numjs dependency.
 * @param {any} node
 * @returns {{ data: Float32Array, size: number[] }}
 */
export function decodeJDataArray(node) {
  const Ctor = TYPED[node._ArrayType_] || Float64Array;
  const size = Array.isArray(node._ArraySize_) ? node._ArraySize_.slice() : [];
  /** @type {ArrayLike<number>} */
  let typed;
  if (node._ArrayZipData_ !== undefined) {
    const raw = b64ToBytes(node._ArrayZipData_);
    const zt = String(node._ArrayZipType_ || 'zlib').toLowerCase();
    const bytes = zt === 'gzip' ? pako.ungzip(raw) : pako.inflate(raw);
    typed = new Ctor(bytes.buffer, bytes.byteOffset, Math.floor(bytes.byteLength / Ctor.BYTES_PER_ELEMENT));
  } else if (Array.isArray(node._ArrayData_)) {
    typed = /** @type {any} */ (Ctor).from(node._ArrayData_.flat(Infinity));
  } else {
    typed = new Float32Array(0);
  }
  // the shader samples a FloatType texture; convert once
  const data = typed instanceof Float32Array ? typed : Float32Array.from(typed);
  const order = String(node._ArrayOrder_ || '').toLowerCase(); // 'c' = row-major (e.g. MCX output)
  return { data, size, order };
}

// --- named colormaps (anchor stops r,g,b in 0..1) -> 256x1 RGBA Uint8Array ---------
const COLORMAPS = {
  viridis: [
    [0.267, 0.005, 0.329], [0.283, 0.141, 0.458], [0.254, 0.265, 0.53],
    [0.207, 0.372, 0.553], [0.164, 0.471, 0.558], [0.128, 0.567, 0.551],
    [0.135, 0.659, 0.518], [0.267, 0.749, 0.441], [0.478, 0.821, 0.318],
    [0.741, 0.873, 0.15], [0.993, 0.906, 0.144],
  ],
  jet: [[0, 0, 0.5], [0, 0, 1], [0, 1, 1], [0.5, 1, 0.5], [1, 1, 0], [1, 0, 0], [0.5, 0, 0]],
  hot: [[0, 0, 0], [0.9, 0, 0], [1, 0.8, 0], [1, 1, 1]],
  plasma: [[0.05, 0.03, 0.53], [0.4, 0.0, 0.66], [0.69, 0.17, 0.5], [0.9, 0.36, 0.29],
    [0.99, 0.65, 0.13], [0.94, 0.98, 0.13]],
  gray: [[0, 0, 0], [1, 1, 1]],
  bone: [[0, 0, 0], [0.32, 0.34, 0.44], [0.65, 0.66, 0.73], [1, 1, 1]],
};

/** names available for the colormap picker */
export const COLORMAP_NAMES = Object.keys(COLORMAPS);

/**
 * Build an RGBA colormap lookup as a Uint8Array (n*4).
 * @param {string} [name] @param {number} [n]
 * @returns {Uint8Array}
 */
export function colormapRGBA(name = 'viridis', n = 256) {
  const stops = COLORMAPS[name] || COLORMAPS.viridis;
  const out = new Uint8Array(n * 4);
  for (let i = 0; i < n; i++) {
    const x = (i / (n - 1)) * (stops.length - 1);
    const j = Math.min(stops.length - 2, Math.floor(x));
    const f = x - j;
    out[i * 4 + 0] = Math.round((stops[j][0] * (1 - f) + stops[j + 1][0] * f) * 255);
    out[i * 4 + 1] = Math.round((stops[j][1] * (1 - f) + stops[j + 1][1] * f) * 255);
    out[i * 4 + 2] = Math.round((stops[j][2] * (1 - f) + stops[j + 1][2] * f) * 255);
    out[i * 4 + 3] = 255;
  }
  return out;
}

/** trigger a browser download of a text/JSON blob */
export function downloadLink(anchor, text, filename) {
  const type = filename.endsWith('.json') || filename.endsWith('.jnii') || filename.endsWith('.jdt') ? 'application/json' : 'text/plain';
  anchor.href = URL.createObjectURL(new Blob([text], { type }));
  anchor.download = filename;
  anchor.hidden = false;
}

/** compress a JS object to a base64 string for a shareable URL (gzip via pako) */
export function encodeStateToUrl(obj) {
  const gz = pako.gzip(JSON.stringify(obj));
  let bin = '';
  for (let i = 0; i < gz.length; i++) bin += String.fromCharCode(gz[i]);
  return btoa(bin);
}
/** inverse of encodeStateToUrl */
export function decodeStateFromUrl(s) {
  const bin = atob(s);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return JSON.parse(pako.ungzip(bytes, { to: 'string' }));
}
