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
  return { data, size };
}

// --- viridis colormap as a 256x1 RGBA DataTexture-ready Uint8Array ---------------
// anchor stops (r,g,b in 0..1) sampled from matplotlib viridis
const VIRIDIS = [
  [0.267, 0.005, 0.329], [0.283, 0.141, 0.458], [0.254, 0.265, 0.53],
  [0.207, 0.372, 0.553], [0.164, 0.471, 0.558], [0.128, 0.567, 0.551],
  [0.135, 0.659, 0.518], [0.267, 0.749, 0.441], [0.478, 0.821, 0.318],
  [0.741, 0.873, 0.15], [0.993, 0.906, 0.144],
];

/**
 * Build an RGBA colormap lookup as a Uint8Array (n*4).
 * @param {'viridis'|'gray'} [name] @param {number} [n]
 * @returns {Uint8Array}
 */
export function colormapRGBA(name = 'viridis', n = 256) {
  const out = new Uint8Array(n * 4);
  for (let i = 0; i < n; i++) {
    const t = i / (n - 1);
    let r, g, b;
    if (name === 'gray') {
      r = g = b = t;
    } else {
      const x = t * (VIRIDIS.length - 1);
      const j = Math.min(VIRIDIS.length - 2, Math.floor(x));
      const f = x - j;
      r = VIRIDIS[j][0] * (1 - f) + VIRIDIS[j + 1][0] * f;
      g = VIRIDIS[j][1] * (1 - f) + VIRIDIS[j + 1][1] * f;
      b = VIRIDIS[j][2] * (1 - f) + VIRIDIS[j + 1][2] * f;
    }
    out[i * 4 + 0] = Math.round(r * 255);
    out[i * 4 + 1] = Math.round(g * 255);
    out[i * 4 + 2] = Math.round(b * 255);
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
