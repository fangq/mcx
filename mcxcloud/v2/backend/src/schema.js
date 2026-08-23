// @ts-check
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import Ajv from 'ajv';

// The authoritative MCX input schema (shared with the frontend editor).
const schemaPath = fileURLToPath(new URL('../../schema/mcx-input.v1.json', import.meta.url));
export const mcxSchema = JSON.parse(readFileSync(schemaPath, 'utf8'));

// The editor schema is stricter than mcx itself: it marks the Session Do* flags required
// and types them as booleans so json-editor renders checkboxes. mcx reads all of these
// via cJSON valueint, i.e. 0/1 integers are equally valid (and are what the official
// examples and mcxlab-exported files use), and every flag has a built-in default. For
// server-side validation, relax a clone of the schema accordingly.
/** @param {any} node */
function relaxBooleans(node) {
  if (Array.isArray(node)) {
    node.forEach(relaxBooleans);
  } else if (node && typeof node === 'object') {
    if (node.type === 'boolean') node.type = ['boolean', 'integer'];
    Object.values(node).forEach(relaxBooleans);
  }
}
const relaxedSchema = structuredClone(mcxSchema);
relaxBooleans(relaxedSchema);
relaxedSchema.properties.Session.required = ['ID', 'Photons'];
// mesh (mmc) inputs need no OriginType/Dim — the mesh geometry defines the domain
relaxedSchema.properties.Domain.required = ['Media'];
// mcx only reads the first letter of OutputType (e.g. "flux" == "f"), so accept any
// string starting with a valid type letter; the editor keeps the single-letter dropdown.
relaxedSchema.properties.Session.properties.OutputType = {
  type: 'string',
  pattern: '^[xfejpmrlstbaduvwq]',
};
// DebugFlag/SaveDataMask letter sets differ per engine (mcx RMPT vs mmc SCBWDIOXATRPEM),
// both engines silently skip unknown letters, and v1-era docs carry numeric strings
// ("2560") — accept any string or integer here; the editor keeps the letter-pattern hint.
relaxedSchema.properties.Session.properties.DebugFlag = { type: ['string', 'integer'] };
relaxedSchema.properties.Session.properties.SaveDataMask = { type: ['string', 'integer'] };

// strict:false -> tolerate JSON-Editor-only keywords (options, propertyOrder, format:"table"…);
// validateFormats:false -> don't fail on non-standard formats. Structural validation
// (type/required/min/max/enum/oneOf) is what we rely on.
const ajv = new Ajv({ strict: false, allErrors: true, validateFormats: false });
export const validateInput = ajv.compile(relaxedSchema);

/**
 * Detect which simulator a validated input targets: a Shapes (or Mesh) object carrying
 * MeshNode is a tetrahedral-mesh domain — mmc by default, or redbird (an FEM diffusion
 * solver sharing the same mesh/optode/property schema) if Session.Engine says so.
 * Everything else (voxel Grid domains) runs mcx.
 * @param {Record<string, any>} cfg
 * @returns {'mcx' | 'mmc' | 'redbird'}
 */
export function detectEngine(cfg) {
  const S = cfg?.Shapes ?? cfg?.Mesh;
  const isMesh = S && typeof S === 'object' && !Array.isArray(S) && 'MeshNode' in S;
  if (!isMesh) return 'mcx';
  return cfg?.Session?.Engine === 'redbird' ? 'redbird' : 'mmc';
}

/**
 * mcx reads RF modulation as Optode.Source.Frequency (Hz, converted internally to
 * omega=2*pi*f); mmc reads it pre-converted as Forward.Omega (rad/s). Rather than make
 * users know the per-engine field name/units, Frequency is the one canonical, schema-facing
 * field for both — this derives mmc's Forward.Omega from it right before dispatch. A no-op
 * for engine!=='mmc' (mcx already reads Frequency natively), so it stays inert/backportable
 * on branches without mmc support at all.
 * @param {Record<string, any>} doc @param {'mcx' | 'mmc'} engine
 * @returns {Record<string, any>}
 */
export function applyFrequency(doc, engine) {
  const freq = doc?.Optode?.Source?.Frequency;
  if (engine !== 'mmc' || !freq) return doc;
  return { ...doc, Forward: { ...doc.Forward, Omega: freq * 2 * Math.PI } };
}

// mmc supports a subset of mcx's source types (mmc_utils.c srctypeid[])
const MMC_SRC_TYPES = new Set([
  'pencil', 'isotropic', 'cone', 'gaussian', 'planar', 'pattern', 'fourier',
  'arcsine', 'disk', 'fourierx', 'fourierx2d', 'zgaussian', 'line', 'slit',
]);

/** row count of a mesh array in either plain (array of rows) or JData (_ArraySize_) form
 *  @param {any} a */
const meshRows = (a) =>
  Array.isArray(a) ? a.length : (Array.isArray(a?._ArraySize_) ? Number(a._ArraySize_[0]) || 0 : 0);

/**
 * Preview-mode resource caps (ported from v1 mcxserver.cgi checklimit / index.html
 * checklimit), plus engine-specific checks: constructs one engine parses but the other
 * silently ignores are rejected so users are not misled about the simulated physics.
 * Returns an error message, or null if within limits.
 * @param {Record<string, any>} cfg
 * @returns {string | null}
 */
export function checkLimits(cfg) {
  const S = cfg?.Session ?? {};
  const F = cfg?.Forward ?? {};
  const D = cfg?.Domain ?? {};
  const src = cfg?.Optode?.Source ?? {};
  const engine = detectEngine(cfg);
  // redbird is schema-recognized (Session.Engine) but not yet dispatchable: docker.js has no
  // worker image/CPU routing for it and would silently misdirect it to the mcx GPU image.
  // Reject up front until that backend wiring (swarm CPU dispatch + cfg-builder) lands.
  if (engine === 'redbird') return 'the redbird (FEM diffusion) engine is not yet available in this preview version';
  if (S.Photons > 5e8) return 'the max photon number is limited to 5e8 in this preview version';
  if (typeof S.DebugFlag === 'string' && /m/i.test(S.DebugFlag))
    return 'storing photon trajectories is not supported in this preview version';
  if (F.T1 && F.Dt && F.T1 / F.Dt > 100)
    return 'the maximum time gate number is limited to 100 in this preview version';
  if (Array.isArray(D.Dim) && D.Dim.length === 3 && D.Dim.some((/** @type {number} */ x) => x > 300))
    return 'the maximum domain dimension is 300 in this preview version';
  if (Array.isArray(D.Media) && D.Media.some((/** @type {any} */ m) => m?.mus > 50))
    return 'scattering coeff (mus) is limited to 50/mm in this preview version';
  // Optode.Source.Frequency (Hz) is the one canonical RF field across engines; Forward.Omega
  // (rad/s) is derived from it server-side just before dispatch (see applyFrequency below) —
  // reject it here so it can never silently diverge from Frequency
  if (F.Omega) return 'Forward.Omega is set automatically from Optode.Source.Frequency (Hz) — set Frequency instead';

  if (engine === 'mmc') {
    const M = cfg?.Shapes ?? cfg?.Mesh ?? {};
    if (meshRows(M.MeshNode) > 300000)
      return 'the mesh node count is limited to 300000 in this preview version';
    if (meshRows(M.MeshElem) > 1500000)
      return 'the mesh element count is limited to 1500000 in this preview version';
    if (typeof src.Type === 'string' && !MMC_SRC_TYPES.has(src.Type))
      return `source type "${src.Type}" is not supported for mesh (mmc) simulations`;
    if (typeof S.OutputType === 'string' && /^[mtb]/.test(S.OutputType))
      return `output type "${S.OutputType}" is not supported for mesh (mmc) simulations`;
    // mmc's detector parser assumes each entry is exactly {Pos, R}: any other member
    // count derails it into a NULL dereference (mmc_utils.c:2171 GetArraySize(det)==2),
    // segfaulting the worker with an empty log — reject such docs up front
    const dets = cfg?.Optode?.Detector;
    if (Array.isArray(dets)) {
      for (const det of dets) {
        const keys = Object.keys(det ?? {}).sort();
        if (keys.join() !== 'Pos,R')
          return `mesh (mmc) detectors must contain exactly Pos and R (found: ${keys.join(', ') || 'none'})`;
      }
    }
    // physics settings mmc's JSON parser does not read — reject rather than silently drop
    for (const [ok, what] of [
      [!('IQUV' in src) && !('WaveLength' in src) && !D.MieScatter, 'polarized MC (IQUV/WaveLength/MieScatter)'],
      [!('AngleInverseCDF' in src), 'Optode.Source.AngleInverseCDF'],
      [!D.InverseCDF, 'Domain.InverseCDF'],
      [!S.BCFlags, 'Session.BCFlags'],
    ]) {
      if (!ok) return `${what} is not supported for mesh (mmc) simulations`;
    }
  } else {
    // mcx cannot produce ascii/bin output and ignores mmc's RF key
    if (S.OutputFormat === 'ascii' || S.OutputFormat === 'bin')
      return `output format "${S.OutputFormat}" is only supported for mesh (mmc) simulations`;
    if (F.N0 !== undefined && F.N0 !== 1)
      return 'Forward.N0 is only supported for mesh (mmc) simulations';
  }
  return null;
}
