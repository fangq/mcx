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
// mcx only reads the first letter of OutputType (e.g. "flux" == "f"), so accept any
// string starting with a valid type letter; the editor keeps the single-letter dropdown.
relaxedSchema.properties.Session.properties.OutputType = {
  type: 'string',
  pattern: '^[xfejpmrlstbaduvwq]',
};

// strict:false -> tolerate JSON-Editor-only keywords (options, propertyOrder, format:"table"…);
// validateFormats:false -> don't fail on non-standard formats. Structural validation
// (type/required/min/max/enum/oneOf) is what we rely on.
const ajv = new Ajv({ strict: false, allErrors: true, validateFormats: false });
export const validateInput = ajv.compile(relaxedSchema);

/**
 * Preview-mode resource caps (ported from v1 mcxserver.cgi checklimit / index.html
 * checklimit). Returns an error message, or null if within limits.
 * @param {Record<string, any>} cfg
 * @returns {string | null}
 */
export function checkLimits(cfg) {
  const S = cfg?.Session ?? {};
  const F = cfg?.Forward ?? {};
  const D = cfg?.Domain ?? {};
  if (S.Photons > 5e8) return 'the max photon number is limited to 5e8 in this preview version';
  if (typeof S.DebugFlag === 'string' && /m/i.test(S.DebugFlag))
    return 'storing photon trajectories is not supported in this preview version';
  if (F.T1 && F.Dt && F.T1 / F.Dt > 100)
    return 'the maximum time gate number is limited to 100 in this preview version';
  if (Array.isArray(D.Dim) && D.Dim.length === 3 && D.Dim.some((/** @type {number} */ x) => x > 300))
    return 'the maximum domain dimension is 300 in this preview version';
  if (Array.isArray(D.Media) && D.Media.some((/** @type {any} */ m) => m?.mus > 50))
    return 'scattering coeff (mus) is limited to 50/mm in this preview version';
  return null;
}
