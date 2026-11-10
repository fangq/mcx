// @ts-check
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { config } from './config.js';
import { validateInput } from './schema.js';

const BRIDGE = fileURLToPath(new URL('../llm/mcxllm_bridge.py', import.meta.url));
const MAX_OUT = 4 * 1024 * 1024; // a generated MCX doc; anything larger is a runaway reply
const MAX_ERR = 8 * 1024; // keep the stderr tail for diagnostics, not the whole langchain log

/**
 * Fields the v2 schema marks required that MCXLLM's post-processor never emits.
 *
 * `reorder_keys()` in LLMJson_post_gpt4o.py rebuilds the document with a fixed key set
 * that predates this schema, so every generated doc is missing these two. Without the
 * fill the editor reports the input as invalid and Run refuses to submit it. Both carry
 * `default: true` in the schema, so supplying that default is exactly what the form would
 * have done — we are not inventing a value, just writing the one down.
 * @type {[string, string, any][]} [section, key, value]
 */
const REQUIRED_DEFAULTS = [
  ['Session', 'DoAutoThread', true],
  ['Domain', 'OriginType', true],
];

/**
 * Add the schema-required keys MCXLLM omits. Mutates `doc`.
 * @param {Record<string, any>} doc
 * @returns {string[]} the dotted names actually filled in
 */
export function fillRequiredDefaults(doc) {
  /** @type {string[]} */
  const filled = [];
  for (const [section, key, value] of REQUIRED_DEFAULTS) {
    const obj = doc?.[section];
    if (obj && typeof obj === 'object' && !(key in obj)) {
      obj[key] = value;
      filled.push(`${section}.${key}`);
    }
  }
  return filled;
}

/**
 * Flag internal contradictions that the schema cannot catch.
 *
 * MCXLLM sets `Shapes[].Grid.Size` from the prompt but leaves `Exterior.Dim` (which the
 * post-processor renames to `Domain.Dim`) at its pydantic default of [60,60,60] — so a
 * prompt asking for a 100^3 domain can yield Grid.Size 100^3 with Domain.Dim 60^3. Both
 * are individually valid, and shapes placed beyond Dim are silently clipped.
 *
 * Reported, not repaired: which field the user meant is a judgement call, and quietly
 * rewriting the model's answer would hide exactly the errors worth knowing about.
 * @param {Record<string, any>} doc
 * @returns {string[]} human-readable warnings
 */
export function findInconsistencies(doc) {
  /** @type {string[]} */
  const warnings = [];
  const dim = doc?.Domain?.Dim;
  const grid = (Array.isArray(doc?.Shapes) ? doc.Shapes : [])
    .map((/** @type {any} */ s) => s?.Grid?.Size)
    .find(Array.isArray);
  if (Array.isArray(dim) && Array.isArray(grid) && String(dim) !== String(grid)) {
    warnings.push(
      `Domain.Dim [${dim}] and Shapes.Grid.Size [${grid}] disagree — ` +
        'the volume is Domain.Dim, so anything outside it is clipped.',
    );
  }
  return warnings;
}

/**
 * @typedef {Object} BridgeResult
 * @property {boolean} ok
 * @property {Record<string, any>} [doc]
 * @property {number} [elapsed]
 * @property {string} [error]
 * @property {string} [raw]
 */

/**
 * @typedef {Object} LlmChoice
 * @property {string} backend          'openai' | 'ollama'
 * @property {string} model
 * @property {string} [host]           ollama host, shown so it is clear where a prompt goes
 * @property {boolean} [needsKey]      the request must carry the user's own API key
 */

/**
 * Every (backend, model) pair this server will run, in dropdown order — the default first.
 *
 * Derived from config rather than listed explicitly so a backend cannot be advertised
 * without what it needs: ollama appears only with a URL. openai needs nothing server-side
 * because every request brings the user's own key (`needsKey`), so it is offered whenever
 * LLM_OPENAI_MODEL is non-empty. The frontend renders this list and echoes one choice back.
 * @returns {LlmChoice[]}
 */
export function availableChoices() {
  /** @type {LlmChoice[]} */
  const out = [];
  if (config.llmOllamaUrl) {
    // the host is shown in the UI so it is obvious whether a prompt leaves the network
    let host = '';
    try { host = new URL(config.llmOllamaUrl).host; } catch { /* malformed URL */ }
    for (const model of config.llmOllamaModels) out.push({ backend: 'ollama', model, host });
  }
  if (config.llmOpenaiModel) out.push({ backend: 'openai', model: config.llmOpenaiModel, needsKey: true });
  return out;
}

/**
 * The preselected choice: LLM_DEFAULT_MODEL within LLM_BACKEND, else that backend's first
 * model, else whatever is first.
 * @param {LlmChoice[]} [choices]
 * @returns {LlmChoice | null}
 */
export function defaultChoice(choices = availableChoices()) {
  const inBackend = choices.filter((c) => c.backend === config.llmBackend);
  return (
    inBackend.find((c) => c.model === config.llmDefaultModel) || inBackend[0] || choices[0] || null
  );
}

/**
 * Resolve a requested choice against what this server actually offers.
 * @param {string} [backend] @param {string} [model]
 * @returns {LlmChoice | null} null when unavailable
 */
export function resolveChoice(backend, model) {
  const choices = availableChoices();
  if (!backend) return defaultChoice(choices);
  return choices.find((c) => c.backend === backend && (!model || c.model === model)) || null;
}

/**
 * Shape check for a user-supplied OpenAI key. Not a validity check — OpenAI decides that —
 * just enough to refuse obvious junk before spending a bridge process on it.
 * @param {unknown} key @returns {key is string}
 */
export function plausibleApiKey(key) {
  return typeof key === 'string' && /^[A-Za-z0-9_-]{20,256}$/.test(key);
}

/**
 * Replace every occurrence of `secret` in `text`, so an upstream error message that echoes
 * the key cannot carry it into logs or the reply.
 * @param {string|undefined} text @param {string} [secret]
 */
function redact(text, secret) {
  return text && secret ? text.split(secret).join('[your API key]') : text;
}

/**
 * Spawn the Python bridge and hand it the request on stdin. One process per request: the
 * langchain import costs ~2 s against a 6-40 s model call, which is not worth a resident
 * worker until the endpoint sees real traffic.
 *
 * The user's API key travels on stdin, not in argv or the environment, so it never shows up
 * in `ps` or /proc/<pid>/environ. Every OPENAI_* variable is stripped from the child's
 * environment: whatever key or org this server happens to have must never be billed for a
 * user's request.
 * @param {string} prompt
 * @param {{ backend: string, model: string }} choice
 * @param {string} [apiKey] the user's OpenAI key (openai backend only)
 * @returns {Promise<BridgeResult>}
 */
function runBridge(prompt, choice, apiKey) {
  return new Promise((resolve) => {
    /** @type {Record<string, string|undefined>} */
    const env = { ...process.env, LLM_MCX_DIR: config.llmMcxDir, LLM_OLLAMA_URL: config.llmOllamaUrl };
    for (const k of Object.keys(env)) if (k.startsWith('OPENAI_')) delete env[k];

    /** @type {import('node:child_process').ChildProcess} */
    let child;
    try {
      child = spawn(config.llmPython, [BRIDGE], { env, stdio: ['pipe', 'pipe', 'pipe'] });
    } catch (err) {
      resolve({ ok: false, error: 'cannot start the LLM bridge: ' + (/** @type {Error} */ (err)).message });
      return;
    }

    let out = '';
    let err = '';
    let settled = false;
    let timedOut = false;

    /** @param {BridgeResult} result */
    const finish = (result) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      resolve(result);
    };

    const timer = setTimeout(() => {
      timedOut = true;
      child.kill('SIGKILL');
    }, config.llmTimeoutMs);

    child.stdout?.on('data', (d) => {
      if (out.length < MAX_OUT) out += d;
    });
    child.stderr?.on('data', (d) => {
      err += d;
      if (err.length > MAX_ERR) err = err.slice(-MAX_ERR);
    });
    child.on('error', (e) => finish({ ok: false, error: 'cannot start the LLM bridge: ' + e.message }));

    child.on('close', () => {
      if (timedOut) {
        finish({ ok: false, error: `the LLM did not answer within ${Math.round(config.llmTimeoutMs / 1000)} s` });
        return;
      }
      const line = out.trim().split('\n').pop() || '';
      if (!line) {
        finish({ ok: false, error: 'the LLM bridge produced no output' + (err ? ': ' + err.trim() : '') });
        return;
      }
      try {
        finish(/** @type {BridgeResult} */ (JSON.parse(line)));
      } catch {
        finish({ ok: false, error: 'unreadable reply from the LLM bridge', raw: line.slice(0, 2000) });
      }
    });

    child.stdin?.on('error', () => { /* killed before the request was written */ });
    child.stdin?.end(
      JSON.stringify({ prompt, backend: choice.backend, model: choice.model, apiKey }),
      'utf8',
    );
  });
}

/**
 * @typedef {Object} GenerateResult
 * @property {boolean} ok
 * @property {Record<string, any>} [doc]     the generated MCX input (schema defaults filled)
 * @property {number} [elapsed]              seconds spent in MCXLLM
 * @property {string[]} [filled]             schema-required keys this added
 * @property {boolean} [valid]               does `doc` pass the MCX input schema?
 * @property {string[]} [errors]             validation messages when `valid` is false
 * @property {string[]} [warnings]           self-contradictions the schema cannot catch
 * @property {string} [error]                why generation failed outright
 * @property {string} [raw]                  unparseable model output, when there is any
 * @property {string} [backend]              which backend answered
 * @property {string} [model]                which model answered
 */

/**
 * Turn a natural-language prompt into an MCX input document.
 *
 * A doc that fails validation is still returned: the user can repair it in the JSON tab,
 * which is more useful than discarding a near-miss. Callers should surface `valid`.
 * @param {string} prompt
 * @param {{ backend: string, model: string }} choice
 * @param {string} [apiKey] the user's OpenAI key (openai backend only)
 * @returns {Promise<GenerateResult>}
 */
export async function promptToDoc(prompt, choice, apiKey) {
  const { backend, model } = choice;
  const r = await runBridge(prompt, choice, apiKey);
  if (!r.ok || !r.doc) {
    return {
      ok: false,
      error: redact(r.error, apiKey) || 'the LLM returned no document',
      raw: redact(r.raw, apiKey),
      backend,
      model,
    };
  }
  const filled = fillRequiredDefaults(r.doc);
  const valid = validateInput(r.doc);
  return {
    ok: true,
    doc: r.doc,
    elapsed: r.elapsed,
    backend,
    model,
    filled,
    warnings: findInconsistencies(r.doc),
    valid: Boolean(valid),
    errors: valid
      ? []
      : (validateInput.errors || []).map(
          (/** @type {any} */ e) => `${e.instancePath || '/'} ${e.message}`,
        ),
  };
}
