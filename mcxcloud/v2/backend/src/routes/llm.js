// @ts-check
import { config } from '../config.js';
import { availableChoices, defaultChoice, plausibleApiKey, promptToDoc, resolveChoice } from '../llm.js';
import { clientIp } from '../net.js';

const MAX_PROMPT = 4000; // a simulation description, not a document

/** in-memory per-client throttle: client IP -> last generate ms */
const lastCall = new Map();

/** @param {import('fastify').FastifyInstance} app */
export async function llmRoutes(app) {
  const prune = setInterval(() => {
    const cut = Date.now() - config.llmMinGapMs * 10;
    for (const [k, v] of lastCall) if (v < cut) lastCall.delete(k);
  }, 60000);
  if (typeof prune.unref === 'function') prune.unref();

  // Capability probe. The frontend is deployed separately from the API and only some
  // manager hosts have the Python/OpenAI side installed, so the LLM tab asks first and
  // hides itself rather than offering a control that always fails.
  app.get('/llm/health', async () => {
    const choices = availableChoices();
    const dflt = defaultChoice(choices);
    return {
      enabled: config.llmEnabled && choices.length > 0,
      default: dflt?.backend,
      defaultModel: dflt?.model,
      choices,
    };
  });

  app.post('/llm/generate', async (req, reply) => {
    if (!config.llmEnabled) {
      return reply.code(503).send({ status: 'disabled', message: 'the LLM service is not configured on this server' });
    }
    // apiKey is the user's own OpenAI key: used for this one request, never stored or logged
    // (Fastify's logger records method + URL, not bodies).
    const { prompt, backend, model, apiKey } =
      /** @type {{ prompt?: unknown, backend?: string, model?: string, apiKey?: unknown }} */ (req.body || {});
    if (typeof prompt !== 'string' || !prompt.trim()) {
      return reply.code(422).send({ status: 'invalid', message: 'a prompt is required' });
    }
    if (prompt.length > MAX_PROMPT) {
      return reply.code(422).send({ status: 'invalid', message: `the prompt is limited to ${MAX_PROMPT} characters` });
    }

    const choice = resolveChoice(backend, model);
    if (!choice) {
      return reply.code(422).send({
        status: 'invalid',
        message: backend
          ? `backend '${backend}'${model ? ` / model '${model}'` : ''} is not available on this server`
          : 'no LLM backend is configured on this server',
      });
    }
    // Hosted models run on the user's key only — the server never lends its own.
    if (choice.needsKey && !plausibleApiKey(typeof apiKey === 'string' ? apiKey.trim() : apiKey)) {
      return reply.code(422).send({
        status: 'invalid',
        message: `${choice.model} needs your own OpenAI API key (sk-…)`,
      });
    }
    const key = choice.needsKey ? /** @type {string} */ (apiKey).trim() : undefined;

    // Every call spends a model request (money, or a GPU slot), so throttle harder than
    // job submission.
    const ip = clientIp(req);
    const now = Date.now();
    const wait = config.llmMinGapMs - (now - (lastCall.get(ip) || 0));
    if (wait > 0) {
      return reply.code(429).send({
        status: 'throttled',
        message: `please wait ${Math.ceil(wait / 1000)} s before generating again`,
        retryAfterMs: wait,
      });
    }
    lastCall.set(ip, now);

    const r = await promptToDoc(prompt.trim(), choice, key);
    if (!r.ok) {
      app.log.warn(`llm generate failed (${choice.backend}/${choice.model}): ${r.error}`);
      return reply.code(502).send({
        status: 'failed',
        message: r.error,
        raw: r.raw,
        backend: choice.backend,
        model: choice.model,
      });
    }
    return reply.send({
      status: 'ok',
      doc: r.doc,
      elapsed: r.elapsed,
      backend: r.backend,
      model: r.model,
      filled: r.filled,
      warnings: r.warnings,
      valid: r.valid,
      errors: r.errors,
    });
  });
}
