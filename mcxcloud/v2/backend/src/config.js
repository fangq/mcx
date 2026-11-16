// @ts-check

/**
 * @typedef {Object} Config
 * @property {number} port
 * @property {string} host
 * @property {string} databaseUrl
 * @property {string} corsOrigin
 * @property {number} threshold
 * @property {string} workerSecret
 * @property {boolean} runScheduler
 * @property {string} workerApiUrl     API base URL reachable from inside the swarm container
 * @property {string} workerImage      docker image for the mcx worker
 * @property {number} maxRuntimeMs      hard cap per simulation before kill
 * @property {number} maxConcurrent     0 = auto-detect from swarm GPU count
 * @property {string} workerConstraint  optional swarm placement constraint (e.g. 'node.hostname==neza'); '' = any node
 * @property {number} jobTtlMs          purge non-library jobs older than this (ms)
 * @property {number} minSubmitGapMs    minimum gap between submissions from one client (ms)
 * @property {string} adminSecret       library review/admin API secret; '' disables it
 * @property {boolean} llmEnabled       serve POST /llm/generate (needs the Python side installed)
 * @property {string} llmPython         interpreter that has MCXLLM's deps (langchain, openai)
 * @property {string} llmMcxDir         directory holding LLMMCX_gpt4o.py
 * @property {number} llmTimeoutMs      hard cap on one prompt -> JSON conversion
 * @property {number} llmMinGapMs       minimum gap between generate calls from one client (ms)
 * @property {string} llmBackend        default backend id ('openai' | 'ollama')
 * @property {string} llmDefaultModel   preselected model within llmBackend ('' = its first model)
 * @property {string} llmOpenaiModel    model used by the openai backend ('' hides it); each
 *   request brings the user's own API key — the server's OPENAI_API_KEY is never used
 * @property {string} llmOllamaUrl      full OpenAI-style chat URL of an ollama server or proxy
 *   ('' disables it); a bare '.../v1' base gets /chat/completions appended
 * @property {string[]} llmOllamaModels models offered by the ollama backend
 */

/** @type {Config} */
export const config = {
  port: Number(process.env.PORT ?? 8080),
  host: process.env.HOST ?? '0.0.0.0',
  databaseUrl: process.env.DATABASE_URL ?? 'postgres://mcxcloud@localhost/mcxcloud',
  corsOrigin: process.env.CORS_ORIGIN ?? 'https://mcx.space',
  threshold: Number(process.env.BLOB_THRESHOLD ?? 4096),
  workerSecret: process.env.WORKER_SECRET ?? 'change-me',
  runScheduler: process.env.RUN_SCHEDULER !== '0',
  workerApiUrl: process.env.WORKER_API_URL ?? 'http://localhost:8080',
  workerImage: process.env.WORKER_IMAGE ?? 'fangqq/mcx:v2024.2',
  maxRuntimeMs: Number(process.env.MAX_RUNTIME_MS ?? 60000),
  maxConcurrent: Number(process.env.MAX_CONCURRENT ?? 0),
  workerConstraint: process.env.WORKER_NODE_CONSTRAINT ?? '',
  jobTtlMs: Number(process.env.JOB_TTL_MS ?? 3600000), // 1 hour
  minSubmitGapMs: Number(process.env.MIN_SUBMIT_GAP_MS ?? 5000), // 5 s
  adminSecret: process.env.ADMIN_SECRET ?? '', // '' disables the library review/admin API
  // --- LLM prompt -> MCX JSON (MCXLLM) ---
  llmEnabled: process.env.LLM_ENABLED === '1',
  llmPython: process.env.LLM_PYTHON ?? 'python3',
  llmMcxDir: process.env.LLM_MCX_DIR ?? '',
  llmTimeoutMs: Number(process.env.LLM_TIMEOUT_MS ?? 120000), // 2 min
  llmMinGapMs: Number(process.env.LLM_MIN_GAP_MS ?? 10000), // 10 s
  llmBackend: process.env.LLM_BACKEND ?? 'ollama',
  llmDefaultModel: process.env.LLM_DEFAULT_MODEL ?? 'qwen3.8:27b',
  llmOpenaiModel: process.env.LLM_OPENAI_MODEL ?? 'gpt-4o',
  llmOllamaUrl: process.env.LLM_OLLAMA_URL ?? '',
  llmOllamaModels: (process.env.LLM_OLLAMA_MODELS ?? '')
    .split(',')
    .map((m) => m.trim())
    .filter(Boolean),
};
