// @ts-check
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { setTimeout as sleep } from 'node:timers/promises';
import { config } from './config.js';
import { pool } from './db.js';
import { publish } from './sse.js';
import { startWork } from './queue.js';
import { countGpus, createMcxService, removeService } from './docker.js';

const RUN_SCRIPT = readFileSync(fileURLToPath(new URL('../worker/mcx-run.sh', import.meta.url)), 'utf8');
// redbird runs a Python driver that the worker image does not ship (only redbirdpy
// itself), so splice our redbird_fwd.py into the run script's heredoc placeholder (see
// redbird-run.sh). Keeping the driver a separate .py file means it stays
// runnable/testable outside the container.
// A function replacer (not a string) so any $-sequence in the Python source is inserted
// literally rather than being read as a replacement pattern.
const REDBIRD_SCRIPT = (() => {
  const sh = readFileSync(fileURLToPath(new URL('../worker/redbird-run.sh', import.meta.url)), 'utf8');
  const py = readFileSync(fileURLToPath(new URL('../worker/redbird_fwd.py', import.meta.url)), 'utf8').trimEnd();
  const TOKEN = '@REDBIRD_FWD_PY@';
  const DELIM = 'REDBIRD_FWD_PY_EOF';
  // fail loudly at startup rather than dispatching a silently-broken worker script
  if (sh.split(TOKEN).length - 1 !== 1) throw new Error(`redbird-run.sh must contain ${TOKEN} exactly once`);
  if (py.includes(DELIM)) throw new Error(`redbird_fwd.py must not contain the heredoc delimiter ${DELIM}`);
  return sh.replace(TOKEN, () => py);
})();
// SUCCESS is checked every poll tick and always stops the wait immediately. 'failed' is
// deliberately NOT here: with --restart-max-attempts > 1 (docker.js), a container failure
// (bad GPU, phantom resource, transient driver error) may still be retried by swarm as a
// fresh scheduling attempt — if we stopped and removeService()'d on the first 'failed', we'd
// delete the service (and any retry swarm was about to run) out from under it. So 'failed'
// is left to ride out the full deadline below, which is sized for the whole retry budget;
// a later successful retry can still overwrite it to 'completed' before that deadline hits.
const SUCCESS = new Set(['completed', 'cached', 'cancelled']);

/** @param {string} jobId @returns {string} */
function serviceName(jobId) {
  return 'mcx_' + jobId.replace(/-/g, '');
}

/**
 * Dispatch one job to the swarm and hold this slot until it finishes (or is killed on
 * timeout). Holding the slot for the sim duration is what caps concurrency at the GPU
 * count. The container pushes results back via the API (no NFS polling).
 * @param {string} jobId
 * @returns {Promise<void>}
 */
async function handle(jobId) {
  // The queue already atomically claimed this job (status queued -> running) before
  // handing it to us; just announce it and dispatch.
  publish(jobId, 'status', { status: 'running' });

  // engine (mcx|mmc|redbird) was detected from the input at submit time; it selects the
  // worker image and the simulator binary inside the shared run script
  const er = await pool.query('select engine from jobs where id = $1', [jobId]);
  const engine = er.rows[0]?.engine || 'mcx';
  // redbird is a CPU/Python FEM solve, not a GPU MC binary — it needs its own entrypoint
  const script = engine === 'redbird' ? REDBIRD_SCRIPT : RUN_SCRIPT;

  const name = serviceName(jobId);
  try {
    await createMcxService({ name, jobId, seed: false, script, engine });
  } catch (err) {
    await pool.query(
      `update jobs set status = 'failed', error = $2, ended_at = now() where id = $1`,
      [jobId, 'dispatch failed: ' + (/** @type {Error} */ (err)).message],
    );
    // dispatch itself failed (no service was ever created) -> nothing can retry this;
    // final=true tells the frontend to stop listening (see api.js streamJob)
    publish(jobId, 'error', { status: 'failed', message: 'dispatch failed', final: true });
    await removeService(name);
    return;
  }

  // Wait for a completion callback to flip the status. The deadline covers the WHOLE
  // retry budget (every attempt gets its own maxRuntimeMs, plus a restart-delay between
  // each) so we don't removeService() out from under a swarm-driven retry that's still in
  // flight; see the SUCCESS comment above for why 'failed' alone doesn't stop the wait.
  const attempts = Math.max(1, config.restartMaxAttempts);
  // redbird gets its own (much larger) per-attempt budget: a direct sparse FEM solve is far
  // slower than a GPU MC run, and its JSON parse/serialize in Python adds to that
  const perAttemptMs = engine === 'redbird' ? config.redbirdMaxRuntimeMs : config.maxRuntimeMs;
  const deadline = Date.now() + attempts * perAttemptMs + (attempts - 1) * config.restartDelayMs + 15000;
  let done = false;
  while (Date.now() < deadline) {
    await sleep(1000);
    const r = await pool.query('select status from jobs where id = $1', [jobId]);
    const st = r.rows[0]?.status;
    if (st && SUCCESS.has(st)) {
      done = true;
      break;
    }
  }
  if (!done) {
    // Exhausted the full retry budget. If a container attempt at least ran and reported
    // (status='failed', with a real mcx log+error already stored), that's more useful to
    // the user than overwriting it — leave it. Only jobs that never got ANY callback across
    // every attempt (status still 'running' — e.g. every attempt failed at the container/
    // GPU-injection level before the script could even start) get the generic timeout.
    const r = await pool.query(
      `update jobs set status = 'killed', error = 'exceeded max runtime', ended_at = now()
       where id = $1 and status = 'running' returning status`,
      [jobId],
    );
    if (r.rowCount) {
      publish(jobId, 'error', { status: 'killed', message: 'exceeded max runtime', final: true });
    } else {
      // a real attempt already reported via /complete?error=1 (log/error already stored);
      // forward it so the now-reconnecting frontend can still show what actually happened
      const last = await pool.query('select log, error from jobs where id = $1', [jobId]);
      publish(jobId, 'error', {
        status: 'failed', message: 'simulation error', final: true,
        log: last.rows[0]?.log, error: last.rows[0]?.error,
      });
    }
  }
  await removeService(name);
}

/**
 * On startup, re-queue any jobs left 'running' by a previous crash so they are not
 * orphaned.
 */
async function recoverStale() {
  await pool.query(`update jobs set status = 'queued', started_at = null where status = 'running'`);
}

/** @returns {Promise<number>} the GPU/concurrency capacity the scheduler is using */
export async function initScheduler() {
  await recoverStale();
  let capacity = config.maxConcurrent;
  if (capacity <= 0) {
    capacity = Math.max(1, await countGpus().catch(() => 1));
  }
  await startWork(capacity, handle);
  return capacity;
}
