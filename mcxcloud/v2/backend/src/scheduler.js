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
const TERMINAL = new Set(['completed', 'cached', 'failed', 'cancelled', 'killed']);

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

  // engine (mcx|mmc) was detected from the input at submit time; it selects the worker
  // image and the simulator binary inside the shared run script
  const er = await pool.query('select engine from jobs where id = $1', [jobId]);
  const engine = er.rows[0]?.engine || 'mcx';

  const name = serviceName(jobId);
  try {
    await createMcxService({ name, jobId, seed: false, script: RUN_SCRIPT, engine });
  } catch (err) {
    await pool.query(
      `update jobs set status = 'failed', error = $2, ended_at = now() where id = $1`,
      [jobId, 'dispatch failed: ' + (/** @type {Error} */ (err)).message],
    );
    publish(jobId, 'error', { status: 'failed', message: 'dispatch failed' });
    await removeService(name);
    return;
  }

  // wait for the container's completion callback to flip the status (grace beyond the
  // hard runtime cap for network/upload time)
  const deadline = Date.now() + config.maxRuntimeMs + 15000;
  let done = false;
  while (Date.now() < deadline) {
    await sleep(1000);
    const r = await pool.query('select status from jobs where id = $1', [jobId]);
    const st = r.rows[0]?.status;
    if (st && TERMINAL.has(st)) {
      done = true;
      break;
    }
  }
  if (!done) {
    await pool.query(
      `update jobs set status = 'killed', error = 'exceeded max runtime', ended_at = now()
       where id = $1 and status not in ('completed','cached','failed')`,
      [jobId],
    );
    publish(jobId, 'error', { status: 'killed', message: 'exceeded max runtime' });
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
