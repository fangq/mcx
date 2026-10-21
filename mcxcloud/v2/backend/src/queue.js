// @ts-check
import pg from 'pg';
import { pool } from './db.js';
import { config } from './config.js';

// Native Postgres job queue — no external queue library, so no minimum Postgres major
// beyond `FOR UPDATE SKIP LOCKED` (PG 9.5+). The `jobs` table IS the queue: submit
// inserts a row with status='queued'; the consumer atomically claims the highest-priority
// queued row with SKIP LOCKED and holds a slot until it finishes. A NOTIFY on submit wakes
// the consumer instantly (LISTEN/NOTIFY); a periodic poll is the safety net for any missed
// signal or crash-requeued job.

const CHANNEL = 'mcx_job_submitted';

/** @type {pg.Client | null} dedicated long-lived connection for LISTEN */
let listener = null;
/** @type {(() => void) | null} */
let onNotify = null;

export async function initQueue() {
  // A pooled connection can't reliably hold a persistent LISTEN, so use a dedicated client.
  listener = new pg.Client({ connectionString: config.databaseUrl });
  await listener.connect();
  listener.on('notification', () => onNotify && onNotify());
  listener.on('error', (err) => {
    // Connection dropped; the periodic poll keeps things moving until it is re-established.
    console.error('queue listener error:', err.message);
  });
  await listener.query(`LISTEN ${CHANNEL}`);
}

/**
 * Signal that a new job is queued. The row is already inserted (status='queued') by the
 * caller inside its transaction; this just wakes the consumer.
 * @param {string} _jobId
 * @param {number} _priority
 * @returns {Promise<void>}
 */
export async function enqueueJob(_jobId, _priority) {
  await pool.query('select pg_notify($1, $2)', [CHANNEL, '']);
}

/**
 * Run the consumer loop, keeping up to `capacity` jobs in flight (= free GPU count).
 * Each iteration atomically claims one queued job (status queued -> running) using
 * FOR UPDATE SKIP LOCKED so concurrent/parallel API processes never grab the same row,
 * then invokes `handle` without awaiting and frees the slot when it settles.
 * @param {number} capacity
 * @param {(jobId: string) => Promise<void>} handle
 * @returns {Promise<void>}
 */
export async function startWork(capacity, handle) {
  const cap = Math.max(1, capacity);
  let active = 0;
  let pumping = false;
  let rerun = false;

  async function claimOne() {
    const r = await pool.query(
      `update jobs set status = 'running', started_at = now()
       where id = (
         select id from jobs where status = 'queued'
         order by priority desc, created_at
         for update skip locked limit 1
       )
       returning id`,
    );
    return r.rows[0]?.id ?? null;
  }

  async function pump() {
    if (pumping) { rerun = true; return; }
    pumping = true;
    try {
      do {
        rerun = false;
        while (active < cap) {
          const id = await claimOne();
          if (!id) break;
          active++;
          Promise.resolve()
            .then(() => handle(id))
            .catch((err) => console.error(`job ${id} handler error:`, err))
            .finally(() => { active--; pump(); });
        }
      } while (rerun);
    } finally {
      pumping = false;
    }
  }

  onNotify = () => { pump(); };
  const poll = setInterval(() => pump(), 5000);
  if (typeof poll.unref === 'function') poll.unref();
  await pump();
}
