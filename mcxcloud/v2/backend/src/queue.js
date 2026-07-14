// @ts-check
import PgBoss from 'pg-boss';
import { config } from './config.js';

// pg-boss keeps the job queue inside Postgres (no Redis). Phase 2 only enqueues on
// submit; the consumer/scheduler that dispatches to Docker Swarm is Phase 3.
export const RUN_QUEUE = 'run-mcx';

/** @type {PgBoss | null} */
let boss = null;

export async function initQueue() {
  boss = new PgBoss(config.databaseUrl);
  await boss.start();
  await boss.createQueue(RUN_QUEUE);
}

/**
 * @param {string} jobId
 * @param {number} priority
 * @returns {Promise<void>}
 */
export async function enqueueJob(jobId, priority) {
  if (!boss) return; // queue not initialized (e.g. DB down)
  await boss.send(RUN_QUEUE, { jobId }, { priority });
}

/**
 * Register the scheduler consumer. pg-boss delivers up to `capacity` jobs per batch and
 * holds them active until the handler resolves (crash-safe redelivery). Processing the
 * batch concurrently caps in-flight simulations at `capacity` (= free GPU count).
 * NOTE: verify the pg-boss v10 work() options (`batchSize`, array handler) against the
 * installed version on the target.
 * @param {number} capacity
 * @param {(jobId: string) => Promise<void>} handle
 * @returns {Promise<void>}
 */
export async function startWork(capacity, handle) {
  if (!boss) throw new Error('queue not initialized');
  await boss.work(RUN_QUEUE, { batchSize: Math.max(1, capacity) }, async (jobs) => {
    const list = Array.isArray(jobs) ? jobs : [jobs];
    await Promise.all(list.map((j) => handle(j.data.jobId)));
  });
}
