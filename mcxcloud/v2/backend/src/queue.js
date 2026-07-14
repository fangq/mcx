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
  if (!boss) return; // queue not initialized (e.g. DB down) — Phase 3 owns dispatch
  await boss.send(RUN_QUEUE, { jobId }, { priority });
}
