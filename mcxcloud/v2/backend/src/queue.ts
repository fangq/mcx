import PgBoss from 'pg-boss';
import { config } from './config.ts';

// pg-boss keeps the job queue inside Postgres (no Redis). Phase 2 only enqueues on
// submit; the consumer/scheduler that dispatches to Docker Swarm is Phase 3.
export const RUN_QUEUE = 'run-mcx';

let boss: PgBoss | null = null;

export async function initQueue(): Promise<void> {
  boss = new PgBoss(config.databaseUrl);
  await boss.start();
  await boss.createQueue(RUN_QUEUE);
}

export async function enqueueJob(jobId: string, priority: number): Promise<void> {
  if (!boss) return; // queue not initialized (e.g. DB down) — Phase 3 owns dispatch
  await boss.send(RUN_QUEUE, { jobId }, { priority });
}
