import { EventEmitter } from 'node:events';

// In-process job event bus. Job progress and the worker completion callback both run
// in this API process, so an in-memory bus is sufficient. (Multi-process would move
// this to Postgres LISTEN/NOTIFY or the pg-boss pub/sub — noted for later.)
const bus = new EventEmitter();
bus.setMaxListeners(0);

export interface JobEvent {
  event: string; // 'status' | 'log' | 'progress' | 'complete' | 'error'
  data: Record<string, unknown>;
}

export function publish(jobId: string, event: string, data: Record<string, unknown>): void {
  bus.emit(jobId, { event, data: { jobid: jobId, ...data } } satisfies JobEvent);
}

export function subscribe(jobId: string, handler: (e: JobEvent) => void): () => void {
  bus.on(jobId, handler);
  return () => bus.off(jobId, handler);
}
