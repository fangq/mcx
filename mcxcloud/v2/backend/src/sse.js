// @ts-check
import { EventEmitter } from 'node:events';

// In-process job event bus. Job progress and the worker completion callback both run
// in this API process, so an in-memory bus is sufficient. (Multi-process would move
// this to Postgres LISTEN/NOTIFY or pg-boss pub/sub — noted for later.)
const bus = new EventEmitter();
bus.setMaxListeners(0);

/**
 * @typedef {Object} JobEvent
 * @property {string} event   'status' | 'log' | 'progress' | 'complete' | 'error'
 * @property {Record<string, unknown>} data
 */

/**
 * @param {string} jobId
 * @param {string} event
 * @param {Record<string, unknown>} data
 */
export function publish(jobId, event, data) {
  bus.emit(jobId, { event, data: { jobid: jobId, ...data } });
}

/**
 * @param {string} jobId
 * @param {(e: JobEvent) => void} handler
 * @returns {() => void} unsubscribe
 */
export function subscribe(jobId, handler) {
  bus.on(jobId, handler);
  return () => bus.off(jobId, handler);
}
