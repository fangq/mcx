// @ts-check
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { config } from './config.js';

const run = promisify(execFile);

/**
 * Total NVIDIA_GPU generic resources advertised across Ready+Active swarm nodes
 * (mirrors v1 mcxcloudd's GPU count). Best-effort; override with MAX_CONCURRENT.
 * @returns {Promise<number>}
 */
export async function countGpus() {
  const { stdout } = await run('docker', ['node', 'ls', '--format', '{{.ID}} {{.Status}} {{.Availability}}']);
  const ids = stdout
    .trim()
    .split('\n')
    .filter((l) => / Ready Active$/.test(l))
    .map((l) => l.split(' ')[0]);
  let count = 0;
  for (const id of ids) {
    const { stdout: res } = await run('docker', [
      'node', 'inspect', id,
      '--format',
      '{{range .Description.Resources.GenericResources}}{{if .NamedResourceSpec}}{{.NamedResourceSpec.Kind}} {{end}}{{end}}',
    ]);
    count += (res.match(/GPU/g) || []).length;
  }
  return count;
}

/**
 * Launch one mcx job as a swarm service, one GPU, no restart. `script` runs as the
 * container command (bash -c); it fetches input from and pushes results to the API.
 * @param {{ name: string, jobId: string, seed: boolean, script: string }} opts
 * @returns {Promise<void>}
 */
export async function createMcxService({ name, jobId, seed, script }) {
  const args = [
    'service', 'create', '--detach',
    '--restart-condition', 'none',
    '--generic-resource', 'NVIDIA_GPU=1',
    '--name', name,
    '-e', `API_URL=${config.workerApiUrl}`,
    '-e', `JOB_ID=${jobId}`,
    '-e', `WORKER_SECRET=${config.workerSecret}`,
    '-e', `SEEDFLAG=${seed ? '--seed -1' : ''}`,
    config.workerImage,
    '/bin/bash', '-c', script,
  ];
  await run('docker', args, { maxBuffer: 16 * 1024 * 1024 });
}

/** @param {string} name @returns {Promise<void>} */
export async function removeService(name) {
  try {
    await run('docker', ['service', 'rm', name]);
  } catch {
    /* already gone */
  }
}
