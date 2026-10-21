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

/** engine -> worker docker image
 *  @returns {Record<string, string>} */
const engineImage = () => ({
  mcx: config.workerImage,
  mmc: config.workerImageMmc,
  redbird: config.workerImageRedbird,
});

/**
 * Launch one simulation job as a swarm service. `script` runs as the container command
 * (bash -c); it fetches input from and pushes results to the API. The `engine` selects the
 * worker image and is exported so the script picks the matching simulator binary — and it
 * also selects the resource claim: the MC engines (mcx/mmc) take one GPU generic-resource,
 * while redbird is a CPU-only FEM solve and reserves cores instead.
 * @param {{ name: string, jobId: string, seed: boolean, script: string, engine?: string }} opts
 * @returns {Promise<void>}
 */
export async function createMcxService({ name, jobId, seed, script, engine = 'mcx' }) {
  const isRedbird = engine === 'redbird';
  // mmc (OpenCL JIT) is driver-sensitive; when set, pin mesh jobs to verified nodes.
  // redbird is CPU-only FEM assembly/solve — pin to high-core-count nodes instead, via
  // its own constraint (never falls back to the GPU-oriented workerConstraint).
  const constraint = isRedbird
    ? config.workerConstraintRedbird
    : (engine === 'mmc' && config.workerConstraintMmc) || config.workerConstraint;
  const args = [
    'service', 'create', '--detach',
    // on-failure (not none): a container exiting non-zero (bad GPU, phantom generic-resource
    // UUID, transient driver error, ...) gets a fresh scheduling attempt — new task, empty
    // NodeID, full node reselection — rather than leaving the job permanently failed for what
    // may be a one-off resource glitch. See config.js restartMaxAttempts for why this doesn't
    // deterministically loop back onto the same broken GPU.
    '--restart-condition', 'on-failure',
    '--restart-max-attempts', String(config.restartMaxAttempts),
    '--restart-delay', `${config.restartDelayMs}ms`,
    // redbird's forward solve is CPU-bound, not GPU-bound: claim CPU cores instead of a
    // GPU generic-resource, so it doesn't compete with mcx/mmc for the (scarce) GPU pool
    ...(isRedbird
      ? ['--reserve-cpu', String(config.redbirdCpuReserve), '--limit-cpu', String(config.redbirdCpuLimit)]
      : ['--generic-resource', 'NVIDIA_GPU=1']),
    ...(constraint ? ['--constraint', constraint] : []),
    '--name', name,
    '-e', `API_URL=${config.workerApiUrl}`,
    '-e', `JOB_ID=${jobId}`,
    '-e', `WORKER_SECRET=${config.workerSecret}`,
    '-e', `SEEDFLAG=${seed ? '--seed -1' : ''}`,
    '-e', `ENGINE=${engine}`,
    // persist the CUDA/OpenCL JIT cache across ephemeral containers on this node — avoids
    // re-paying a ~10s PTX recompile on every job (see config.js nvCachePath). Meaningless
    // for redbird (no GPU kernel involved), so skip the mount/env entirely for it.
    ...(!isRedbird && config.nvCachePath
      ? ['--mount', `type=bind,source=${config.nvCachePath},destination=${config.nvCachePath}`,
        '-e', `CUDA_CACHE_PATH=${config.nvCachePath}`]
      : []),
    engineImage()[engine] ?? config.workerImage,
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
