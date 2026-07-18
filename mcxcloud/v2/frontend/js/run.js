// @ts-check
import { $, downloadLink } from './util.js';
import { submitJob, cancelJob, fetchOutput, fetchDetphoton, streamJob } from './api.js';
import { state } from './state.js';

const USER_FIELDS = ['fullname', 'email', 'inst', 'netname'];
/** @type {(() => void)|null} */
let closeStream = null;

function loadUser() {
  try {
    const u = JSON.parse(localStorage.getItem('mcx.user') || '{}');
    for (const f of USER_FIELDS) if (u[f]) /** @type {HTMLInputElement} */ ($('#' + f)).value = u[f];
    state.user = u;
  } catch { /* ignore */ }
}

function saveUser() {
  /** @type {Record<string,string>} */
  const u = {};
  for (const f of USER_FIELDS) u[f] = /** @type {HTMLInputElement} */ ($('#' + f)).value.trim();
  state.user = u;
  localStorage.setItem('mcx.user', JSON.stringify(u));
  return u;
}

function log(line) {
  const t = new Date().toLocaleTimeString();
  state.log += `${t}  ${line}\n`;
  const box = /** @type {HTMLTextAreaElement} */ ($('#run-log'));
  box.value = state.log;
  box.scrollTop = box.scrollHeight;
}

/** append the mcx run log, stripping ANSI color escapes (like v1) */
function showMcxLog(text) {
  if (!text) return;
  const clean = String(text).replace(/\x1b\[[0-9;?]*[A-Za-z]/g, '').trimEnd();
  log('––––– mcx log –––––\n' + clean);
}

function resetButton() {
  const btn = $('#run-btn');
  btn.textContent = 'Submit';
  btn.onclick = submit;
}

async function submit() {
  if (!state.doc || !state.valid) { alert('input JSON is not valid'); return; }
  const user = saveUser();
  if (USER_FIELDS.some((f) => !user[f])) { alert('all identity fields are required'); return; }
  if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(String(user.email || ''))) {
    alert('please enter a valid email address');
    return;
  }

  state.log = '';
  log('submitting…');
  try {
    const { id, token, status, cached } = await submitJob(state.doc, user);
    state.jobId = id;
    state.token = token;
    state.status = status;
    log(`submitted job ${id}${cached ? ' (cached result)' : ''}`);

    const btn = $('#run-btn');
    btn.textContent = 'Cancel';
    btn.onclick = cancel;

    closeStream = streamJob(id, token, onEvent);
  } catch (err) {
    log('submit failed: ' + (/** @type {Error} */ (err)).message);
  }
}

/** @param {string} event @param {any} data */
async function onEvent(event, data) {
  if (event === 'status') { state.status = data.status; log('status: ' + data.status + (data.queuePos != null ? ` (queue #${data.queuePos})` : '')); }
  else if (event === 'log' && data.line) log(data.line);
  else if (event === 'progress') log(`progress: ${data.percent ?? '?'}%`);
  else if (event === 'error') { log(`${data.status || 'error'}: ${data.message || ''}`); showMcxLog(data.log); resetButton(); }
  else if (event === 'complete') { await onComplete(data); }
}

/** @param {any} data */
async function onComplete(data) {
  log(`completed (${data.runtime ?? '?'} s)`);
  showMcxLog(data.log);
  resetButton();
  try {
    const out = await fetchOutput(state.jobId, state.token);
    const outStr = JSON.stringify(out);
    downloadLink($('#run-output'), outStr, 'output.jnii');
    downloadLink($('#pv-output'), outStr, 'output.jnii'); // Preview-panel copy
    $('#draw-output').removeAttribute('disabled');
    if (data.hasDetphoton) {
      state.hasDetphoton = true;
      const detp = await fetchDetphoton(state.jobId, state.token);
      const detpStr = JSON.stringify(detp);
      downloadLink($('#run-detp'), detpStr, 'detphoton.jdt');
      downloadLink($('#pv-detp'), detpStr, 'detphoton.jdt'); // Preview-panel copy
    }
    // set state.output LAST so the app's subscriber auto-renders + downloads with the
    // download links already populated
    state.output = out;
    log('output received — rendering (use the Download links to save)');
  } catch (err) {
    log('failed to fetch output: ' + (/** @type {Error} */ (err)).message);
  }
}

async function cancel() {
  if (!state.jobId || !state.token) return;
  try {
    await cancelJob(state.jobId, state.token);
    if (closeStream) closeStream();
    log('cancelled');
  } catch (err) {
    log('cancel failed: ' + (/** @type {Error} */ (err)).message);
  }
  resetButton();
}

export function initRun() {
  loadUser();
  resetButton();
}
