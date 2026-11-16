// @ts-check
// LLM tab: describe a simulation in plain language, get an MCX input document back.
// The conversion itself (MCXLLM + a pydantic/langchain post-processor) runs on the server.
// Local ollama models need nothing from the user; hosted OpenAI models run on the user's
// own API key, typed here, kept only in this input (not in state, not in storage) and sent
// with each request. See backend/src/routes/llm.js.

import { $ } from './util.js';
import { generateFromPrompt, llmHealth } from './api.js';
import { setEditorValue } from './editor.js';
import { state } from './state.js';

/** @type {((name: string) => void)|null} */
let goToTab = null;
let busy = false;
/** @type {{ backend: string, model: string, host?: string, needsKey?: boolean }[]} */
let choices = [];

/** the choice currently selected in the model picker */
function selected() {
  return choices[Number(/** @type {HTMLSelectElement} */ ($('#llm-model')).value) || 0];
}

/**
 * Append a timestamped line to the status box (mirrors the Run tab's log()).
 * @param {string} line
 */
function log(line) {
  const t = new Date().toLocaleTimeString();
  state.llmLog += `${t}  ${line}\n`;
  const box = /** @type {HTMLTextAreaElement} */ ($('#llm-log'));
  box.value = state.llmLog;
  box.scrollTop = box.scrollHeight;
}

async function generate() {
  if (busy) return;
  const prompt = /** @type {HTMLTextAreaElement} */ ($('#llm-prompt')).value.trim();
  if (!prompt) { alert('describe the simulation you want first'); return; }

  const choice = selected();
  const keyBox = /** @type {HTMLInputElement} */ ($('#llm-key'));
  const apiKey = choice?.needsKey ? keyBox.value.trim() : undefined;
  if (choice?.needsKey && !apiKey) {
    state.llmLog = '';
    log(`${choice.model} runs on your own OpenAI API key — enter it above, or pick a local model.`);
    keyBox.focus();
    return;
  }
  const btn = /** @type {HTMLButtonElement} */ ($('#llm-btn'));
  busy = true;
  btn.disabled = true;
  btn.textContent = 'Generating…';
  state.llmLog = '';
  log(`sending the prompt to ${choice ? choice.model : 'the model'}…`);
  // Self-hosted models are a few times slower than OpenAI's; say so rather than let a
  // 40 s wait look like a hang.
  log(choice?.backend === 'ollama' ? 'a self-hosted model can take 15–60 s.' : 'this usually takes 5–15 s.');

  try {
    const { doc, elapsed, backend, model, filled, warnings, valid, errors } =
      await generateFromPrompt(prompt, choice, apiKey);
    log(`${model || 'model'} (${backend}) answered in ${elapsed ?? '?'} s`);
    if (filled?.length) {
      // MCXLLM predates two schema-required keys; the server supplies their defaults.
      log(`filled in schema defaults: ${filled.join(', ')}`);
    }
    for (const w of warnings || []) log('warning: ' + w);
    if (valid) {
      log('the generated input passes the MCX schema.');
    } else {
      log('warning: the generated input does NOT pass the MCX schema —');
      for (const e of errors || []) log('  · ' + e);
      log('loading it anyway so you can fix it here.');
    }

    // Push through the editor, not straight into state: JSON-Editor's change event is what
    // sets state.doc/state.valid, which is what fills the JSON textarea and the Create form.
    setEditorValue(doc);
    log('loaded — showing the JSON tab.');
    if (goToTab) goToTab('json');
  } catch (err) {
    log('failed: ' + (/** @type {Error} */ (err)).message);
  } finally {
    busy = false;
    btn.disabled = false;
    btn.textContent = 'Generate JSON';
  }
}

/**
 * Put the example prompt into the box, ready to edit or generate. The example is read from
 * the textarea's placeholder (minus its "e.g. "), so index.html holds it in one place.
 */
function useExample() {
  const box = /** @type {HTMLTextAreaElement} */ ($('#llm-prompt'));
  box.value = box.placeholder.replace(/^e\.g\.\s*/i, '');
  box.focus();
  box.setSelectionRange(box.value.length, box.value.length);
}

/** Sync the parts of the tab that depend on the selected model. */
function onModelChange() {
  const c = selected();
  if (!c) return;
  $('#llm-key-row').toggleAttribute('hidden', !c.needsKey);
}

/**
 * Populate the model picker from what the server actually offers. Shown only when there
 * is a choice to make — one configured model needs no dropdown.
 * @param {{ backend: string, model: string, host?: string, needsKey?: boolean }[]} list
 * @param {string} [dflt] default backend id @param {string} [dfltModel] default model
 */
function fillModels(list, dflt, dfltModel) {
  const sel = /** @type {HTMLSelectElement} */ ($('#llm-model'));
  sel.innerHTML = '';
  list.forEach((c, i) => {
    const o = document.createElement('option');
    o.value = String(i);
    o.textContent = `${c.model} (${c.backend}${c.needsKey ? ', your API key' : ''})`;
    sel.appendChild(o);
  });
  let i = list.findIndex((c) => c.backend === dflt && c.model === dfltModel);
  if (i < 0) i = list.findIndex((c) => c.backend === dflt);
  sel.value = String(Math.max(i, 0));
  $('#llm-model-row').toggleAttribute('hidden', list.length < 2);
  onModelChange();
}

/**
 * Wire the tab. Hides itself when the API has no LLM configured — the frontend and the
 * backend are deployed separately, so the capability has to be asked for, not assumed.
 * @param {(name: string) => void} showTab
 */
export async function initLlm(showTab) {
  goToTab = showTab;
  $('#llm-btn').addEventListener('click', generate);
  $('#llm-example').addEventListener('click', useExample);
  $('#llm-prompt').addEventListener('keydown', (e) => {
    const ev = /** @type {KeyboardEvent} */ (e);
    // Ctrl/Cmd+Enter submits, like most prompt boxes.
    if (ev.key === 'Enter' && (ev.ctrlKey || ev.metaKey)) { ev.preventDefault(); generate(); }
    // Tab accepts the placeholder example — but only in an empty box, so Tab still moves
    // focus out of a box that has text and keyboard navigation keeps working.
    const box = /** @type {HTMLTextAreaElement} */ (ev.target);
    if (ev.key === 'Tab' && !ev.shiftKey && !ev.ctrlKey && !ev.altKey && !ev.metaKey && !box.value.trim()) {
      ev.preventDefault();
      useExample();
    }
  });

  const health = await llmHealth();
  if (!health.enabled) {
    document.querySelectorAll('[data-tab="llm"]').forEach((el) => el.setAttribute('hidden', ''));
    return;
  }
  choices = health.choices;
  fillModels(choices, health.default, health.defaultModel);
  $('#llm-model').addEventListener('change', onModelChange);
}
