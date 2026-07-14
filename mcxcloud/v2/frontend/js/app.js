// @ts-check
import { $, $$, downloadLink } from './util.js';
import { fetchSchema } from './api.js';
import { state, subscribe } from './state.js';
import { initEditor, setEditorValue } from './editor.js';
import { initPreview, drawPreview } from './preview.js';
import { initRun } from './run.js';
import { initLibrary } from './library.js';

let previewReady = false;

/** @param {string} name */
function showTab(name) {
  $$('#tabs button').forEach((b) => b.classList.toggle('active', b.dataset.tab === name));
  $$('.tab').forEach((s) => s.classList.toggle('active', s.id === 'tab-' + name));
  if (name === 'preview' && !previewReady) {
    initPreview(); // canvas has real dimensions only once visible
    previewReady = true;
    if (state.doc) drawPreview(state.doc);
  }
}

function wireTabs() {
  $('#tabs').addEventListener('click', (e) => {
    const t = /** @type {HTMLElement} */ (e.target).dataset.tab;
    if (t) showTab(t);
  });
}

function wireJsonTab() {
  // reflect editor state into the JSON textarea + validity badge
  subscribe((key) => {
    if (key === 'doc' || key === 'valid') {
      /** @type {HTMLTextAreaElement} */ ($('#json-text')).value = state.doc ? JSON.stringify(state.doc, null, 2) : '';
      const badge = $('#json-valid');
      badge.textContent = state.valid ? 'valid' : 'invalid';
      badge.className = 'badge ' + (state.valid ? 'ok' : 'bad');
      if (state.doc) downloadLink($('#json-download'), JSON.stringify(state.doc, null, 2), 'mcxinput.json');
    }
  });
  $('#json-update').addEventListener('click', () => {
    try {
      setEditorValue(JSON.parse(/** @type {HTMLTextAreaElement} */ ($('#json-text')).value));
    } catch (err) {
      alert('invalid JSON: ' + (/** @type {Error} */ (err)).message);
    }
  });
}

function wirePreviewButtons() {
  $('#draw-input').addEventListener('click', () => { if (state.doc) { showTab('preview'); drawPreview(state.doc); } });
  $('#draw-output').addEventListener('click', () => { if (state.output) { showTab('preview'); drawPreview(state.output); } });
}

async function boot() {
  wireTabs();
  wireJsonTab();
  wirePreviewButtons();
  initRun();
  initLibrary(showTab);

  try {
    const schema = await fetchSchema();
    state.schema = schema;
    initEditor($('#editor-form'), schema, (doc, valid) => {
      state.doc = doc;
      state.valid = valid;
    });
  } catch (err) {
    $('#editor-form').textContent = 'failed to load schema from the API: ' + (/** @type {Error} */ (err)).message;
  }
}

boot();
