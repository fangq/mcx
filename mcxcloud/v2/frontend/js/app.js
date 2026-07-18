// @ts-check
import { $, $$, downloadLink, encodeStateToUrl, decodeStateFromUrl } from './util.js';
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
  document.body.classList.toggle('preview-full', name === 'preview'); // let Preview use the full window width
  if (name === 'preview') window.dispatchEvent(new Event('resize'));   // canvas reflows to the new width
  if (name === 'preview' && !previewReady) {
    initPreview(); // canvas has real dimensions only once visible
    previewReady = true;
    if (state.doc) drawPreview(state.doc);
  }
}

function wireTabs() {
  // one delegated handler for the top nav, the home-page tiles, and the logo — anything
  // carrying data-tab (closest() so clicks on inner <svg>/<span> still resolve)
  document.body.addEventListener('click', (e) => {
    const el = /** @type {HTMLElement} */ (e.target).closest('[data-tab]');
    if (el && el.dataset.tab) showTab(el.dataset.tab);
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

function wireAutoOutput() {
  // when a job's output arrives, show it in Preview and download the .jnii automatically
  // (no need to click "Draw Output" or the download link)
  subscribe((key) => {
    if (key !== 'output' || !state.output) return;
    showTab('preview');
    drawPreview(state.output);
    // download links are populated (Run tab + Preview panel); the user clicks to save —
    // do NOT auto-download.
  });
}

function wireThemes() {
  const apply = (t) => {
    document.documentElement.dataset.theme = t;
    try { localStorage.setItem('mcx.theme', t); } catch { /* ignore */ }
    $$('#themes .theme-btn').forEach((b) => b.classList.toggle('active', b.dataset.theme === t));
  };
  $$('#themes .theme-btn').forEach((b) => b.addEventListener('click', () => apply(/** @type {string} */ (b.dataset.theme))));
  let saved = 'ocean';
  try { saved = localStorage.getItem('mcx.theme') || 'ocean'; } catch { /* ignore */ }
  apply(saved);
}

function wireActions() {
  // Reset: reload with the default example (drops any ?data= state)
  $('#btn-reset').addEventListener('click', () => { window.location.href = window.location.pathname; });
  // Link: copy a shareable URL that carries the current input (gzip+base64 via pako)
  $('#btn-link').addEventListener('click', async () => {
    if (!state.doc) { alert('Load or create a simulation first.'); return; }
    const url = window.location.origin + window.location.pathname + '?tab=preview&data=' + encodeURIComponent(encodeStateToUrl(state.doc));
    try { await navigator.clipboard.writeText(url); alert('Shareable link copied to clipboard.'); }
    catch { window.prompt('Copy this shareable link:', url); }
  });
}

// On load, honor ?data= (a shared input) and ?tab= (open a tab).
function parseUrl() {
  const p = new URLSearchParams(window.location.search);
  const d = p.get('data');
  if (d) {
    try { setEditorValue(decodeStateFromUrl(d)); }
    catch (err) { console.error('could not decode ?data= link:', err); }
  }
  const t = p.get('tab');
  if (t) showTab(t);
}

// Draggable splitter between the render canvas (left) and controls (right).
function wirePreviewSplitter() {
  const bar = $('#preview-splitter');
  const layout = /** @type {HTMLElement | null} */ (document.querySelector('.preview-layout'));
  if (!bar || !layout) return;
  let dragging = false;
  const move = (clientX) => {
    const rect = layout.getBoundingClientRect();
    const pct = Math.max(25, Math.min(85, ((clientX - rect.left) / rect.width) * 100));
    layout.style.setProperty('--split', pct.toFixed(1) + '%');
    window.dispatchEvent(new Event('resize')); // three.js reflows the canvas to the new width
  };
  bar.addEventListener('pointerdown', (e) => {
    dragging = true; bar.classList.add('dragging');
    /** @type {Element} */ (bar).setPointerCapture(/** @type {PointerEvent} */ (e).pointerId);
    e.preventDefault();
  });
  bar.addEventListener('pointermove', (e) => { if (dragging) move(/** @type {PointerEvent} */ (e).clientX); });
  const end = (e) => {
    if (!dragging) return;
    dragging = false; bar.classList.remove('dragging');
    try { /** @type {Element} */ (bar).releasePointerCapture(/** @type {PointerEvent} */ (e).pointerId); } catch { /* ignore */ }
    window.dispatchEvent(new Event('resize'));
  };
  bar.addEventListener('pointerup', end);
  bar.addEventListener('pointercancel', end);
}

async function boot() {
  wireTabs();
  wireJsonTab();
  wirePreviewButtons();
  wirePreviewSplitter();
  wireActions();
  wireThemes();
  wireAutoOutput();
  initRun();
  initLibrary(showTab);

  try {
    const schema = await fetchSchema();
    state.schema = schema;
    initEditor($('#editor-form'), schema, (doc, valid) => {
      state.doc = doc;
      state.valid = valid;
    });
    parseUrl(); // load ?data=/?tab= now that the editor can accept a value
  } catch (err) {
    $('#editor-form').textContent = 'failed to load schema from the API: ' + (/** @type {Error} */ (err)).message;
  }
}

boot();
