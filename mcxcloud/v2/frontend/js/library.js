// @ts-check
import { $ } from './util.js';
import { searchLibrary, loadLibraryEntry, shareLibrary, assetUrl } from './api.js';
import { state } from './state.js';
import { setEditorValue } from './editor.js';

/**
 * Wire the Browse and Share tabs.
 * @param {(tab: string) => void} showTab
 */
export function initLibrary(showTab) {
  const results = $('#browse-results');

  async function runSearch() {
    const q = /** @type {HTMLInputElement} */ ($('#browse-q')).value.trim();
    results.textContent = 'searching…';
    try {
      const cards = await searchLibrary(q);
      results.textContent = '';
      if (!cards.length) { results.textContent = 'no matches'; return; }
      for (const c of cards) results.appendChild(card(c));
    } catch (err) {
      results.textContent = 'search failed: ' + (/** @type {Error} */ (err)).message;
    }
  }

  /** @param {any} c */
  function card(c) {
    const el = document.createElement('div');
    el.className = 'card';
    const img = document.createElement('img');
    if (c.thumbnail) img.src = assetUrl(c.thumbnail);
    const h = document.createElement('h3');
    h.textContent = c.title;
    const p = document.createElement('p');
    p.textContent = c.description || '';
    const meta = document.createElement('div');
    meta.className = 'meta';
    const runs = document.createElement('span');
    runs.textContent = `${c.runCount ?? 0} runs · ${c.license || ''}`;
    const load = document.createElement('button');
    load.textContent = 'Load';
    load.onclick = () => loadEntry(c.id);
    meta.append(runs, load);
    el.append(img, h, p, meta);
    return el;
  }

  /** @param {string} id */
  async function loadEntry(id) {
    if (!confirm('Replace the current simulation with the selected one?')) return;
    try {
      const entry = await loadLibraryEntry(id);
      setEditorValue(entry.doc); // editor 'change' pushes doc/valid into state
      showTab('create');
    } catch (err) {
      alert('load failed: ' + (/** @type {Error} */ (err)).message);
    }
  }

  $('#browse-go').addEventListener('click', runSearch);
  $('#browse-q').addEventListener('keydown', (e) => {
    if (/** @type {KeyboardEvent} */ (e).key === 'Enter') runSearch();
  });

  $('#s-submit').addEventListener('click', async () => {
    if (!state.doc || !state.valid) { alert('the current input is not valid'); return; }
    const entry = {
      title: /** @type {HTMLInputElement} */ ($('#s-title')).value.trim(),
      description: /** @type {HTMLTextAreaElement} */ ($('#s-desc')).value.trim(),
      license: /** @type {HTMLSelectElement} */ ($('#s-license')).value,
      doc: state.doc,
      user: state.user,
    };
    if (!entry.title || !entry.description) { alert('title and description are required'); return; }
    try {
      await shareLibrary(entry);
      alert('shared — thank you!');
    } catch (err) {
      alert('share failed: ' + (/** @type {Error} */ (err)).message);
    }
  });
}
