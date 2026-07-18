// @ts-check
import { $ } from './util.js';
import {
  searchLibrary, loadLibraryEntry, shareLibrary, assetUrl,
  adminLogin, adminLogout, isAdmin, adminListLibrary, adminLoadEntry, adminApprove, adminReject, adminUpdate,
} from './api.js';
import { state } from './state.js';
import { setEditorValue } from './editor.js';
import { drawPreview, getThumbnail, captureThumbnail } from './preview.js';

/**
 * Format a library title/description like v1: HTML-escape first (safety), then apply v1's
 * light markup — newlines→<br>, indented lines→<pre>, "* item"→bullets, and auto-linked URLs.
 * @param {string} s @param {boolean} [multiline]
 */
function formatText(s, multiline) {
  let t = String(s || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  if (multiline) {
    t = t.replace(/\n +([^\n]*)\n/g, '<pre>$1</pre>')
      .replace(/\n\*([^\n]+)/g, '<ul><li>$1</li></ul>')
      .replace(/\n/g, '<br/>')
      .replace(/(https?:\/\/[^\s<]+)/g, '<a target="_blank" rel="noopener" href="$1">$1</a>');
  } else {
    t = t.replace(/\n/g, '<br/>');
  }
  return t;
}

/**
 * Wire the Browse and Share tabs.
 * @param {(tab: string) => void} showTab
 */
export function initLibrary(showTab) {
  const results = $('#browse-results');
  const detail = $('#browse-detail');
  const PAGE = 5;                 // load 5 at a time, like v1
  let currentQ = '';
  let offset = 0;
  /** @type {HTMLButtonElement | null} */
  let moreBtn = null;
  let adminStatus = 'pending';   // which submissions the review panel shows
  let editingLibId = '';         // non-empty => Share submit replaces this submission (admin)
  let editThumbDirty = false;    // admin captured a new thumbnail during this edit session

  function ensureMoreButton() {
    if (moreBtn) return moreBtn;
    moreBtn = document.createElement('button');
    moreBtn.id = 'browse-more';
    moreBtn.textContent = 'Show more';
    moreBtn.style.display = 'none';
    moreBtn.addEventListener('click', () => runSearch(true));
    results.after(moreBtn);
    return moreBtn;
  }

  /** @param {boolean} [append] false = new search (reset), true = next page */
  async function runSearch(append = false) {
    const btn = ensureMoreButton();
    if (!append) {
      const raw = /** @type {HTMLInputElement} */ ($('#browse-q')).value.trim();
      if (raw === '!review') {                 // secret UI trigger: reveal the admin login
        /** @type {HTMLInputElement} */ ($('#browse-q')).value = '';
        $('#admin-bar').hidden = false;
        /** @type {HTMLInputElement} */ ($('#admin-secret')).focus();
        return;
      }
      currentQ = raw;
      offset = 0;
      results.textContent = 'searching…';
    }
    btn.style.display = 'none';
    try {
      const cards = await searchLibrary(currentQ, PAGE, offset);
      if (!append) results.textContent = '';
      if (!append && !cards.length) { results.textContent = 'no matches'; return; }
      for (const c of cards) results.appendChild(card(c));
      offset += cards.length;
      btn.style.display = cards.length === PAGE ? '' : 'none'; // maybe more to fetch
    } catch (err) {
      if (!append) results.textContent = 'search failed: ' + (/** @type {Error} */ (err)).message;
    }
  }

  // A card is thumbnail-only (hover shows the title); clicking opens the side detail panel
  // so the grid layout isn't disturbed.
  /** @param {any} c @param {boolean} [review] render as a review card (distinct border) */
  function card(c, review) {
    const el = document.createElement('div');
    el.className = review ? 'card review' + (c.status === 'approved' ? ' ok' : '') : 'card';
    el.title = c.title; // hover tooltip
    const img = document.createElement('img');
    if (c.thumbnail) img.src = assetUrl(c.thumbnail);
    img.alt = c.title;
    el.appendChild(img);
    el.addEventListener('click', () => (review ? showReviewDetail(c, el) : showDetail(c, el)));
    return el;
  }

  /** show a selected simulation's details in the side panel @param {any} c @param {HTMLElement} el */
  function showDetail(c, el) {
    results.querySelectorAll('.card.selected').forEach((s) => s.classList.remove('selected'));
    el.classList.add('selected');
    detail.textContent = '';
    const close = document.createElement('button');
    close.className = 'ghost detail-close'; close.textContent = '×'; close.title = 'close';
    close.addEventListener('click', () => { detail.hidden = true; el.classList.remove('selected'); });
    const img = document.createElement('img'); img.className = 'thumb-view';
    if (c.thumbnail) img.src = assetUrl(c.thumbnail);
    const h = document.createElement('h3'); h.innerHTML = formatText(c.title, false);
    const p = document.createElement('p'); p.innerHTML = formatText(c.description, true);
    const meta = document.createElement('div'); meta.className = 'meta';
    meta.textContent = `${c.runCount ?? 0} runs · ${c.license || ''}`;
    const load = document.createElement('button'); load.textContent = 'Load this simulation';
    load.addEventListener('click', () => loadEntry(c.id));
    detail.append(close, img, h, p, meta, load);
    detail.hidden = false;
  }

  /** @param {string} id — load into the editor AND render the preview immediately. */
  async function loadEntry(id) {
    if (!confirm('Replace the current simulation with the selected one?')) return;
    try {
      const entry = await loadLibraryEntry(id);
      setEditorValue(entry.doc); // editor 'change' pushes doc/valid into state
      showTab('preview');        // makes the canvas visible + inits it once
      drawPreview(entry.doc);    // render now — no need to click "Draw Input"
      setTimeout(() => captureThumbnail(), 200); // auto-update the thumbnail on load
    } catch (err) {
      alert('load failed: ' + (/** @type {Error} */ (err)).message);
    }
  }

  // ---- admin / review ---------------------------------------------------------
  /** Fetch and render the submissions of the current review status into the grid. */
  async function loadReview() {
    detail.hidden = true;
    if (moreBtn) moreBtn.style.display = 'none';
    results.textContent = 'loading submissions…';
    try {
      const list = await adminListLibrary(/** @type {any} */ (adminStatus));
      results.textContent = '';
      if (!list.length) { results.textContent = `no ${adminStatus} submissions`; return; }
      for (const c of list) results.appendChild(card(c, true));
    } catch (err) {
      results.textContent = 'load failed: ' + (/** @type {Error} */ (err)).message;
    }
  }

  /** Review side panel: shows submitter info + Load & run / Approve / Edit / Reject. @param {any} c @param {HTMLElement} el */
  function showReviewDetail(c, el) {
    results.querySelectorAll('.card.selected').forEach((s) => s.classList.remove('selected'));
    el.classList.add('selected');
    detail.textContent = '';
    const close = document.createElement('button');
    close.className = 'ghost detail-close'; close.textContent = '×'; close.title = 'close';
    close.addEventListener('click', () => { detail.hidden = true; el.classList.remove('selected'); });
    const img = document.createElement('img'); img.className = 'thumb-view';
    if (c.thumbnail) img.src = assetUrl(c.thumbnail);
    const h = document.createElement('h3'); h.innerHTML = formatText(c.title, false);
    const p = document.createElement('p'); p.innerHTML = formatText(c.description, true);
    const s = c.submitter || {};
    const sub = document.createElement('div'); sub.className = 'submitter';
    sub.innerHTML = 'Submitted by ' + formatText(`${s.fullname || '?'} <${s.email || '?'}>${s.inst ? ', ' + s.inst : ''}`, false);
    const meta = document.createElement('div'); meta.className = 'meta';
    meta.textContent = `${c.status} · ${c.runCount ?? 0} runs · ${c.license || ''}`;
    const actions = document.createElement('div'); actions.className = 'detail-actions';
    const load = document.createElement('button'); load.className = 'ghost'; load.textContent = 'Load & run';
    load.addEventListener('click', () => adminLoad(c.id));
    const edit = document.createElement('button'); edit.className = 'ghost'; edit.textContent = 'Edit…';
    edit.addEventListener('click', () => startEdit(c.id));
    const approve = document.createElement('button'); approve.textContent = 'Approve';
    approve.hidden = c.status === 'approved';
    approve.addEventListener('click', () => reviewAction(() => adminApprove(c.id), `Approved "${c.title}".`));
    const reject = document.createElement('button'); reject.className = 'warn'; reject.textContent = 'Reject';
    reject.addEventListener('click', () => {
      if (!confirm(`Reject and permanently delete "${c.title}"?`)) return;
      reviewAction(() => adminReject(c.id), `Rejected "${c.title}".`);
    });
    actions.append(load, edit, approve, reject);
    detail.append(close, img, h, p, sub, meta, actions);
    detail.hidden = false;
  }

  /** Run an approve/reject action, then refresh the review list. @param {() => Promise<any>} fn @param {string} okMsg */
  async function reviewAction(fn, okMsg) {
    try {
      await fn();
      $('#admin-msg').textContent = okMsg;
      loadReview();
    } catch (err) {
      alert('action failed: ' + (/** @type {Error} */ (err)).message);
    }
  }

  /** Load a submission (via the authed endpoint) into the editor + preview to test it. @param {string} id */
  async function adminLoad(id) {
    try {
      const entry = await adminLoadEntry(id);
      setEditorValue(entry.doc);
      showTab('preview');
      drawPreview(entry.doc);
      setTimeout(() => captureThumbnail(), 200);
    } catch (err) {
      alert('load failed: ' + (/** @type {Error} */ (err)).message);
    }
  }

  /** Send a submission to the Share tab for admin-curated editing; resubmit replaces it. @param {string} id */
  async function startEdit(id) {
    try {
      const entry = await adminLoadEntry(id);
      editingLibId = id;
      editThumbDirty = false;
      /** @type {HTMLInputElement} */ ($('#s-title')).value = entry.title || '';
      /** @type {HTMLTextAreaElement} */ ($('#s-desc')).value = entry.description || '';
      /** @type {HTMLSelectElement} */ ($('#s-license')).value = entry.license || 'CC0';
      setEditorValue(entry.doc); // populates state.doc for the resubmit
      const s = /** @type {any} */ (entry).submitter || {};
      $('#s-edit-info').textContent =
        `Editing "${entry.title}" by ${s.fullname || '?'} <${s.email || '?'}> — resubmitting replaces the original ` +
        `(keeps the thumbnail unless you capture a new one; stays ${/** @type {any} */ (entry).status}).`;
      $('#s-edit-banner').hidden = false;
      showTab('share');
    } catch (err) {
      alert('edit load failed: ' + (/** @type {Error} */ (err)).message);
    }
  }

  $('#admin-unlock').addEventListener('click', async () => {
    const secret = /** @type {HTMLInputElement} */ ($('#admin-secret')).value;
    if (!secret) return;
    $('#admin-msg').textContent = 'authenticating…';
    try {
      await adminLogin(secret);
      /** @type {HTMLInputElement} */ ($('#admin-secret')).value = ''; // don't linger in the field
      $('#admin-login').hidden = true;
      $('#admin-review-controls').hidden = false;
      $('#admin-msg').textContent = '';
      loadReview();
    } catch (err) {
      $('#admin-msg').textContent = 'login failed: ' + (/** @type {Error} */ (err)).message;
    }
  });
  $('#admin-secret').addEventListener('keydown', (e) => {
    if (/** @type {KeyboardEvent} */ (e).key === 'Enter') $('#admin-unlock').click();
  });
  $('#admin-status').addEventListener('change', () => {
    adminStatus = /** @type {HTMLSelectElement} */ ($('#admin-status')).value;
    loadReview();
  });
  $('#admin-exit').addEventListener('click', () => {
    adminLogout();
    $('#admin-bar').hidden = true;
    $('#admin-login').hidden = false;
    $('#admin-review-controls').hidden = true;
    runSearch(false); // back to the public library
  });
  $('#s-edit-cancel').addEventListener('click', () => {
    editingLibId = '';
    $('#s-edit-banner').hidden = true;
  });

  $('#browse-go').addEventListener('click', () => runSearch(false));
  $('#browse-q').addEventListener('keydown', (e) => {
    if (/** @type {KeyboardEvent} */ (e).key === 'Enter') runSearch(false);
  });
  runSearch(false); // preload the first page so Browse isn't empty

  $('#s-update-thumb').addEventListener('click', () => {
    if (!captureThumbnail()) { alert('Open the Preview tab and draw a volume first, then update the thumbnail.'); return; }
    editThumbDirty = true; // so an admin edit sends the new thumbnail (otherwise the original is kept)
  });

  $('#s-submit').addEventListener('click', async () => {
    if (!state.doc || !state.valid) { alert('the current input is not valid'); return; }
    /** @type {any} */
    const entry = {
      title: /** @type {HTMLInputElement} */ ($('#s-title')).value.trim(),
      description: /** @type {HTMLTextAreaElement} */ ($('#s-desc')).value.trim(),
      license: /** @type {HTMLSelectElement} */ ($('#s-license')).value,
      doc: state.doc,
    };
    if (!entry.title || !entry.description) { alert('title and description are required'); return; }
    try {
      if (editingLibId && isAdmin()) {
        // admin-curated replace: keep the original thumbnail unless a new one was captured
        if (editThumbDirty) entry.thumbnail = getThumbnail() || undefined;
        await adminUpdate(editingLibId, entry);
        editingLibId = '';
        $('#s-edit-banner').hidden = true;
        alert('submission updated (replaced the original).');
        showTab('browse');
        loadReview();
      } else {
        entry.user = state.user;
        entry.thumbnail = getThumbnail() || undefined; // preview snapshot (see "Update thumbnail")
        await shareLibrary(entry);
        alert('shared — thank you! It will appear in the public library once an admin approves it.');
      }
    } catch (err) {
      alert('submit failed: ' + (/** @type {Error} */ (err)).message);
    }
  });
}
