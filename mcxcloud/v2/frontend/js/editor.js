// @ts-check
import * as JE from '@json-editor/json-editor';

// esm.sh may expose the class as a named or default export — be tolerant.
const JSONEditor = /** @type {any} */ (JE).JSONEditor || /** @type {any} */ (JE).default || /** @type {any} */ (JE);

/** @type {any} */
let editor = null;
/** value requested before the editor finished its async build; applied on 'ready' */
let pendingValue = null;

/**
 * (Re)build the schema-driven form.
 * @param {HTMLElement} container
 * @param {object} schema
 * @param {(doc: object, valid: boolean) => void} onChange
 * @param {object} [startval]
 */
export function initEditor(container, schema, onChange, startval) {
  if (editor) editor.destroy();
  pendingValue = null;
  editor = new JSONEditor(container, {
    schema,
    theme: 'html',
    iconlib: null,
    disable_edit_json: true,
    disable_properties: true,
    disable_collapse: false,
    show_opt_in: true,
    no_additional_properties: true,
    startval,
  });
  editor.on('ready', () => {
    if (pendingValue !== null) { editor.setValue(pendingValue); pendingValue = null; }
  });
  editor.on('change', () => {
    const errors = editor.validate();
    onChange(editor.getValue(), errors.length === 0);
  });
  return editor;
}

/** @param {object} value */
export function setEditorValue(value) {
  if (!editor) return;
  // json-editor throws ("not ready yet") if setValue runs before the async build completes
  // (e.g. a ?data= link applied during boot) — queue it and let the 'ready' handler apply it.
  if (editor.ready) editor.setValue(value);
  else pendingValue = value;
}

export function getEditor() {
  return editor;
}
