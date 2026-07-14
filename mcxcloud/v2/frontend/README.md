# MCX Cloud v2 — frontend

No-build, framework-free web app (native ES modules + import maps). See
[`../V2_DESIGN.md`](../V2_DESIGN.md) §8. Copy this folder onto any static host
(IONOS/mcx.space); it talks to the v2 backend API over HTTPS.

## Run

It must be **served over http(s)** (native multi-file ESM does not load from `file://`):

```sh
python3 -m http.server 8000        # then open http://localhost:8000
```

Point it at your API by setting `window.MCX_API_BASE` (default is the kwafoo manager) —
edit the one-line `<script>` in `index.html`, or serve the frontend same-origin behind a
reverse proxy that exposes the API at `/api`.

## Structure

```
index.html    shell: <importmap> (three, @json-editor, pako) + tabs + <script module app.js>
style.css     modern CSS (custom properties)
js/
  app.js      boot + tab wiring + glue
  state.js    reactive Proxy store (single source of truth)
  api.js      fetch (REST) + EventSource (SSE)
  util.js     DOM helpers + self-contained JData volume decode (pako) + colormap LUT
  editor.js   @json-editor/json-editor, schema fetched from the API
  library.js  Browse / Share tabs
  run.js      submit -> SSE log/progress -> output download
  preview.js  three.js volume raycaster (GLSL3) + shapes/source/detector, cross-sections
```

Dependencies are pinned in the import map and loaded from CDN ESM; there is **no
`node_modules`, no bundler, no transpile**. Volume rendering is self-contained: the packed
JData array is inflated with `pako` and the colormap is generated in code (no `jdata`,
`numjs`, or external PNG fetch that v1 relied on).

## Verification status

All modules parse as valid ES2022 ESM. **Not yet run in a browser against a live API**
(the dev sandbox has no browser/DOM/WebGL). Expected QA focus points:
- `@json-editor/json-editor` ESM interop (named vs default export) and the `html` theme;
- the three.js volume raycaster (GLSL3 shader compile, `Data3DTexture`, and the volume
  **axis order** fed to the texture — v1 applied a transpose that this port does not);
- SSE/CORS against the deployed API origin.
