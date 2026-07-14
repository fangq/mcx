# MCX input schema (v2)

- **`mcx-input.v1.json`** — the MCX simulation-input JSON Schema (JSON-Editor / JSON-Schema
  draft used by v1). Verbatim extraction of the inline `defaultSchema` object from
  `../../v1/frontend/index.html` (v1, lines 597–1649), validated as well-formed JSON.

## Why it lives here

In v1 this ~1,050-line schema was pasted inline in the HTML. In v2 it is the **single
source of truth** shared by:

- the **frontend editor** — fetched at load via `GET /schema/mcx-input.v1` (no inline blob);
- the **backend** — Fastify validates every submitted `doc` against it before normalization.

## Structure (top level)

`Session`, `Forward`, `Optode` (Source/Detector), `Shapes`, `Domain`.

Heavy JData packed-array nodes (`_ArrayZipData_`, zlib+base64) occur at `Shapes` (the
`oneOf[1]` volume form) and `Optode.Source.Pattern`; these are what the normalizer extracts
to the content-addressed `blobs` store (see `../contracts/normalization.md`).

## Versioning

Filename carries the version (`.v1`). Breaking changes → `mcx-input.v2.json` + a new
`/schema/mcx-input.v2` route; the API keeps serving old versions for existing clients.
