# Content-Addressed Normalization & Reassembly (v2)

Solves the v1 database bloat (§2.5-3 of `V2_DESIGN.md`): large JData-encoded arrays are
stored inline in every simulation record, so cloned simulations duplicate 0.5–1 MB blobs
and the DB grows past 1 GB. v2 extracts those arrays into a content-addressed `blobs`
table (once per unique content) and leaves only a `_DataLink_` reference in the document.

## 1. What gets extracted

An **extractable data node** is any JSON object that carries a JData packed-array marker:

- `_ArrayZipData_` (zlib+base64 packed N-D array — the heavy case), or
- `_ArrayData_` (unpacked N-D array payload).

In real MCX inputs these appear at:

| Location | Field | Notes |
|---|---|---|
| `Shapes` | the `oneOf[1]` form — a single packed volume | primary bloat source |
| `Optode.Source.Pattern` | illumination pattern array | can be large |
| (outputs) | `output.jnii` `NIFTIData`, `output_detp.jdat` | same mechanism reused for results |

The rule is **structural, not path-based** — walk the whole document and extract any
qualifying node — so it also covers future fields and the JNIfTI/JData outputs. Plain JSON
arrays (e.g. `Grid.Size: [60,60,60]`) have no marker and are never extracted.

A node is extracted only if its canonical serialization is **≥ `THRESHOLD` (default 4096
bytes)** — tiny packed arrays stay inline (not worth a row/round-trip).

## 2. Canonicalization (deterministic hashing → maximal dedup)

To guarantee that identical content from different submissions hashes identically
(regardless of incoming whitespace or key order), define:

```
canon(node) = JSON serialization with object keys sorted lexicographically,
              no insignificant whitespace, UTF-8, no trailing newline.
hash(node)  = "sha256/" + hex( sha256( canon(node) ) )
```

MCX/JData is insensitive to object-key order, so re-ordering keys for canonicalization is
semantically safe and it is what lets two clones share one blob row.

## 3. Reference form (`_DataLink_`)

An extracted node is replaced **in place** by a JData-spec `_DataLink_` object:

```json
{ "_DataLink_": "cas:sha256/<hex>" }
```

- `cas:` is the internal content-addressed-store URI scheme (resolved by the API).
- The normalized document remains a **valid JData document** (a `_DataLink_` node is a
  standard external reference), satisfying the "spec-valid" decision.
- **On export** (a library entry fetched by an external/generic JData tool), links are
  rewritten to an absolute, resolvable URL: `cas:sha256/<hex>` →
  `https://<api-base>/blobs/sha256/<hex>` (that endpoint returns the canonical array JSON).

## 4. Normalize (write path)

```
normalize(doc):
  refs = []
  walk doc depth-first:
    if node is an extractable data node and size(canon(node)) >= THRESHOLD:
      h = hash(node)
      upsert blobs(hash=h, data=canon(node), encoding=node._ArrayZipType_ or 'raw',
                   size=..., refcount += 1)   # dedup: existing row just bumps refcount
      replace node with { "_DataLink_": "cas:" + h }
      refs.push(h)
  doc_hash = hash(doc)          # over the normalized doc → whole-record cache key
  return doc, doc_hash, refs
```

- `jobs.input_doc` / `library.input_doc` store the **normalized** (small) JSONB.
- `refs` drives refcounting; store the job↔blob edges so deletes can decrement.
- `doc_hash` reproduces v1's `md5_hex` whole-record dedup — a re-submitted clone maps to an
  existing completed result (unifies the `workspace/_<hash>` cache).

## 5. Reassemble (read path)

Before dispatching to MCX, and before returning a document for external use:

```
reassemble(doc):
  walk doc:
    if node == { "_DataLink_": "cas:sha256/<hex>" } (only key):
      blob = blobs.get(hex)          # 404 → 422 "missing blob <hex>"
      replace node with JSON.parse(blob.data)
  return doc
```

Reassembly is the exact inverse of normalize; MCX always receives a fully-inlined,
standard JData/MCX input.

## 6. Refcount & GC

- Insert (job or library entry): for each unique `ref`, `INSERT ... ON CONFLICT DO UPDATE
  SET refcount = refcount + 1`; record the edge.
- Delete/expire: decrement refcount for each edge; `DELETE FROM blobs WHERE refcount <= 0`
  (or a periodic sweep).
- Outputs are stored the same way (`jobs.output_hash` → `blobs`) so results dedup too.

## 7. Open parameters

- `THRESHOLD` default 4096 B — tune against real submissions.
- Whether to also extract *un-marked* large strings (base64 blobs not in JData wrapping) —
  currently no; all MCX heavy fields use JData array markers.
