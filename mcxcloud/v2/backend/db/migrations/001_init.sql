-- MCX Cloud v2 — initial schema (see ../../V2_DESIGN.md §6, ../../contracts/normalization.md)
-- Portable to Postgres 9.5+ : primary-key UUIDs are minted in the application (Node's
-- crypto.randomUUID()) and passed on INSERT, so no gen_random_uuid()/pgcrypto is needed
-- and no minimum Postgres major is imposed by the schema.

-- Content-addressed blob store: the dedup core. Every heavy JData array (and every
-- output) is stored here exactly once, keyed by sha256; clones share one row.
create table if not exists blobs (
  hash        text primary key,               -- 'sha256/<hex>'
  size        integer not null,
  encoding    text,                           -- e.g. 'zlib' (from _ArrayZipType_), or null
  refcount    integer not null default 0,
  data        bytea not null,                 -- canonical JSON bytes; Postgres TOASTs large values
  created_at  timestamptz not null default now()
);

-- Which owner references which blob (drives refcount/GC).
create table if not exists blob_refs (
  hash        text not null references blobs(hash) on delete cascade,
  owner_kind  text not null,                  -- 'job' | 'library'
  owner_id    uuid not null,
  primary key (hash, owner_kind, owner_id)
);

-- Job queue + results. input_doc is the NORMALIZED (small) document.
create table if not exists jobs (
  id          uuid primary key,
  input_doc   jsonb not null,                 -- heavy fields replaced by {"_DataLink_":"cas:sha256/…"}
  doc_hash    text not null,                  -- 'sha256/<hex>' over the normalized doc (whole-record cache key)
  status      text not null default 'queued',
  priority    real not null default 50,
  submitter   jsonb,                          -- {fullname,email,inst,netname}
  token_hash  text not null,                  -- sha256(capability token)
  output_hash text references blobs(hash),
  detp_hash   text references blobs(hash),
  log         text,
  runtime     real,
  error       text,
  gpu         text,
  node        text,
  created_at  timestamptz not null default now(),
  started_at  timestamptz,
  ended_at    timestamptz
);
create index if not exists jobs_sched   on jobs (status, priority desc, created_at);
create index if not exists jobs_dochash on jobs (doc_hash);

-- Public shared-simulation library (v1 mcxpub). input_doc is normalized.
create table if not exists library (
  id             uuid primary key,
  title          text not null,
  description    text not null,
  license        text not null,
  submitter      jsonb,
  input_doc      jsonb not null,
  doc_hash       text not null,
  thumbnail_hash text references blobs(hash),
  upvotes        integer not null default 0,
  downvotes      integer not null default 0,
  read_count     integer not null default 0,
  run_count      integer not null default 0,
  created_at     timestamptz not null default now()
);
-- Full-text search over title/description (replaces v1's LIKE '%...%').
create index if not exists library_fts on library
  using gin (to_tsvector('english', coalesce(title,'') || ' ' || coalesce(description,'')));
