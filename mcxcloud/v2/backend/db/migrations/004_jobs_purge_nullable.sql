-- Purged jobs are now blanked in place (content columns set to null) rather than deleted,
-- so the row survives as a permanent usage record (submitter/engine/status/timestamps) for
-- reporting — see purge.js. The three NOT NULL constraints from the original schema (valid
-- for a live job, which always has these) must be relaxed to allow that.
alter table jobs alter column input_doc drop not null;
alter table jobs alter column doc_hash drop not null;
alter table jobs alter column token_hash drop not null;
