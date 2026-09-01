-- Record the submitting client's IP so usage can be attributed by origin (the submitter
-- name/email/institution fields are free-text and unverified, so they cannot support that
-- on their own). The value was already computed for the per-client submit throttle in
-- routes/jobs.js clientIp() and then discarded.
--
-- inet, not text: it validates on write, stores compactly, and supports the network
-- operators an origin report actually needs (<<= for subnet grouping, family() to split
-- v4/v6). The caller passes NULL rather than a malformed value, so a spoofed or unparsable
-- X-Forwarded-File can never fail an INSERT and break job submission.
--
-- Nullable on purpose: rows predating this column, and any request whose forwarded address
-- does not parse, simply have no origin recorded.
--
-- NOTE: this is personal data. purge.js blanks a job's CONTENT columns but deliberately
-- keeps the row as a permanent usage record, and this column follows that rule -- so IPs
-- accumulate indefinitely alongside the submitter identity. If that retention is not
-- wanted, either coarsen on write (store a /24 or a country code) or add an age-based
-- scrub of this column to purgeOldJobs.
alter table jobs add column if not exists ip inet;

-- origin reports group by IP over a time range
create index if not exists jobs_ip on jobs (ip);
