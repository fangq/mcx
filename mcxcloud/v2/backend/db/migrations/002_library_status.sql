-- Library moderation: shared submissions start as 'pending' and are hidden from the public
-- Browse until an admin approves them.
alter table library add column if not exists status text not null default 'pending';
create index if not exists library_status on library (status);

-- One-time seed: approve the 12 most-run existing entries so there is a public library to
-- show, and leave the rest as 'pending' (review needed). The guard makes this a no-op on
-- every subsequent startup: once any row is approved, no rows match.
update library set status = 'approved'
 where id in (select id from library order by run_count desc nulls last, created_at limit 12)
   and not exists (select 1 from library where status = 'approved');
