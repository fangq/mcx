-- Per-job simulator engine: 'mcx' (voxel domains) or 'mmc' (tetrahedral-mesh domains,
-- detected from Shapes.MeshNode at submit). Future engines (e.g. 'redbird') extend the
-- same column. Drives the worker image + command selection in the scheduler.
alter table jobs add column if not exists engine text not null default 'mcx';
