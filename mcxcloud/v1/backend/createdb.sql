create table mcxcloud(
time integer primary key, name varchar(32), inst varchar(64), email varchar(64), netname varchar(32), json varchar(1024), jobid varchar(32), hash varchar(32), status integer, priority float, starttime integer, endtime integer, ip varchar(32)
);
create table mcxpub(
time integer primary key, title varchar(64), comment varchar(64), license varchar(10), name varchar(32), inst varchar(64), email varchar(64), netname varchar(32), json varchar(1024), hash varchar(32), thumbnail varchar(1024), createtime integer, upvote integer, downvote integer, readcount integer, runcount integer, ip varchar(32)
);


