cfg.nphoton=1e7;   
cfg.vol=uint8(ones(60,60,60));
cfg.srcpos=[30 30 1];
cfg.srcdir=[0 0 1];
cfg.gpuid=1;
cfg.autopilot=1;
cfg.prop=[0 0 1 1;0.005 1 1 0];
cfg.tstart=0;
cfg.tend=5e-9;
cfg.tstep=5e-10;

cfg.session='testmcx';

cfgs(1)=cfg;
cfgs(2)=cfg;
cfgs(2).session='testmcx2';

mcxlab(cfgs);
