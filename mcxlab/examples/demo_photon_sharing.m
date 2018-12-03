clear cfg;

mcximg=[0 1 1 0 0 0 0 0 1 1 0
0 0 0 1 1 0 1 1 0 0 0
0 0 0 0 0 1 0 0 0 0 0
0 0 0 1 1 0 1 1 0 0 0
0 1 1 0 0 0 0 0 1 1 0
0 0 0 0 0 0 0 0 0 0 0
0 0 1 0 0 0 0 0 1 0 0
0 1 0 0 0 0 0 0 0 1 0
0 1 0 0 0 0 0 0 0 1 0
0 1 0 0 0 0 0 0 0 1 0
0 0 1 1 1 1 1 1 1 0 0
0 0 0 0 0 0 0 0 0 0 0
0 1 1 1 1 1 1 1 1 1 0
0 0 0 1 0 0 0 0 0 0 0
0 0 0 0 1 1 0 0 0 0 0
0 0 0 1 0 0 0 0 0 0 0
0 1 1 1 1 1 1 1 1 1 0
0 0 0 0 0 0 0 0 0 0 0];

cfg.nphoton=1e7;
cfg.vol=uint8(ones(60,60,60));
cfg.srctype='pattern';
cfg.srcpattern=permute(reshape(mcximg,[6,3,size(mcximg,2)]),[2 1 3]);
cfg.srcnum=3;
cfg.srcpos=[-10*sqrt(2) 0 40];
cfg.srcdir=[1 1 0]/sqrt(2);
cfg.srcparam1=[20/sqrt(2) -20/sqrt(2) 0 size(mcximg,1)];
cfg.srcparam2=[0 0 -15 size(mcximg,2)];
cfg.tstart=0;
cfg.tend=2e-10;
cfg.tstep=2e-10;
cfg.voidtime=0;
cfg.gpuid=1;
cfg.autopilot=1;
cfg.prop=[0 0 1 1;0.005 1 0.8 1.37];
cfg.seed=99999;

flux=mcxlab(cfg);
fcw=flux.data*cfg.tstep;
fcw=sum(fcw,4);
subplot(224);
hs=slice(log10(abs(double(fcw))),1,1,60);
set(hs,'linestyle','none');
axis equal; colorbar
title('an arbitrary pattern source from an angle');
