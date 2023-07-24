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
cfg.vol(:,:,1)=0;
cfg.issaveref=1;
cfg.srctype='pattern';
cfg.srcpattern=permute(reshape(mcximg,[6,3,size(mcximg,2)]),[2 1 3]);
cfg.srcnum=3;
cfg.srcpos=[0 0 0];
cfg.issrcfrom0=1;
cfg.srcdir=[0 0 1];
cfg.srcparam1=[60 0 0 size(cfg.srcpattern,2)];
cfg.srcparam2=[0 60 0 size(cfg.srcpattern,3)];
cfg.tstart=0;
cfg.tend=5e-9;
cfg.tstep=5e-9;
cfg.voidtime=0;
cfg.gpuid=1;
cfg.autopilot=1;
cfg.prop=[0 0 1 1;0.005 1 0.8 1.37];
cfg.seed=99999;

% when photon sharing is used, the detpt.w0 output of detected photon stores the
% 1-D index of the pixel within a 2D pattern (col-major, starting from 0)
% where the photon was initially launched. using detpt.w0, one can look up
% the initial launch weight of each photon in all patterns

cfg.detpos=[25 30 0 2];
cfg.savedetflag='dpw';   

[flux, detpt]=mcxlab(cfg);

fcw=flux.data*cfg.tstep;
for i=1:3
    subplot(1,3,i);
    hs=slice(log10(abs(double(fcw(:,:,:,i)))),1,1,2);
    view([1 1 1])
    set(hs,'linestyle','none');
    axis equal; colorbar
    title(sprintf('pattern #%d',i));
end
figure;
fcw=flux.dref*cfg.tstep;
for i=1:3
    subplot(1,3,i);
    hs=slice(abs(double(fcw(:,:,:,i))),[],[],1);
    view([1 1 1])
    set(hs,'linestyle','none');
    axis equal; colorbar
    title(sprintf('pattern #%d',i));
end

figure;
hist(double(detpt.w0))
xlabel('1D pixel index of the launch position');
