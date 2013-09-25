%==========================================================================
%  wide-field source tests
%==========================================================================

%% test group 1

% a regular pencil beam at the center of the volume
clear cfg;
cfg.nphoton=1e7;
cfg.vol=uint8(ones(60,60,60));
cfg.srcpos=[30 30 30];
cfg.srcdir=[0 0 1];
cfg.gpuid=1;
cfg.autopilot=1;
cfg.prop=[0 0 1 1;0.005 1 0.8 1.37];
cfg.tstart=0;
cfg.tend=5e-9;
cfg.tstep=5e-9;
%cfg.printnum=10;
cfg.seed=99999;
cfg.srctype='pencil';
flux=mcxlab(cfg);
fcw=flux.data*cfg.tstep;

figure;
subplot(221);
imagesc(log10(abs(squeeze(fcw(:,30,:)))))
axis equal; colorbar
title('pencil beam at volume center');

% an isotropic source at the center of the volume
cfg.srctype='isotropic';
flux=mcxlab(cfg);
fcw=flux.data*cfg.tstep;
subplot(222);
imagesc(log10(abs(squeeze(fcw(:,30,:)))))
axis equal; colorbar
title('isotropic source at volume center');

% a pencil beam outside the volume
cfg.srctype='pencil';
cfg.srcpos=[30 30 -10];
flux=mcxlab(cfg);
fcw=flux.data*cfg.tstep;
subplot(223);
hs=slice(log10(abs(double(fcw))),[],30,1);
set(hs,'linestyle','none');
axis equal; colorbar
title('pencil beam launched from outside the volume');

% an isotropic source outside the volume
cfg.srctype='isotropic'
cfg.srcpos=[30 30 -10];
cfg.tend=1e-9;
cfg.tstep=1e-9;
flux=mcxlab(cfg);
fcw=flux.data*cfg.tstep;
subplot(224);
hs=slice(log10(abs(double(fcw))),[],30,1);
set(hs,'linestyle','none');
axis equal; colorbar
title('isotrpoic source at [30 30 -10]');

%% test group 2

clear cfg;
figure;
cfg.nphoton=1e8;
cfg.vol=uint8(ones(60,60,60));
cfg.srcdir=[0 0 1];
cfg.gpuid=1;
cfg.autopilot=1;
cfg.prop=[0 0 1 1;0.005 1 0.8 1.37];
cfg.tstart=0;
cfg.seed=99999;

% a uniform planar source outside the volume
cfg.srctype='planar';
cfg.srcpos=[10 10 -1];
cfg.srcparam1=[40 0 0 0];
cfg.srcparam2=[0 40 0 0];
cfg.tend=0.4e-11;
cfg.tstep=0.4e-11;
flux=mcxlab(cfg);
fcw=flux.data*cfg.tstep;
subplot(221);
imagesc(log10(abs(squeeze(fcw(:,:,1)))))
axis equal; colorbar
title('a uniform planar source');

cfg.srctype='fourier';
cfg.srcparam1=[40 10 0 2];
cfg.srcparam2=[0 40 0 2];
flux=mcxlab(cfg);
fcw=flux.data*cfg.tstep;
subplot(222);
imagesc(log10(abs(squeeze(fcw(:,:,1)))))
axis equal; colorbar
title('an SFDI source (2,2) in a quadrilateral');

cfg.srctype='fourier';
cfg.srcpos=[0 0 70];
cfg.srcdir=[0 0 -1];
cfg.srcparam1=[60 0 0 2];
cfg.srcparam2=[0 60 0 1];
cfg.tend=1e-9;
cfg.tstep=1e-9;
flux=mcxlab(cfg);
fcw=flux.data*cfg.tstep;
subplot(223);
hs=slice(log10(abs(double(fcw))),1,1,60);
set(hs,'linestyle','none');
axis equal; colorbar;box on;
title('a spatial frequency domain source (2,1)');

mcximg=[0 0 0 0 0 0 0 0 0 0 0
0 1 1 0 0 0 0 0 1 1 0
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
cfg.srctype='pattern';
cfg.srcpattern=mcximg;
cfg.srcpos=[-10*sqrt(2) 0 40];
cfg.srcdir=[1 1 0]/sqrt(2);
cfg.srcparam1=[20/sqrt(2) -20/sqrt(2) 0 size(mcximg,1)];
cfg.srcparam2=[0 0 -15 size(mcximg,2)];
cfg.tend=2e-10;
cfg.tstep=2e-10;
cfg.voidtime=0;
flux=mcxlab(cfg);
fcw=flux.data*cfg.tstep;
subplot(224);
hs=slice(log10(abs(double(fcw))),1,1,60);
set(hs,'linestyle','none');
axis equal; colorbar
title('an arbitrary pattern source from an angle');

%% test group 3

clear cfg;
figure;
cfg.nphoton=1e8;
cfg.vol=uint8(ones(60,60,60));
cfg.srcdir=[0 0 1];
cfg.gpuid=1;
cfg.autopilot=1;
cfg.prop=[0 0 1 1;0.005 1 0.8 1.37];
cfg.tstart=0;
cfg.seed=99999;

% a uniform planar source outside the volume
cfg.srctype='fourierx';
cfg.srcpos=[10 10 -1];
cfg.srcparam1=[40 0 0 2.0];
cfg.srcparam2=[40 0 0 1.5];
cfg.tend=0.4e-11;
cfg.tstep=0.4e-11;
flux=mcxlab(cfg);
fcw=flux.data*cfg.tstep;
subplot(221);
imagesc(log10(abs(squeeze(fcw(:,:,1)))))
axis equal; colorbar
title('a general fourier source (2,1.5)');

% a uniform planar source outside the volume
cfg.srctype='fourierx2d';
cfg.srcpos=[10 10 -1];
cfg.srcparam1=[40 0 0 1.5];
cfg.srcparam2=[40 pi/2 pi/4 3];
cfg.tend=0.4e-11;
cfg.tstep=0.4e-11;
flux=mcxlab(cfg);
fcw=flux.data*cfg.tstep;
subplot(222);
imagesc(log10(abs(squeeze(fcw(:,:,1)))))
axis equal; colorbar
title('a general 2d Fourier pattern (1.5,3)');
%% test group 4

clear cfg;
figure;
cfg.nphoton=1e7;
cfg.vol=uint8(ones(60,60,60));
cfg.gpuid=1;
cfg.autopilot=1;
cfg.prop=[0 0 1 1;0.005 1 0.8 1.37];
cfg.tstart=0;
cfg.seed=99999;

% a cone beam
cfg.srctype='cone';
cfg.srcpos=[30 30 -10];
cfg.srcdir=[0 0 1];
cfg.tend=5e-11;
cfg.tstep=5e-11;
cfg.srcparam1=[pi/6 0 0 0];
cfg.srcparam2=[0 0 0 0];
flux=mcxlab(cfg);
fcw=flux.data*cfg.tstep;
subplot(221);
imagesc(log10(abs(squeeze(fcw(:,:,1)))))
axis equal; colorbar
title('a uniform cone beam (uniform solid-angle)');

% a beam with arcsine distribution profile
cfg.srctype='arcsine';
flux=mcxlab(cfg);
fcw=flux.data*cfg.tstep;
subplot(222);
imagesc(log10(abs(squeeze(fcw(:,:,1)))))
axis equal; colorbar
title('an arcsine-distribution beam');

% a uniform disk source
cfg.srctype='disk';
cfg.srcparam1=[20 0 0 0];
cfg.srcparam2=[0 0 0 0];
flux=mcxlab(cfg);
fcw=flux.data*cfg.tstep;
subplot(223);
imagesc(log10(abs(squeeze(fcw(:,:,1)))))
axis equal; colorbar
title('a uniform disk source');

% a gaussian beam source
cfg.srctype='gaussian';
cfg.srcparam1=[10 0 0 0];
cfg.srcparam2=[0 0 0 0];
cfg.tend=5e-11;
cfg.tstep=5e-11;
flux=mcxlab(cfg);
fcw=flux.data*cfg.tstep;
subplot(224);
imagesc(log10(abs(squeeze(fcw(:,:,1)))))
axis equal; colorbar
title('a gaussian beam source');

%% test group 5

%debug flag to retrieve/test build-in RNG
cfg.vol=uint8(ones(100,100,100));
cfg.debuglevel='R';
cfg.isnormalized=0;
flux=mcxlab(cfg);
rng=flux.data(:);
figure
subplot(211);
hist(rng,1000);
title('raw RNG distribution');
uniformrng=acos(2*rng-1)/pi;
subplot(212);
hist(uniformrng,1000);
title('converted to uniform 0-1 distribution');
cfg=rmfield(cfg,'debuglevel');
