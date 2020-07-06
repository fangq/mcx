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
cfg.nphoton=1e7;
cfg.vol=uint8(ones(60,60,60));
cfg.srcdir=[0 0 1 0];
cfg.gpuid=1;
cfg.autopilot=1;
cfg.prop=[0 0 1 1;0.005 1 0.8 1.37];
cfg.tstart=0;
cfg.seed=99999;

% a uniform planar source outside the volume
cfg.srctype='planar';
cfg.srcpos=[10 10 0];
cfg.srcparam1=[40 0 0 0];
cfg.srcparam2=[0 40 0 0];
cfg.tend=0.4e-10;
cfg.tstep=0.4e-10;
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
cfg.srcparam1=[40 0 0 40];
cfg.srcparam2=[2 1.5 0 0];
cfg.tend=0.4e-10;
cfg.tstep=0.4e-10;
flux=mcxlab(cfg);
fcw=flux.data*cfg.tstep;
subplot(221);
imagesc(log10(abs(squeeze(fcw(:,:,1)))))
axis equal; colorbar
title('a general Fourier source (2,1.5)');

% a uniform planar source outside the volume
cfg.srctype='slit';
cfg.srcpos=[10 30 0];
cfg.srcdir=[0 1 1]/sqrt(2);
cfg.srcparam1=[40 0 0 0];
cfg.srcparam2=[0 0 0 0];
cfg.prop=[0 0 1 1;0.005 0.1 0.9 1.37];
cfg.tend=5e-9;
cfg.tstep=5e-9;
flux=mcxlab(cfg);
fcw=flux.data*cfg.tstep;
subplot(222);
hs=slice(log10(abs(double(fcw))),[],[15 45],[1]);
set(hs,'linestyle','none');
axis equal; colorbar
title('a slit source');

cfg.nphoton=1e7;
cfg.vol=uint8(ones(60,60,60));
cfg.vol(25:35,25:35,25:35)=2;
cfg.prop=[0 0 1 1;0.005 0.01 0.8 1.37;0.01 1,0.8,1.37];
cfg.srctype='pattern';
cfg.srcpattern=mcximg(1:6,:);
cfg.srcpos=[-13 13 13];
cfg.srcdir=[1 0 0];
cfg.srcparam1=[0 30 0 size(cfg.srcpattern,1)];
cfg.srcparam2=[0 0 30 size(cfg.srcpattern,2)];
cfg.tend=0.3e-9;
cfg.tstep=0.1e-10;
cfg.voidtime=0;
flux=mcxlab(cfg);
fcw1=flux.data*cfg.tstep;

cfg.srcpattern=rot90(mcximg(7:12,:),3);
cfg.srcpos=[17 17 60+1];
cfg.srcdir=[0 0 -1];
cfg.srcparam1=[30 0 0 size(cfg.srcpattern,1)];
cfg.srcparam2=[0 30 0 size(cfg.srcpattern,2)];
flux=mcxlab(cfg);
fcw2=flux.data*cfg.tstep;

cfg.srcpattern=mcximg(13:end,:);
cfg.srcpos=[60-15 -1 60-15];
cfg.srcdir=[0 1 0];
cfg.srcparam1=[-30 0 0 size(cfg.srcpattern,1)];
cfg.srcparam2=[0 0 -30 size(cfg.srcpattern,2)];
flux=mcxlab(cfg);
fcw3=flux.data*cfg.tstep;

% fcw=fcw1+fcw2+fcw3;
%
% axis tight;
% set(gca,'nextplot','replacechildren','visible','off');
% set(gcf,'color','w');
% dd=log10(abs(double(squeeze(fcw(:,:,:,1)))));
% dd(isinf(dd))=-8;
% hs=slice(dd,1,1,60);
% set(hs,'linestyle','none');
% axis equal; set(gca,'clim',[-6.5 -3])
% box on;
% set(gca,'xtick',[]);set(gca,'ytick',[]);set(gca,'ztick',[]);
% drawnow;
% fm = getframe;
% [mcxframe,map] = rgb2ind(fm.cdata,256,'nodither');
% mcxframe(1,1,1,size(fcw1,4)) = 0;
% for i=1:size(fcw1,4)
%     dd=log10(abs(double(squeeze(fcw(:,:,:,i)))));
%     dd(isinf(dd))=-8;
%     hs=slice(dd,1,1,60);
%     set(hs,'linestyle','none');
%     axis equal; set(gca,'clim',[-6.5 -3])
%     set(gca,'xtick',[]);set(gca,'ytick',[]);set(gca,'ztick',[]);
%     axis on;box on;
%     drawnow;
%     fm = getframe;
%     mcxframe(:,:,1,i) = rgb2ind(fm.cdata,map,'nodither');
% end
% % movie(mcxframe,3,2);
% imwrite(mcxframe,map,'mcx_dice.gif','DelayTime',0.5,'LoopCount',inf);

fcw=sum(fcw1+fcw2+fcw3,4);
subplot(223);
hs=slice(log10(abs(double(fcw))),1,1,60);
set(hs,'linestyle','none');
axis equal; colorbar
box on;
title('an arbitrary pattern source from an angle');


%% volumetric source 
Rsrc=20;
[xi,yi,zi]=ndgrid(-Rsrc:Rsrc,-Rsrc:Rsrc,-Rsrc:Rsrc);
sphsrc=((xi.*xi+yi.*yi+zi.*zi)<=Rsrc*Rsrc);

dim=60;

% define source pattern
clear cfg;
% basic settings
cfg.nphoton=1e7;
cfg.tstart=0;
cfg.tend=5e-9;
cfg.tstep=5e-9;
cfg.respin=1;
cfg.seed=99999;
cfg.outputtype = 'energy'; %should get the energy deposits in each voxel

cfg.srctype='pattern3d';
cfg.srcpattern=sphsrc;
cfg.srcparam1=size(cfg.srcpattern);
cfg.srcpos=[10,10,20];
cfg.srcdir=[0 0 1 nan];
cfg.autopilot=1;

% define volume and inclusions
cfg.vol=ones(dim,dim,dim);
%%you can use a JSON string to define cfg.shapes
cfg.shapes='{"Shapes":[{"Sphere":{"O":[25,21,10],"R":10,"Tag":2}}]}';
cfg.prop=[0      0    1    1;    % Boundary
          0.003  0.03 0.8  1;    % Tissue mua = 0.003 mus = 0.03 n = 1.85/1.37
          0.006  0.09 0.8  1;    % Cancer mus  = 1.09 n = 1.37
          1.3    0.8  0.8  1];   % TAM mua = 0.003, n = 0.095
cfg.unitinmm=1e-3;

flux = mcxlab(cfg);
subplot(224);
flux=sum(flux.data,4);
hs=slice(log10(double(flux)),20,40,4);
set(hs,'linestyle','none')
set(gca,'xlim',[0 dim],'ylim',[0 dim],'zlim',[0 dim]);
axis equal;

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
hist(rng,1000);
title('raw RNG distribution');
cfg=rmfield(cfg,'debuglevel');
