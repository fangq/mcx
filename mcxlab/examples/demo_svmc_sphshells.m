%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MCXLAB - Monte Carlo eXtreme for MATLAB/Octave by Qianqina Fang
%
% In this example, we compare SVMC algorithm against conventional VMC in a
% heterogeneous domain. (see Fig. 3(a) of [Yan2020] and Fig. 1(e) of [Yan2019])
%
% [Yan2020] Shijie Yan and Qianqian Fang, "Hybrid mesh and voxel based Monte 
% Carlo algorithm for accurate and efficient photon transport modeling in 
% complex bio-tissues," Biomed. Opt. Express 11, 6262-6270 (2020)
%
% [Yan2019] Yan S, Tran AP, Fang Q. Dual-grid mesh-based Monte Carlo algorithm 
% for efficient photon transport simulations in complex three-dimensional 
% media. J Biomed Opt. 2019 Feb;24(2):1-4.
%
% This file is part of Monte Carlo eXtreme (MCX) URL:http://mcx.sf.net
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc
clear
close all

%% common MC setup
cfg.nphoton=1e8;
cfg.seed=randi([1 2^31-1], 1, 1); %random seed

% pencil beam light source
cfg.srcpos=[30.5 30.5 0];
cfg.srcdir=[0 0 1];
cfg.issrcfrom0=1;

% optical properties (tissue-like multi-layered media)
cfg.prop=[0.0   0.0   1.0  1.0;   % background (air, void)
          0.02  7.0   0.89 1.37;  % scalp/skull
          0.004 0.009 0.89 1.37;  % CSF
          0.02  9.0   0.89 1.37;  % gray matter
          0.05  0.0   1.0  1.37]; % non-scattering inclusion
      
% time-domain simulation parameters
cfg.tstart=0;
cfg.tend=5e-9;
cfg.tstep=5e-10;

% enable boundary reflection/refraction
cfg.isreflect=1;

% spatial resolution
cfg.unitinmm=1;

% output fluence
cfg.outputtype='fluence';

% GPU settings
cfg.gpuid=1;
cfg.autopilot=1;

%% prepare mcx input volume
dim=60;
[xi,yi,zi]=ndgrid(0.5:(dim-0.5),0.5:(dim-0.5),0.5:(dim-0.5));
dist=(xi-30.5).^2+(yi-30.5).^2+(zi-30.5).^2;
mcxvol=ones(size(xi));
mcxvol(dist<625)=2;
mcxvol(dist<529)=3;
mcxvol(dist<100)=4;

cfg_mcx=cfg;
cfg_mcx.vol=uint8(mcxvol);

%% prepare svmc input volume
tic;
[xi,yi,zi]=ndgrid(1:dim,1:dim,1:dim);
dist=(xi-30.5).^2+(yi-30.5).^2+(zi-30.5).^2;
vol2=ones(size(xi));
vol2(dist<625)=2;
vol2(dist<529)=3;
vol2(dist<100)=4;

addpath('../../utils');
svmcvol=mcxsvmc(vol2,'smoothing',1); % 1: enable gaussian smoothing 0: otherwise
fprintf('SVMC preprocessing complete, ');
toc;

cfg_svmc=cfg;
cfg_svmc.vol=uint8(svmcvol);

%% run simulations
addpath ../
output_vmc=mcxlab(cfg_mcx);   % conventional vmc
output_svmc=mcxlab(cfg_svmc); % svmc

%% convert time-resolved fluence to CW fluence
phi_vmc=sum(output_vmc.data,4);
phi_svmc=sum(output_svmc.data,4);

%% compare CW fluence distributions using contour lines
figure;
clines=-10:0.5:10;
contourf(log10(abs(squeeze(phi_svmc(31,:,:))')),clines,'linestyle','-',...
    'color','k','linewidth',2,'DisplayName','SVMC');
hold on;
contour(log10(abs(squeeze(phi_vmc(31,:,:))')),clines,'linestyle','--',...
    'color','w','linewidth',2,'DisplayName','VMC');
colorbar('EastOutside');

% plot media boundaries
[xcirc,ycirc] = cylinder(10,200);
xcirc=xcirc(1,:)+31;
ycirc=ycirc(1,:)+31;
plot(xcirc,ycirc,'--','linewidth',1.5,'color',[.5 .5 .5],'HandleVisibility','off');

[xcirc,ycirc] = cylinder(23,200);
xcirc=xcirc(1,:)+31;
ycirc=ycirc(1,:)+31;
plot(xcirc,ycirc,'--','linewidth',1.5,'color',[.5 .5 .5],'HandleVisibility','off');

[xcirc,ycirc] = cylinder(25,200);
xcirc=xcirc(1,:)+31;
ycirc=ycirc(1,:)+31;
plot(xcirc,ycirc,'--','linewidth',1.5,'color',[.5 .5 .5],'HandleVisibility','off');

% other plot settings
axis equal;
ylabel('z(mm)');
xlabel('y(mm)');
lg=legend;
set(lg,'Color',[0.5 0.5 0.5]);
set(lg,'box','on');
set(gca,'fontsize',15);