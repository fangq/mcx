%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MCXLAB - Monte Carlo eXtreme for MATLAB/Octave by Qianqina Fang
%
% In this example, we compare SVMC algorithm against conventional VMC in a
% heterogeneous domain. (see Fig. 3(b) of [Yan2020])
%
% [Yan2020] Shijie Yan and Qianqian Fang, "Hybrid mesh and voxel based Monte 
% Carlo algorithm for accurate and efficient photon transport modeling in 
% complex bio-tissues," Biomed. Opt. Express 11, 6262-6270 (2020)
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

% optical properties
cfg.prop=[0.0   0.0 1.0  1.0;   % background (air, void)
          0.005 1.0 0.89 1.37]; % spherical inclusion
      
% time-domain simulation parameters
cfg.tstart=0;
cfg.tend=5e-9;
cfg.tstep=5e-10;

% enable boundary reflection/refraction
cfg.isreflect=1; % turn off boundary reflection to see matched results between VMC and SVMC

% spatial resolution
cfg.unitinmm=1;

% output fluence
cfg.outputtype='fluence';

% GPU settings
cfg.gpuid=1;
cfg.autopilot=1;

%% prepare mcx input volume
dim=60;
[xi,yi,zi]=ndgrid(0.25:0.5:(dim-0.25),0.25:0.5:(dim-0.25),0.25:0.5:(dim-0.25));
dist=(xi-30.5).^2+(yi-30.5).^2+(zi-30.5).^2;
mcxvol=zeros(size(xi));
mcxvol(dist<625)=1;

cfg_mcx=cfg;
cfg_mcx.vol=uint8(mcxvol);

% finer spatial resolution
cfg_mcx.unitinmm=0.5;
cfg_mcx.srcpos=cfg_mcx.srcpos/cfg_mcx.unitinmm;

%% prepare svmc input volume
tic;
[xi,yi,zi]=ndgrid(1:dim,1:dim,1:dim);
dist=(xi-30.5).^2+(yi-30.5).^2+(zi-30.5).^2;
vol2=zeros(size(xi));
vol2(dist<625)=1;

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
[yy_svmc,zz_svmc]=ndgrid(0.5:59.5,0.5:59.5);
[yy_vmc,zz_vmc]=ndgrid(0.25:0.5:59.75,0.25:0.5:59.75);

figure;
clines=-10:0.5:10;
contourf(yy_svmc,zz_svmc,log10(abs(squeeze(phi_svmc(31,:,:)))),clines,'linestyle','-',...
    'color','k','linewidth',2,'DisplayName','SVMC');
hold on;
contour(yy_vmc,zz_vmc,log10(abs(squeeze(phi_vmc(61,:,:)))),clines,'linestyle','--',...
    'color','w','linewidth',2,'DisplayName','VMC');
colorbar('EastOutside');

% plot media boundaries
[xcirc,ycirc] = cylinder(25,200);
xcirc=xcirc(1,:)+30.5;
ycirc=ycirc(1,:)+30.5;
plot(xcirc,ycirc,'--','linewidth',1.5,'color',[.5 .5 .5],'HandleVisibility','off');

% other plot settings
axis equal;
ylabel('z(mm)');
xlabel('y(mm)');
lg=legend;
set(lg,'Color',[0.5 0.5 0.5]);
set(lg,'box','on');
set(gca,'fontsize',15);