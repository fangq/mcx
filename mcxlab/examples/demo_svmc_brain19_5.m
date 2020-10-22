%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MCXLAB - Monte Carlo eXtreme for MATLAB/Octave by Qianqina Fang
%
% In this example, we compare SVMC algorithm against conventional VMC in a
% full-head atlas template domain(USC 19.5 year group[Sanchez2012]).
%
% This demo is similar to the MCX simulation used for Fig. 3(c) in [Yan2020]
% and Fig. 9(a) in [TranYan2019].
%
% [Sanchez2012] C.E.Sanchez J.E.Richards and C.R.Almli, “Age-Specific MRI Templates
% for Pediatric Neuroimaging,” Developmental Neuropsychology 37, 379–399 (2012).
%
% [Yan2020] Shijie Yan and Qianqian Fang, "Hybrid mesh and voxel based Monte 
% Carlo algorithm for accurate and efficient photon transport modeling in 
% complex bio-tissues," Biomed. Opt. Express 11, 6262-6270 (2020)
%
% [TranYan2019] Anh Phong Tran, Shijie Yan, Qianqian Fang, "Improving model-based 
% functional near-infrared spectroscopy analysis using mesh-based anatomical and 
% light-transport models," Neurophoton. 7(1) 015008 (22 February 2020)
%
% This file is part of Monte Carlo eXtreme (MCX) URL:http://mcx.sf.net
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc
clear
close all

%% common MC setup
cfg.nphoton=1e8;
cfg.seed=randi([1 2^31-1], 1, 1); %random seed

% inward-pointing pencil beam light source
cfg.srctype='pencil';
cfg.srcpos=[133.5370,90.1988,200.0700];
cfg.srcdir=[-0.5086,-0.1822,-0.8415];
cfg.issrcfrom0=1;

% optical properties
cfg.prop=[0.0   0.0   1.0  1.0;   % background medium
          0.019 7.8   0.89 1.37;  % scalp
          0.019 7.8   0.89 1.37;  % skull
          0.004 0.009 0.89 1.37;  % CSF
          0.02  9.0   0.89 1.37;  % gray matter
          0.08  40.9  0.84 1.37;  % white matter
          0.0   0.0   1.0  1.0];  % air cavity
      
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

%% prepare vmc input volume 
load('fullhead_atlas.mat');
cfg_mcx=cfg;
cfg_mcx.vol=USC_atlas;

%% prepare svmc input volume
addpath('../../utils');
tic;
[svmcvol]=mcxsvmc(USC_atlas,'smoothing',1);
fprintf('SVMC preprocessing complete, ');
toc;

cfg_svmc=cfg;
cfg_svmc.srcpos=cfg_svmc.srcpos+0.5;
cfg_svmc.vol=uint8(svmcvol);

%% run simulations
addpath ../
output_vmc=mcxlab(cfg_mcx);   % conventional vmc
output_svmc=mcxlab(cfg_svmc); % svmc

%% convert time-resolved fluence to CW fluence
phi_vmc=sum(output_vmc.data,4);
phi_svmc=sum(output_svmc.data,4);

%% replace zero with nan
phi_vmc(phi_vmc==0)=nan;
phi_svmc(phi_svmc==0)=nan;

%% compare CW fluence distributions using contour lines
y_plane=90.5; % coronal plane selected for fluence plot
[xx,zz]=meshgrid(0.5:(size(USC_atlas,1)-0.5),0.5:(size(USC_atlas,3)-0.5));

% interpolate SVMC results to slice y=91 (as marching cube mesh has an spatial offset of 0.5)
y=0.5:((size(USC_atlas,2)-0.5));
phi_svmc_y_pivot=permute(phi_svmc,[2,1,3]);
phi_svmc_interp=squeeze(interp1(y,phi_svmc_y_pivot,y_plane+0.5));

% plot CW fluence distribution using contour lines
figure;
clines=-20:0.5:0;
contourf(xx-0.5,zz-0.5,log10(abs(phi_svmc_interp')),clines,'linestyle','-',...
    'color','k','linewidth',2,'DisplayName','SVMC');
hold on;
contour(xx,zz,log10(abs(squeeze(phi_vmc(:,ceil(y_plane),:))')),clines,'linestyle','--',...
    'color','w','linewidth',2,'DisplayName','VMC');
colorbar('EastOutside');

% plot tissue boundaries
contour(squeeze(USC_atlas(:,ceil(y_plane),:))','linestyle','--',...
    'color',[0.5 0.5 0.5],'linewidth',1.5,'HandleVisibility','off');

axis equal;
lg=legend;
set(lg,'color','[0.5 0.5 0.5]');
set(lg,'box','on');

set(gca,'ylim', [160 220]);ylabel('z(mm)');
set(gca,'xlim', [45 165]);xlabel('x(mm)');
set(gca,'clim',[-12 0]);
set(gca,'fontsize',18);
