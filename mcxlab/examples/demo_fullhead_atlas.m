%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MCXLAB - Monte Carlo eXtreme for MATLAB/Octave by Qianqina Fang
%
% In this example, we demonstrate light transport simulation in a full-head 
% atlas template(USC 19.5 year group[Sanchez2012]). 
%
% This demo is identical to the MCX simulation used for Fig.9(a) in
% TranYan2019(submitted).
%
% [Sanchez2012] C.E.Sanchez J.E.Richards and C.R.Almli, “Age-Specific MRI Templates
% for Pediatric Neuroimaging,” Developmental Neuropsychology 37, 379–399 (2012).
%
% [TranYan2019] A.P.Tran, S.Yan and Q.Fang, "Improving model-based fNIRS 
% analysis using mesh-based anatomical and light-transport models".
%
% This file is part of Monte Carlo eXtreme (MCX) URL:http://mcx.sf.net
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc
clear
load('fullhead_atlas.mat');
%% prepare cfg for MCX simulation
clear cfg
cfg.nphoton=1e8;
cfg.outputtype='fluence';

% tissue labels:0-ambient air,1-scalp,2-skull,3-csf,4-gray matter,5-white matter,6-air cavities
cfg.vol=USC_atlas;
cfg.prop=[0,0,1,1;0.019 7.8 0.89 1.37;0.02 9.0 0.89 1.37;0.004 0.009 0.89 1.37;0.019 7.8 0.89 1.37;0.08 40.9 0.84 1.37;0,0,1,1];

% light source
cfg.srcnum=1;
cfg.srcpos=[133.5370,90.1988,200.0700]; %pencil beam source placed at EEG 10-5 landmark:"C4h"
cfg.srctype='pencil';
cfg.srcdir=[-0.5086,-0.1822,-0.8415]; %inward-pointing source
cfg.issrcfrom0=1;

% time windows
cfg.tstart=0;
cfg.tend=5e-9;
cfg.tstep=5e-10;

% other simulation parameters
cfg.isspecular=0;
cfg.isreflect=1;
cfg.autopilot=1;
cfg.gpuid=1;

%% run MCX simulation
[flux]=mcxlab(cfg);

%% post-simulation data processing and visualization
% convert time-resolved fluence to CW fluence
CWfluence=sum(flux.data,4); 

% coronal plane selected for fluence plot
y_plane=90.5;   
[xx,zz]=meshgrid(1:size(cfg.vol,1),1:size(cfg.vol,3));

% plot CW fluence distribution using contour lines
figure;
clines=-20:0.5:0;
contourf(xx,zz,log10(abs(squeeze(CWfluence(:,ceil(y_plane),:))')),clines,'linestyle','--','color',[0.9100    0.4100    0.1700],'linewidth',1.5,'DisplayName','MCX');
hold on;axis equal;
colorbar('EastOutside');

% plot tissue boundary contour, source, legend, etc.
contour(squeeze(cfg.vol(:,ceil(y_plane),:))','k--','linewidth',1.25,'HandleVisibility','off');
plot(cfg.srcpos(1,1),cfg.srcpos(1,3),'o','MarkerEdgeColor','r','MarkerFaceColor','r','MarkerSize',10,'DisplayName','source');
lg=legend('Location','northeast');
set(lg,'color','[0.85 0.85 0.85]');
set(lg,'box','on');
set(gca,'ylim', [160 225]);ylabel('z [mm]');
set(gca,'xlim', [10 165]);xlabel('x [mm]');
set(gca,'clim',[-12 0]);
set(gca,'fontsize',18);
set(gca, 'FontName', 'Times New Roman');
