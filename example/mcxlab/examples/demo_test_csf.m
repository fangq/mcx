%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Error evaluation when assuming musp(CSF) with 1/average(thickness)
%    from Custo Applied Optics 2006
%
% This file is part of Monte Carlo eXtreme (MCX) URL:http://mcx.sf.net
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear cfg;

%% preparing the input data
% set seed to make the simulation repeatible
cfg.seed=hex2dec('623F9A9E'); 

cfg.nphoton=2e8;

% define a 4 layer structure
cfg.vol=ones(100,100,50);
cfg.vol(:,:,11:16)=2;
cfg.vol(:,:,17:18)=3;
cfg.vol(:,:,19:end)=4;
cfg.vol=uint8(cfg.vol);

% define the source position
cfg.srcpos=[50,50,0]+1;
cfg.srcdir=[0 0 1];

% use the brain optical properties defined at
% http://mcx.sourceforge.net/cgi-bin/index.cgi?MMC/CollinsAtlasMesh
% format: [mua(1/mm) mus(1/mm) g n]

cfg.prop=[0 0 1 1            % medium 0: the environment
   0.019 7.8   0.89 1.37     % medium 1: scalp
   0.019 7.8   0.89 1.37     % medium 2: skull
   0.004 0.009 0.89 1.37     % medium 3: CSF
   0.02  9.0   0.89 1.37     % medium 4: gray matter
   0.08 40.9   0.84 1.37];   % medium 5: white matter (not used)

% time-domain simulation parameters
cfg.tstart=0;
cfg.tend=5e-9;
cfg.tstep=5e-9;

cfg.gscatter=50; % make scattering faster, with some approximations

% GPU thread configuration
cfg.autopilot=1;
cfg.gpuid=1;

cfg.isreflect=1; % enable reflection at exterior boundary

% use literature CSF optical properties
tic;
f1=mcxlab(cfg);
toc;

% use approximated CSF mus', from Custo Applied Optics 2006
%cfg.prop(4,:)=cfg.prop(3,:); % make csf the same as the scalp
cfg.prop(4,2)=1/2; 
cfg.prop(4,3)=0;
tic;
f2=mcxlab(cfg);
toc;

% 
% cfg.prop(4,:)=cfg.prop(3,:); % make csf the same as the scalp
% tic;
% f3=mcxlab(cfg);
% toc;

dd=f1.data./f2.data;

figure

subplot(131)
imagesc(log10(abs(squeeze(f1.data(:,51,:)))));
axis equal;
colorbar;
title('with csf');
hold on;
plot([10 10],[1 101],'--',[16 16],[1 101],'--',[18 18],[1 101],'--');


subplot(132)
imagesc(log10(abs(squeeze(f2.data(:,51,:)))));
axis equal;
colorbar;
title('csf mus''=1/thickness');
hold on;
plot([10 10],[1 101],'--',[16 16],[1 101],'--',[18 18],[1 101],'--');

subplot(133)
imagesc(log10(abs(squeeze(dd(:,51,:)))));
axis equal;
colorbar;
title('difference log10(with/without)');
hold on;
plot([10 10],[1 101],'--',[16 16],[1 101],'--',[18 18],[1 101],'--');