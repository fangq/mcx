%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MCXLAB - Monte Carlo eXtreme for MATLAB/Octave by Qianqina Fang
%
% In this example, we test the sub-millimeter voxel feature using 
% the settings in examples/quicktest (i.e. comparing run_qtest.sh and 
% run_grid2x.sh)
%
% This file is part of Monte Carlo eXtreme (MCX) URL:http://mcx.sf.net
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

cfg.nphoton=1e7;
cfg.vol=uint8(ones(60,60,60));
cfg.srcpos=[30 30 1];
cfg.srcdir=[0 0 1];
cfg.gpuid=1;
cfg.autopilot=1;
cfg.prop=[0 0 1 1;0.005 1 0 1.0];
cfg.tstart=0;
cfg.tend=5e-9;
cfg.tstep=5e-9;
% calculate the flux distribution with the given config
[f1,det1]=mcxlab(cfg);

cfg.vol=uint8(ones(120,120,120));
cfg.srcpos=[60 60 1];
cfg.unitinmm=0.5;
cfg.nphoton=8e7;  % you need to simulate 8x photons to get the same noise

[f2,det2]=mcxlab(cfg);

figure;
subplot(121);
imagesc(squeeze(log(f1(1).data(:,30,:,1))));
colorbar;axis equal;axis off;
cl=get(gca,'clim');
subplot(122);
imagesc(squeeze(log(f2(1).data(:,60,:,1))));
colorbar;axis equal;axis off;
set(gca,'clim',cl);
