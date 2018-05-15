%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MCXLAB - Monte Carlo eXtreme for MATLAB/Octave by Qianqina Fang
%
% In this example, we show how to use replay to obtain time-resolved
% Jacobians - setting cfg.replaydet to -1 to replay all detectors
%
% This file is part of Monte Carlo eXtreme (MCX) URL:http://mcx.sf.net
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear cfg cfgs
cfg.nphoton=1e8;
cfg.vol=uint8(ones(60,60,20));
cfg.srcpos=[30 30 0];
cfg.srcdir=[0 0 1];
cfg.gpuid=1;
% cfg.gpuid='11'; % use two GPUs together
cfg.autopilot=1;
cfg.issrcfrom0=1;
cfg.prop=[0 0 1 1;0.005 1 0 1.37];
cfg.tstart=0;
cfg.tend=5e-9;
cfg.tstep=2e-10;
% calculate the flux distribution with the given config
cfg.detpos=[30 30 20 2; 30 40 20 2; 30 50 20 2];
[flux, detp, vol, seeds]=mcxlab(cfg);

%cfg.replaydet=0;  % replay all det and sum all
%cfg.replaydet=2;  % replay only the 2nd detector
%cfg.replaydet=3;  % replay only the 3rd detector
cfg.replaydet=-1; % replay all det and save all

newcfg=cfg;
newcfg.seed=seeds.data;
newcfg.outputtype='jacobian';
newcfg.detphotons=detp.data;
[flux2, detp2, vol2, seeds2]=mcxlab(newcfg);
for i=1:size(flux2.data,4)
    imagesc(log10(abs(squeeze(flux2.data(30,:,:,i)))))
    title(sprintf('%d',i));
    waitforbuttonpress;
end
jac=sum(flux2.data,4);
for i=1:size(jac,ndims(jac))
    subplot(1,size(jac,ndims(jac)),i);
    imagesc(log10(abs(squeeze(jac(30,:,:,i)))))
end
