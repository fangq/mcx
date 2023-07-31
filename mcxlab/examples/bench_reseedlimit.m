%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MCXLAB - Monte Carlo eXtreme for MATLAB/Octave by Qianqina Fang
%
% In this example, we show the most basic usage of MCXLAB.
%
% This file is part of Monte Carlo eXtreme (MCX) URL:http://mcx.sf.net
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

counts=[1e5 sqrt(10)*1e5 1e6 sqrt(10)*1e6 1e7 sqrt(10)*1e7 1e8];
nrepeat=10;

cfg.vol=uint8(ones(60,60,60));
cfg.srcpos=[30 30 1];
cfg.srcdir=[0 0 1];
cfg.gpuid=1;
cfg.autopilot=1;
cfg.prop=[0 0 1 1;0.005 1 0 1.37];
cfg.tstart=0;
cfg.tend=5e-9;
cfg.tstep=5e-9;
cfg.reseedlimit=1e4;

% calculate the flux distribution with the given config

data=zeros(length(counts),nrepeat,numel(cfg.vol));
for i=1:length(counts)
   cfg.nphoton=counts(i);
   for j=1:nrepeat
      flux=mcxlab(cfg);
      cfg.seed=hex2dec('623F9A9E')+i*nrepeat+j;
      data(i,j,:)=flux.data(:);
   end
end

datastd=zeros(size(data,1),size(data,3));
for i=1:size(data,1)
    datastd(i,:)=std(squeeze(data(i,:,:)),0,1);
end

idx1=sub2ind(size(cfg.vol),30,30,10);
idx2=sub2ind(size(cfg.vol),30,30,30);
idx3=sub2ind(size(cfg.vol),30,30,50);

figure;
loglog(counts, datastd(:,[idx1 idx2 idx3]),'-o');
legend('z=10','z=30','z=50');