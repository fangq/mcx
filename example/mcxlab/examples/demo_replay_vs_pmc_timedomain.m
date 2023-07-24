%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MCXLAB - Monte Carlo eXtreme for MATLAB/Octave by Qianqina Fang
%
% In this example, we compare perturbation MC and replay in predicting
% time-resolved measurement change with respect to mua change in a layer
%
% This file is part of Monte Carlo eXtreme (MCX) URL:http://mcx.sf.net
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear cfg
cfg.nphoton=2e8;
cfg.vol=uint8(ones(60,60,30));
cfg.vol(:,:,15:end)=2;
cfg.srcpos=[30 30 0];
cfg.srcdir=[0 0 1];
cfg.gpuid=1;
% cfg.gpuid='11'; % use two GPUs together
cfg.autopilot=1;
cfg.issrcfrom0=1;
cfg.prop=[0 0 1 1;0.005 1 0 1.37; 0.01 1 0 1.37];
cfg.tstart=0;
cfg.tend=5e-9;
cfg.tstep=2e-10;
cfg.detpos=[15 30 0 2];
[flux, detp, vol, seeds]=mcxlab(cfg);

%% replay to get time-domain J_mua

newcfg=cfg;
newcfg.seed=seeds.data;
newcfg.outputtype='jacobian';
newcfg.detphotons=detp.data;
[flux2, detp2, vol2, seeds2]=mcxlab(newcfg);
jac=flux2.data;

%% predict time-domain measurement change using pMC and partial-path

dmua=0.0001;
w1=mcxdetweight(detp,cfg.prop);
dprop=cfg.prop;
dprop(3,1)=dprop(3,1)+dmua;
w2=mcxdetweight(detp,dprop);
dw=w2-w1;
tof=mcxdettime(detp,cfg.prop);
[counts, idx]=histc(tof,0:cfg.tstep:cfg.tend);

%% predict time-domain measurement change using TD Jacobian and compare

dphi=zeros(size(jac,4),1);  % change of measurements using replay Jacobian
dphi2=zeros(size(jac,4),1); % change of measurements from pMC

for i=1:size(jac,4)
   dmeas=jac(:,:,15:end,i)*dmua;
   dphi(i)=-sum(dmeas(:));
   dphi2(i)=sum(dw(idx==i))/sum(w1);
end

plot(1:size(jac,4), dphi, 'r-+',1:size(jac,4), dphi2, 'g-+');
legend('use replay Jacobian','use pMC')
xlabel('time gates')
ylabel('measurement change')