%==========================================================================
% Script to verify Lambertian photon exiting angular profile
%
% Author: Qianqian Fang <q.fang at neu.edu>
% Initial version: May 5, 2018
%==========================================================================

clear cfg
cfg.nphoton=1e8;
cfg.vol=uint8(ones(60,60,60));
cfg.srcpos=[20, 30, 0];
cfg.srcdir=[0 0 1];

cfg.maxdetphoton=1500000;
cfg.gpuid=1;
cfg.autopilot=1;

cfg.prop=[0 0 1 1;0.005 1 0 1.37];

cfg.tstart=0;
cfg.tend=5e-9;
cfg.tstep=5e-9;

cfg.issrcfrom0=1;
cfg.isreflect=1;
cfg.issaveexit=1;

cfg.detpos = [25 30 0 2];

%cfg.prop=[0 0 1 1;0.005 0.01 0 1.37]; %try this low-scattering case

% calculate the flux distribution with the given config
[flux1,detp1] = mcxlab(cfg);

% it took me a while to figure out, but the key is to divide the area :)

el=asin(detp1.v(:,3));  % elevation angle of v, el=0 parallel to surface, el=pi/2 at normal dir

edges=linspace(min(el),max(el),100);  % angle bins for hisotogram

[ct, bin]=histc(el,edges); % count of photons per angle bin

R=cfg.detpos(1,4);         % radius of the det
hedges=abs(R*sin(edges));  % height of each spherical segment for each bin
zonearea=2*pi*R*(diff(hedges)); % area of each spherical segment

detweight=mmcdetweight(detp1, cfg.prop); % get detected photon weight
angularweight=accumarray(bin,detweight); % sum total weight per zone

angularflux=angularweight(1:end-1)'./zonearea;  % calculate flux per angle

polar((edges(1:end-1)+edges(2:end))*0.5, angularflux); % plot flux vs angle
hold on;
polar(pi-(edges(1:end-1)+edges(2:end))*0.5, angularflux); % mirror to form a circle