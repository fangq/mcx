%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MCXLAB - Monte Carlo eXtreme for MATLAB/Octave by Qianqina Fang
%
% Speed benchmarks for MCXLAB (equivalent to the examples in mcx/examples/benchmark)
%
% This file is part of Monte Carlo eXtreme (MCX) URL:http://mcx.sf.net
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear cfg

gpuinfo=mcxlab('gpuinfo');

cfg.gpuid=1;  % gpuid can be an integer 1-N to specify the n-th GPU
              % or it can be a string containing only '0's and '1's. 
	      % exp: if gpuid='1010', it means the 1st and 3rd GPU are both used.

%cfg.gpuid='111';          % on wazu, GPU#1,2,3 are 980Ti, 590 Core1 and Core 2
%cfg.workload=[90,10,10];  % workload distribution between the 3 GPUs

gpuid=cfg.gpuid;

if(ischar(gpuid) && regexp(gpuid,'^[01]+$'))
    gpuid=char(gpuid)-'0';
    lastdev=find(gpuid,1,'last');
    if(lastdev>length(gpuinfo))
        error('you specified a non-existant GPU');
    end
    gpuid=gpuid(1:lastdev);
elseif(isnumeric(gpuid) && gpuid>0)
    gpuid=[zeros(1,gpuid-1) 1];
else
    error('the specified GPU id is invalid');
end

dates=datestr(now,'yyyy-mm-dd HH:MM:SS');
mcxbenchmark=struct('date',dates,'gpu',gpuinfo(gpuid==1));

try
    hbar=waitbar(0,'Running benchmarks #1');
catch
    hbar=[];
end
count=0;

cfg.nphoton=1e8;
cfg.vol=uint8(ones(60,60,60));
cfg.srcpos=[29 29 0];
cfg.srcdir=[0 0 1];
cfg.issrcfrom0=1;
cfg.autopilot=1;
cfg.prop=[0 0 1 1;0.005 1 0.01 1.37;0.002 5.0 0.9, 1.0];
cfg.tstart=0;
cfg.tend=5e-9;
cfg.tstep=5e-9;
cfg.seed=29012392;
cfg.detpos=[29 19 0 1;29 39 0 1;19 29 0 1;39 29 0 1];
cfg.isreflect=0;
% calculate the flux distribution with the given config

speed=zeros(3);

for i=1:size(speed,1)
    [flux, detps]=mcxlab(cfg);
    if(abs(flux.stat.energyabs/flux.stat.energytot-0.1769)>0.005 || ...
       abs(flux.stat.energytot-flux.stat.nphoton)>10)
        flux.stat
        error('output absorption fraction is incorrect');
    end
    count=count+1;
    if(~isempty(hbar))
        waitbar(count/9,hbar,'Running benchmarks #1');
    end
    speed(i,1)=flux.stat.nphoton/flux.stat.runtime;
end
mcxbenchmark.benchmark1.stat=flux.stat;

if(~isempty(hbar))
    waitbar(count/9,hbar,'Running benchmarks #2');
end

cfg.shapes='{"Shapes":[{"Sphere":{"Tag":2, "O":[30,30,30],"R":15}}]}';
cfg.isreflect=1;

for i=1:size(speed,1)
    [flux, detps]=mcxlab(cfg);
    if(abs(flux.stat.energyabs/flux.stat.energytot-0.2701)>0.005|| ...
       abs(flux.stat.energytot-flux.stat.nphoton)>10)
        flux.stat
        error('output absorption fraction is incorrect');
    end
    count=count+1;
    if(~isempty(hbar))
        waitbar(count/9,hbar,'Running benchmarks #2');
    end
    speed(i,2)=flux.stat.nphoton/flux.stat.runtime;
end
mcxbenchmark.benchmark2.stat=flux.stat;

if(~isempty(hbar))
    waitbar(count/9,hbar,'Running benchmarks #3');
end

cfg=rmfield(cfg,'shapes');
cfg.srctype='planar';
cfg.srcparam1=[40 0 0 0];
cfg.srcparam2=[0 40 0 0];
cfg.srcpos=[10 10 -10];

for i=1:size(speed,1)
    [flux, detps]=mcxlab(cfg);
    if(abs(flux.stat.energyabs/flux.stat.energytot-0.2551)>0.005)
        flux.stat
        error('output absorption fraction is incorrect');
    end
    count=count+1;
    if(~isempty(hbar))
        waitbar(count/9,hbar,'Running benchmarks #3');
    end
    speed(i,3)=flux.stat.nphoton/flux.stat.runtime;
end
mcxbenchmark.benchmark3.stat=flux.stat;

speed=mean(speed);

for i=1:length(speed)
   mcxbenchmark.(sprintf('benchmark%d',i)).speed=speed(i);
end

mcxbenchmark.speedsum=sum(speed);

if(~isempty(hbar))
    delete(hbar);
end
