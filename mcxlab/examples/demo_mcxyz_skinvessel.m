%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  mcxyz skinvessel benchmark
%
%  must change mcxyz maketissue.m boundaryflag variable from 2 to 1 to get
%  comparable absorption fraction (40%), otherwise, mcxyz obtains slightly
%  higher absorption (~42%) with boundaryflag=2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear cfg flux

load mcxyz_skinvessel.mat

cfg.vol=zeros(200,200,200);
cfg.shapes=['{"Shapes":[{"ZLayers":[[1,20,1],[21,32,4],[33,200,3]]},' ...
    '{"Cylinder": {"Tag":2, "C0": [0,100.5,100.5], "C1": [200,100.5,100.5], "R": 20}}]}'];

cfg.nphoton=1e7;
cfg.issrcfrom0=1;
cfg.srcpos=[99.5 99.5 20];
cfg.tstart=0;
cfg.tend=5e-8;
cfg.tstep=5e-8;
cfg.srcdir=[0 0 1];
cfg.srctype='disk';
cfg.srcparam1=[0.3/cfg.unitinmm 0 0 0];
cfg.isreflect=1;
cfg.autopilot=1;
cfg.gpuid=1;

cfg.outputtype='energy';
%cfg.outputtype='flux';
flux=mcxlab(cfg);

% convert mcx solution to mcxyz's output
% cfg.tstep is used in mcx's normalization, must undo
% 100 converts 1/mm^2 from mcx output to 1/cm^2 as in mcxyz
mcxdata=flux.data/cfg.tstep*100;

figure;
dim=size(cfg.vol);
yi=((1:dim(2))-floor(dim(2)/2))*cfg.unitinmm;
zi=(1:dim(3))*cfg.unitinmm;

imagesc(yi,zi,log10(abs(squeeze(mcxdata(100,:,:))))')
axis equal;
colorbar
