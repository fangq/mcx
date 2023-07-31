%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MCXLAB - Monte Carlo eXtreme for MATLAB/Octave by Qianqina Fang
%
% In this example, we model propagation of polarized light inside a
% heterogeneous media.
%
% This file is part of Monte Carlo eXtreme (MCX) URL:http://mcx.sf.net
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear cfg

%% simulation configurations
% domain bounding box
dim=[100,100,12];
cfg.vol=ones(dim);

% add spherical inclusions
c=[60,60,6]; % center
r=5;         % radius
[yi,xi,zi]=meshgrid(0.5:dim(1,1)-0.5, 0.5:dim(1,2)-0.5, 0.5:dim(1,3)-0.5);
dist_sq=(xi-c(1,1)).^2+(yi-c(1,2)).^2+(zi-c(1,3)).^2;
cfg.vol(dist_sq<r^2)=2;

% dx, dy, dz in mm
cfg.unitinmm=0.1;

% Mie scattering parameters of spherical particles:
% mua[1/mm], radius[micron], rho[1/micron^3], n_sph, n_med
cfg.polprop=[
    0.005, 0.05, 19.1142,    1.59, 1.33; % label 1
    0.005, 1.0,  1.11078e-3, 1.59, 1.33; % label 2
    ];

% optical properties
cfg.prop=[
    0 0 1.0 1.0;
    0 0 1.0 1.0;
    0 0 1.0 1.0];

% planar wide-field source
cfg.srctype='planar';
cfg.srcpos=[0 0 0];
cfg.issrcfrom0=1;
cfg.srcparam1=[dim(1,1) 0 0];
cfg.srcparam2=[0 dim(1,2) 0];

% source direction
cfg.srcdir=[0 0 1];

% incident Stokes vector
cfg.srciquv=[1.0, 1.0, 0.0, 0.0];

% GPU settings
cfg.gpuid=1;
cfg.autopilot=1;

% time gate settings
cfg.tstart=0;
cfg.tend=5e-8;
cfg.tstep=5e-8;

% light wavelength[nm]
cfg.lambda=632.8;

% disable boundary reflection/refraction
cfg.isreflect=0;

% For each detected photon, save stokes parameter[i], exit position[x] and
% direction[v] and initial photon energy[w]
cfg.savedetflag='ixvw';

% cyclic boundary condition to mimic horizontally infinite slabs
cfg.bc='cc_cc_001000';

% photon number
cfg.nphoton=1e7;
cfg.maxdetphoton=1e7; % max number of detected photons

%% run simulation and plot results
[frate,detphoton]=mcxlab(cfg);

% compute backscattered IQUV
phi=atan2(detphoton.v(:,2),detphoton.v(:,1));
s=detphoton.s;
dets=[s(:,1),s(:,2).*cos(2.*phi)+s(:,3).*sin(2.*phi),...
   -s(:,2).*sin(2.*phi)+s(:,3).*cos(2.*phi),s(:,4)];

% weights of exit photons
w=mcxdetweight(detphoton,detphoton.prop,cfg.unitinmm);

% compute total reflectanced IQUV
R=dets'*w/cfg.nphoton;

% plot 2D(40x40) distributions for I Q U V
NN = 40;
ix=discretize(detphoton.p(:,1),0:dim(1,1)/NN:dim(1,1));
iy=discretize(detphoton.p(:,2),0:dim(1,2)/NN:dim(1,1));
HIQUV=zeros(4,NN,NN);

figure(1);
arr=['I','Q','U','V'];
for i=1:4
    HIQUV(i,:,:)=accumarray([iy,ix],dets(:,i),[NN,NN]);
    subplot(2,2,i);
    imagesc(squeeze(HIQUV(i,:,:)));
    axis equal;
    title(arr(1,i));
    colorbar;
end
