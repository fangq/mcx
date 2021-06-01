%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MCXLAB - Monte Carlo eXtreme for MATLAB/Octave by Qianqina Fang
%
% In this example, we show two demos for polarized light MC simulation.
% 1. a homogenous slab, infinite along x and y(cyclic boundary condition)
% 2. a three-layer slab, infinite along x and y(cyclic boundary condition)
%
% This file is part of Monte Carlo eXtreme (MCX) URL:http://mcx.sf.net
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear
clc
close all

%% a homogenous slab
cfg1.vol=uint8(ones(20,20,10)); % 3-D label based volume
cfg1.unitinmm=1;                % dx, dy, dz in mm

% mua(in 1/mm),radius(in micron),rho(in 1/micron^3),n_sph,n_med
cfg1.polprop=[
    0, 1.015, 1.152e-4, 1.59, 1.33; % represent label 1
    ];

% optical property of ambient material: mua, mus, g, n
cfg1.prop=[
    0 0 1 1; % represent label 0 (if present in vol)
    ];

% add pencil beam source
cfg1.srcpos=[10 10 0]; % location at center of the bottom
cfg1.srcdir=[0 0 1];   % incident direction cosine
cfg1.issrcfrom0=1;

% GPU settings
cfg1.gpuid=1;
cfg1.autopilot=1;

% time gate settings
cfg1.tstart=0;
cfg1.tend=5e-9;
cfg1.tstep=5e-9;

% light wavelength(in nm)
cfg1.lambda=632.8;

% disable boundary reflection/refraction for now
cfg1.isreflect=0;

% initial stokes vector
cfg1.srciquv=[1.0, 1.0, 0.0, 0.0];

% For each exiting photon, save stokes parameter(i), exiting position(x) and 
% direction(v), number of scattering events(s) and partial path length(p) 
% in each medium, initial photon energy(w).
cfg1.savedetflag='ixvspw';

% cyclic boundary condition towards -x, +x, -y and +y direction
cfg1.bc='______001000'; % for reflectance; use '______000001' instead for transmittance

% photon number
cfg1.nphoton=1e7; 
cfg1.maxdetphoton=1e7; % max number of detected photons

% roulette threshold to match Jessica's setting
cfg1.minenergy=0.01;

% run simulation, output profile of detected photons
[~,detphoton1]=mcxlab(cfg1);

%% compute output Stokes parameters
phi=atan2(detphoton1.v(:,2),detphoton1.v(:,1));
s=detphoton1.s;
s2=[s(:,1),s(:,2).*cos(2.*phi)+s(:,3).*sin(2.*phi),...
   -s(:,2).*sin(2.*phi)+s(:,3).*cos(2.*phi),s(:,4)];

% compute total reflectanced IQUV
mua=cfg1.polprop(:,1);
prop=[0 0 1 1;[mua,zeros(size(mua,1),3)]];
w=mcxdetweight(detphoton1,prop,cfg1.unitinmm); % photon exit weight
R1=s2'*w/cfg1.nphoton;

% output 2D maps (100x100) for I Q U V
ix=discretize(detphoton1.p(:,1),0:0.2:20);
iy=discretize(detphoton1.p(:,2),0:0.2:20);
idx1d=sub2ind([100,100],iy,ix);
HIQUV1=zeros(4,100,100);
for i=1:4
    HIQUV1(i,:,:)=reshape(accumarray(idx1d,s2(:,i)),[100,100]);
end

% plot
figure(1);
subplot(2,2,1);imagesc(squeeze(HIQUV1(1,:,:)));axis equal;title("HI");colorbar;
subplot(2,2,2);imagesc(squeeze(HIQUV1(2,:,:)));axis equal;title("HQ");colorbar;
subplot(2,2,3);imagesc(squeeze(HIQUV1(3,:,:)));axis equal;title("HU");colorbar;
subplot(2,2,4);imagesc(squeeze(HIQUV1(4,:,:)));axis equal;title("HV");colorbar;

%% three-layer hetergeneous slab
cfg2=cfg1;

% add two additional layers (label 2,3)
cfg2.vol(:,:,4:6)=2;
cfg2.vol(:,:,7:10)=3;

% mua(in 1/mm),radius(in micron),rho(in 1/micron^3),n_sph,n_med
cfg2.polprop=[
    0.0, 1.015, 1.152e-4, 1.59, 1.33; % represent label 1
    0.0, 0.500, 1.152e-4, 1.55, 1.33; % represent label 2
    0.0, 1.500, 1.152e-4, 1.50, 1.33; % represent label 3
    ];

% run simulation, output profile of detected photons
[~,detphoton2]=mcxlab(cfg2);

%% compute output Stokes parameters
phi=atan2(detphoton2.v(:,2),detphoton2.v(:,1));
s=detphoton2.s;
s2=[s(:,1),s(:,2).*cos(2.*phi)+s(:,3).*sin(2.*phi),...
   -s(:,2).*sin(2.*phi)+s(:,3).*cos(2.*phi),s(:,4)];

% compute total reflectanced IQUV
mua=cfg2.polprop(:,1);
prop=[0 0 1 1;[mua,zeros(size(mua,1),3)]];
w=mcxdetweight(detphoton2,prop,cfg2.unitinmm); % photon exit weight
R2=s2'*w/cfg2.nphoton;

% output 2D maps (100x100) for I Q U V
ix=discretize(detphoton2.p(:,1),0:0.2:20);
iy=discretize(detphoton2.p(:,2),0:0.2:20);
idx1d=sub2ind([100,100],iy,ix);
HIQUV2=zeros(4,100,100);
for i=1:4
    HIQUV2(i,:,:)=reshape(accumarray(idx1d,s2(:,i)),[100,100]);
end

% plot
figure(2);
subplot(2,2,1);imagesc(squeeze(HIQUV2(1,:,:)));axis equal;title("HI");colorbar;
subplot(2,2,2);imagesc(squeeze(HIQUV2(2,:,:)));axis equal;title("HQ");colorbar;
subplot(2,2,3);imagesc(squeeze(HIQUV2(3,:,:)));axis equal;title("HU");colorbar;
subplot(2,2,4);imagesc(squeeze(HIQUV2(4,:,:)));axis equal;title("HV");colorbar;