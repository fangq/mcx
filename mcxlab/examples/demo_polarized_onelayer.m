clear
clc

%% major input parameters in Jessica's original code
Nphotons  = 1e7;      % photon number

mua       = 0.0;      % background media mua in 1/cm
radius    = 2.03/2;   % sphere radius in micron
lambda    = 0.6328;   % light wavelength in micron
rho       = 1.152e-4; % particle volume density in 1/micron^3
nre_p     = 1.59;     % particle refractive index
nre_med   = 1.33;     % background refractive index

slabsize  = 1;        % slab thickness in cm
hw        = 1;        % slab size along x and y in cm
NN        = 100;      % output image resolution

THRESHOLD = 0.01;     % Roulett threshold

%% create corresponding struct for mcxlab simulation
clear cfg
cfg.nphoton=Nphotons;

% volume discretization resolution: dx,dy,dz(in mm)
cfg.unitinmm=1; 

% create label-based 3D volume
dimxyz=[2*hw,2*hw,slabsize]*10/cfg.unitinmm; 
cfg.vol=uint8(ones(dimxyz)); % dim must be an interger array!

% pencil beam source
cfg.srcpos=[dimxyz(1,1)/2 dimxyz(1,2)/2 0]; % location at center of the bottom
cfg.srcdir=[0 0 1];   % incident direction cosine
cfg.issrcfrom0=1;

% GPU settings
cfg.gpuid=1;
cfg.autopilot=1;

% time gate settings
cfg.tstart=0;
cfg.tend=5e-9;
cfg.tstep=5e-9;

% light wavelength(in nm)
cfg.lambda=lambda*1e3; % convert from micron to nm

% mua(in 1/mm),radius(in micron),rho(in 1/micron^3),n_sph,n_med
cfg.polprop=[
    mua/10., radius, rho, nre_p, nre_med; % represent label 1
    ];

% optical property of ambient material: mua, mus, g, n
cfg.prop=[
    0 0 1 1; % represent label 0 (if present in vol)
    ];

% boundary reflection/refraction
cfg.isreflect=0; % for now disable it

% initial stokes vector
cfg.srciquv=[1.0, 1.0, 0.0, 0.0];

% For each exiting photon, save stokes parameter(i), exiting position(x) and 
% direction(v), number of scattering events(s) and partial path length(p) 
% in each medium, initial photon energy(w).
cfg.savedetflag='ixvspw';

% cyclic boundary condition towards -x, +x, -y and +y direction
cfg.bc='cc_cc_001000'; % use 'cc_cc_000001' for transmittance

% max number of detected photons
cfg.maxdetphoton=1e7;

% roulette threshold to match Jessica's setting
cfg.minenergy=0.01;

% run simulation, output profile of detected photons
[~,detphoton]=mcxlab(cfg);

%% post-processing (will create a function for this later)
U=detphoton.v; % photon exiting direction
S=detphoton.s; % photon exiting Stokes Vector
phi=atan2(U(:,2),U(:,1));

S2=[S(:,1),S(:,2).*cos(2.*phi)+S(:,3).*sin(2.*phi),...
   -S(:,2).*sin(2.*phi)+S(:,3).*cos(2.*phi),S(:,4)];

% total reflectance
w=mcxdetweight(detphoton,detphoton.prop,cfg.unitinmm); % final weight of exiting photons
IQUV=S2'*w/cfg.nphoton; % total reflected I Q U V

% extract I Q U V respectively
I=S2(:,1);
Q=S2(:,2);
U=S2(:,3);
V=S2(:,4);

% extract x and y values of the exiting positions.
x=detphoton.p(:,1);
y=detphoton.p(:,2);

% 2D exiting positions are binned on the 2D plane (z=0)
xedges=0:dimxyz(1,1)/NN:dimxyz(1,1);
yedges=0:dimxyz(1,2)/NN:dimxyz(1,2);
ix=discretize(x,xedges);
iy=discretize(y,yedges);

% create 2D maps to store I Q U V
HI=zeros(NN,NN);
HQ=zeros(NN,NN);
HU=zeros(NN,NN);
HV=zeros(NN,NN);

% accumulate I Q U V
for i=1:length(I)
   HI(iy(i),ix(i))=HI(iy(i),ix(i))+I(i); % x and y is flipped to match Jessica's code
   HQ(iy(i),ix(i))=HQ(iy(i),ix(i))+Q(i);
   HU(iy(i),ix(i))=HU(iy(i),ix(i))+U(i);
   HV(iy(i),ix(i))=HV(iy(i),ix(i))+V(i);
end

% plot
figure;imagesc(HI);axis equal;title("HI");colorbar;
figure;imagesc(HQ);axis equal;title("HQ");colorbar;
figure;imagesc(HU);axis equal;title("HU");colorbar;
figure;imagesc(HV);axis equal;title("HV");colorbar;
