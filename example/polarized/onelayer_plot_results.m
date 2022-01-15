%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% matlab script to compute the total reflectance(IQUV) and plot
% 2D distribution of I, Q, U, V. To run this script, one must
% first run onelayer.sh to generate the needed input file.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% load mcx input and output file
data=loadmch("onelayer.mch");
cfg=loadjson("onelayer.json");

%% compute total reflectance
nphotons=cfg.Session.Photons;

% extract simulation settings needed for post-processing
unitinmm=cfg.Domain.LengthUnit; % dx,dy,dz in mm

% default ambient medium
prop=[0 0 1 1];

% absorption coefficent (in 1/mm) of each tissue type
mua=cellfun(@(f)getfield(f,'mua'),cfg.Domain.MieScatter)';
prop=[prop;mua, zeros(size(mua,1),3)];

% photon exiting direction
u=data(:,6:8);
phi=atan2(u(:,2),u(:,1));

% photon exiting Stokes Vector
s=data(:,10:13);
S2=[s(:,1),s(:,2).*cos(2.*phi)+s(:,3).*sin(2.*phi),...
   -s(:,2).*sin(2.*phi)+s(:,3).*cos(2.*phi),s(:,4)];

% final weight of exiting photons
detphoton=struct();
detphoton.ppath=data(:,2);
w=mcxdetweight(detphoton,prop,unitinmm);

% total reflected IQUV
R=S2'*w/nphotons;

%% plot 2-D images for I, Q, U, V
% extract x and y values of the exiting positions.
x=data(:,3);
y=data(:,4);

% define image dimension
NN=100;

% 2D exiting positions are binned on the 2D plane (z=0)
xedges=0:cfg.Domain.Dim(1,1)/NN:cfg.Domain.Dim(1,1);
yedges=0:cfg.Domain.Dim(1,2)/NN:cfg.Domain.Dim(1,2);
ix=discretize(x,xedges);
iy=discretize(y,yedges);
idx1d=sub2ind([NN,NN],iy,ix); % x and y is flipped to match Jessica's code

% accumulate I Q U V
HIQUV=zeros(4,NN,NN);
for i=1:4
    HIQUV(i,:,:)=reshape(accumarray(idx1d,S2(:,i)),[NN,NN]);
end

% plot
subplot(2,2,1);imagesc(squeeze(HIQUV(1,:,:)));axis equal;title("HI");colorbar;
subplot(2,2,2);imagesc(squeeze(HIQUV(2,:,:)));axis equal;title("HQ");colorbar;
subplot(2,2,3);imagesc(squeeze(HIQUV(3,:,:)));axis equal;title("HU");colorbar;
subplot(2,2,4);imagesc(squeeze(HIQUV(4,:,:)));axis equal;title("HV");colorbar;