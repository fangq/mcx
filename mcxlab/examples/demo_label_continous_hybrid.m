clear
clc

%% case 1: homogenous media, update the refractive index from n_old to n_new for all voxels
clear cfg1
n_old=1.37; 
n_new=1.55;

cfg1.nphoton=1e8;
cfg1.srcpos=[30 30 1];
cfg1.srcdir=[0 0 1];
cfg1.gpuid=1;
cfg1.autopilot=1;
cfg1.tstart=0;
cfg1.tend=5e-9;
cfg1.tstep=5e-9;
cfg1.isreflect=1;

% conventional way to define optical properties: label each voxel with
% tissue types
cfg1.vol=uint8(ones(60,60,60));
cfg1.prop=[0 0 1 1;0.005 1 0 n_new];
flux1_conventional=mcxlab(cfg1);

% tissue-type based continuous varying media
cfg1.prop=[0 0 1 1;0.005 1 0 n_old];                 % n=1.37 will be updated during simulation
property_substitute=single(4*ones(size(cfg1.vol)));  % 0-no update, 1-update mua, 2-update mus, 3-update g, 4-update n
label=single(ones(size(cfg1.vol)));                  % label-based tissue type 0-255
replace_property=single(n_new*ones(size(cfg1.vol))); % values to substitute corresponding optical properties
cfg1.vol=permute(cat(4,replace_property,label,property_substitute),[4 1 2 3]); % Concatenated as a 4-D single array 
flux1_new=mcxlab(cfg1);

% compare fluence maps between two approaches
figure(1);
clines=-7:1:-1;
contourf(log10(squeeze(flux1_conventional.data(31,:,:))*cfg1.tstep),clines,'r-');axis equal;
hold on;
contour(log10(squeeze(flux1_new.data(31,:,:))*cfg1.tstep),clines,'w--');
legend('conventional','new');
colorbar;
title('homogeneous cube');

%% case 2: Two-layer slab: The refractive indices of tissue type 1 and 2 will be updated to 1.45 and 1.6 respectively
clear cfg2
n_old_1=1.37; 
n_new_1=1.45;
    
n_old=1.37; 
n_new_2=1.6;

cfg2.nphoton=1e8;
cfg2.srcpos=[30 30 1];
cfg2.srcdir=[0 0 1];
cfg2.gpuid=1;
cfg2.autopilot=1;
cfg2.tstart=0;
cfg2.tend=5e-9;
cfg2.tstep=5e-9;
cfg2.isreflect=1;

% conventional way to define optical properties: label each voxel with
% tissue types
cfg2.vol=uint8(ones(60,60,60));
cfg2.vol(:,:,31:60)=2;
cfg2.prop=[0 0 1 1;0.005 1 0 n_new_1;0.01 2 0 n_new_2];
flux2_conventional=mcxlab(cfg2);

% tissue-type based continuous varying media
cfg2.prop=[0 0 1 1;0.005 1 0 n_old_1;0.01 2 0 n_old];  % n=1.37 will be updated during the simulation
property_substitute=single(4*ones(size(cfg2.vol)));    % 0-no update, 1-update mua, 2-update mus, 3-update g, 4-update n
label=single(ones(size(cfg2.vol)));                    % label-based tissue type 0-255
label(:,:,31:60)=2;
replace_property=single(n_new_1*ones(size(cfg2.vol))); % values to substitute corresponding optical properties
replace_property(:,:,31:60)=n_new_2;
cfg2.vol=permute(cat(4,replace_property,label,property_substitute),[4 1 2 3]); % Concatenated as a 4-D array 
flux2_new=mcxlab(cfg2);

clines=[-10:0.5:-2];
% compare
figure(2);
contourf(log10(squeeze(flux2_conventional.data(31,:,:))*cfg2.tstep),clines,'r-');axis equal;
hold on;
contour(log10(squeeze(flux2_new.data(31,:,:))*cfg2.tstep),clines,'w--');
legend('conventional','new');
title('two layer slab');

%% case 3: multi-layer slab: mua of tissue type 1 will be updated to 0.05 while the refractive indices of tissue type 2 will be updated to [1.30,1.35,1.40,1.45,1.50,1.55]
clear cfg3
mua_old_1=0.005; 
mua_new_1=0.05;
    
n_old=1.37;
n_new_1=1.30;
n_new_2=1.35;
n_new_3=1.40;
n_new_4=1.45;
n_new_5=1.50;
n_new_6=1.55;

cfg3.nphoton=1e8;
cfg3.srcpos=[30 30 1];
cfg3.srcdir=[0 0 1];
cfg3.gpuid=1;
cfg3.autopilot=1;
cfg3.tstart=0;
cfg3.tend=5e-9;
cfg3.tstep=5e-9;
cfg3.isreflect=1;

% conventional way to define optical properties: label each voxel with
% tissue types
cfg3.vol=uint8(ones(60,60,60));
for i=1:6     % each layer
    cfg3.vol(:,:,30+i*5)=i+1;
    cfg3.vol(:,:,30+i*5-1)=i+1;
    cfg3.vol(:,:,30+i*5-2)=i+1;
    cfg3.vol(:,:,30+i*5-3)=i+1;
    cfg3.vol(:,:,30+i*5-4)=i+1;
end
cfg3.prop=[0 0 1 1;
           mua_new_1 1 0 1.37;
           0.01 2 0 n_new_1;
           0.01 2 0 n_new_2;
           0.01 2 0 n_new_3;
           0.01 2 0 n_new_4;
           0.01 2 0 n_new_5;
           0.01 2 0 n_new_6];
flux2_conventional=mcxlab(cfg3);

% tissue-type based continuous varying media
cfg3.prop=[0 0 1 1;0.005 1 0 1.37;0.01 2 0 n_old];     % n=1.37 will be updated during the simulation
property_substitute=single(ones(size(cfg3.vol)));      % 0-no update, 1-update mua, 2-update mus, 3-update g, 4-update n
property_substitute(:,:,31:60)=4;
label=single(ones(size(cfg3.vol)));                    % label-based tissue type 0-255
label(:,:,31:60)=2;
replace_property=single(mua_new_1*ones(size(cfg3.vol))); % values to substitute corresponding optical properties
replace_property(:,:,31:35)=n_new_1;
replace_property(:,:,36:40)=n_new_2;
replace_property(:,:,41:45)=n_new_3;
replace_property(:,:,46:50)=n_new_4;
replace_property(:,:,51:55)=n_new_5;
replace_property(:,:,56:60)=n_new_6;
cfg3.vol=permute(cat(4,replace_property,label,property_substitute),[4 1 2 3]); % Concatenated as a 4-D array 
flux2_new=mcxlab(cfg3);

clines=[-12:1:-2];
% compare
figure(3);
contourf(log10(squeeze(flux2_conventional.data(31,:,:))*cfg3.tstep),clines,'r-');axis equal;
hold on;
contour(log10(squeeze(flux2_new.data(31,:,:))*cfg3.tstep),clines,'w--');
legend('conventional','new');
title('multi-layered slab');