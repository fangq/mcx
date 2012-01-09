fid=fopen('jsonshape.mask','rb');
dat=fread(fid,inf,'uchar');
fclose(fid);

dat=reshape(dat,[40 60 50]);
figure;
imagesc(squeeze(dat(:,:,15)));
axis equal

