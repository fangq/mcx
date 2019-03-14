dat=loadmc2('multipattern.mc2',[3 60 60 60]);
figure;
subplot(131);
imagesc(squeeze(log10(abs(dat(1,:,:,10))))')
axis equal
subplot(132);
imagesc(squeeze(log10(abs(dat(2,:,:,10))))')
axis equal
subplot(133);
imagesc(squeeze(log10(abs(dat(3,:,:,10))))')
axis equal
