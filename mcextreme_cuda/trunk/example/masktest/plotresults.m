fid=fopen('mtest.mask','rb'); 
dat=fread(fid,inf,'uchar');
fclose(fid);

dat=reshape(dat,[8 8 8]);
pcolor(dat(:,:,1)')

r=3;
s=0:pi/20:2*pi;
c0=[4.9,4.1];
x=c0(1)+r*cos(s);
y=c0(2)+r*sin(s);

hold on
plot(x,y,'y-');

