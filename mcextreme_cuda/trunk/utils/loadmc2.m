function data=loadmc2(fname,dim,format)
if(nargin==2)
   format='float';
end

fid=fopen(fname,'rb');
data=fread(fid,inf,format);
fclose(fid);

data=reshape(data,dim);
