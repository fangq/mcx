function [data, dref]=loadmc2(fname,dim,format)
%
%    data=loadmc2(fname,dim,format)
%       or
%    [data dref]=loadmc2(fname,dim,format)
%
%    author: Qianqian Fang (q.fang <at> neu.edu)
%
%    input:
%        fname: the file name to the output .mc2 file
%        dim:   an array to specify the output data dimension
%               normally, dim=[nx,ny,nz,nt]
%        format:a string to indicate the format used to save
%               the .mc2 file; if omitted, it is set to 'float'
%
%    output:
%        data:  the output MCX solution data array, in the
%               same dimension specified by dim
%        dref(optional): diffuse reflectance at the surface of the domain.
%               if this output is not given while diffuse reflectance 
%               is recorded, dref is shown as the negative values in 
%               the data output.
%
%    this file is part of Monte Carlo eXtreme (MCX)
%    License: GPLv3, see http://mcx.sf.net for details
%

if(nargin==2)
   format='float';
end

fid=fopen(fname,'rb');
data=fread(fid,inf,format);
fclose(fid);

data=reshape(data,dim);

if(nargout>1)
   dref=-data;
   dref(dref<0)=0;
   data(data<0)=0;
end
