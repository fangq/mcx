function [data, header]=loadmch(fname,format)
%    [data, header]=loadmch(fname,format)
%
%    author: Qianqian Fang (fangq <at> nmr.mgh.harvard.edu)
%
%    input:
%        fname: the file name to the output .mch file
%        format:a string to indicate the format used to save
%               the .mc2 file; if omitted, it is set to 'float'
%
%    output:
%        data:  the output MCX detected photon data array
%        hd:    file header info, a row vector containing:
% [version,medianum,detnum,recordnum,totalphoton,detectedphoton,savedphoton,lengthunit]
%
%    this file is part of Monte Carlo eXtreme (MCX)
%    License: GPLv3, see http://mcx.sf.net for details


if(nargin==1)
   format='float';
end

fid=fopen(fname,'rb');
magicheader=fread(fid,4,'char')
hd=fread(fid,7,'uint')
unitmm=fread(fid,1,'float32')
junk=fread(fid,7,'uint')

dat=fread(fid,hd(7)*hd(4),format);
fclose(fid);

dat=reshape(dat,[hd(4),hd(7)])';

header=[hd;unitmm]';
