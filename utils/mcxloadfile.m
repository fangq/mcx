function [data, header]=mcxloadfile(fname,varargin)
%
%    [data, header]=mcxloadfile(fname)
%       or
%    [data, header]=mcxplotvol(fname,dim,format)
%
%    author: Qianqian Fang (q.fang <at> neu.edu)
%
%    input:
%        fname: the file name to the output .mc2/.nii/binary volume file
%        dim:   an array to specify the output data dimension
%               normally, dim=[nx,ny,nz,nt]
%        format:a string to indicate the format used to save
%               the .mc2 file; if omitted, it is set to 'float'
%
%    output:
%        data:  the 3-D or 4-D data being loaded
%        header: a structure recording the metadata of the file
%
%    this file is part of Monte Carlo eXtreme (MCX)
%    License: GPLv3, see http://mcx.sf.net for details
%

[pathstr,name,ext] = fileparts(fname);

if(strcmpi(ext,'.nii'))
    nii=mcxloadnii(fname,varargin{:});
    data=nii.img;
    header=nii.hdr;
elseif(strcmpi(ext,'.mc2'))
    data=loadmc2(fname,varargin{:});
    data=log10(data);
    header.dim=varargin{1};
    header.format=class(data);
    header.scale='log10';
elseif(strcmpi(ext,'.mch'))
    [data, header]=loadmch(fname),varargin{:};
else
    data=loadmc2(fname,varargin{:});
end
