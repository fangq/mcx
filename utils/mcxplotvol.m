function [dat, filename]=mcxplotvol(varargin)
%
%    [dat, filename]=mcxplotvol()
%       or
%    [dat, filename]=mcxplotvol(fname)
%    [dat, filename]=mcxplotvol(data)
%    [dat, filename]=mcxplotvol(fname,dim,format)
%
%    author: Qianqian Fang (q.fang <at> neu.edu)
%
%    input:
%        fname: the file name to the output .mc2/.nii/binary volume file
%        data:  if the first input is an array, it is treated as the data
%               to be plotted
%        dim:   an array to specify the output data dimension
%               normally, dim=[nx,ny,nz,nt]
%        format:a string to indicate the format used to save
%               the .mc2 file; if omitted, it is set to 'float'
%
%    output:
%        dat:  the 3-D or 4-D data being plotted
%        filename: the name of the file being plotted
%
%    this file is part of Monte Carlo eXtreme (MCX)
%    License: GPLv3, see http://mcx.sf.net for details
%

if(nargin>=1)
    if(ischar(varargin{1}))
        fname=varargin{1};
        data=mcxloadfile(fname, varargin{2:end});
    else
        data=varargin{1};
    end
else
    [fname pathname]=uigetfile( {'*.*'},'Pick a file');
    fname=fullfile(pathname, fname);
    if(isempty(fname))
        return;
    end
    [pathstr,name,ext] = fileparts(fname);
    if(~strcmpi(ext,'.nii') && length(varargin)<=1)
        prompt = {'Enter x-dimension:','Enter y-dimension:','Enter z-dimension:','Enter frame count:','Format:'};
        dlg_title = 'Input';
        num_lines = 1;
        defaultans = {'','','','1','float32'};
        dim= inputdlg(prompt,dlg_title,num_lines,defaultans);
        dataformat=dim{5};
        dim=cellfun(@(x) str2num(x), dim(1:4));
        data=mcxloadfile(fname, dim(:)',dataformat);
    else
        data=mcxloadfile(fname, varargin{2:end});
    end
end

if(ndims(squeeze(data))==4)
    data=sum(squeeze(data),4);
end

if(nargout>=2)
    filename=fname;
elseif(nargout>=1)
    dat=data;
end

islicer(data);
