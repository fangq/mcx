function [dat, filename, handles, hfig]=mcxplotvol(varargin)
%
%    [dat, filename, handles]=mcxplotvol()
%       or
%    [dat, filename, handles]=mcxplotvol(fname)
%    [dat, filename, handles]=mcxplotvol(data)
%    [dat, filename, handles]=mcxplotvol(fname,dim,format)
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
%        dat:(optional)  the 3-D or 4-D data being plotted
%        filename: (optional) the name of the file being plotted
%        handles:(optional) the handles to the slice surface object
%
%    this file is part of Monte Carlo eXtreme (MCX)
%    License: GPLv3, see http://mcx.sf.net for details
%

if(nargin>=1)
    if(ischar(varargin{1}))
        fname=varargin{1};
        data=mcxloadfile(fname, varargin{2:end});
    else
        fname='';
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

%if(ndims(squeeze(data))==4)
%    data=sum(squeeze(data),4);
%end

if(nargout>=2)
    filename=fname;
elseif(nargout>=1)
    dat=data;
end

hfig=figure;

guidata=struct('filename',fname,'data',data,'frame',1);
guidata.handles=islicer(data(:,:,:,guidata.frame));

set(hfig,'WindowKeyPressFcn',@changeframe)
set(hfig,'name',['MCX Viewer (mcxplotvol) ' fname]);
set(hfig,'NumberTitle','off');
set(gca,'UserData',guidata);


helpinfo=sprintf('Drag mouse left-btn  to move slices;\r Click and drag mouse mid-btn to rotate; \rDrag right-key up-down to change color level;\rUp-arrow key :next time-gate; \rDown-arrow key:prev time-gate');

helpbtn=uicontrol('Parent',hfig,'Style','pushbutton','String','Help','Units','points','Position',[10 10 50 20],'Visible','on','Callback',@showhelp);

set(helpbtn,'TooltipString',helpinfo);


xlabel(sprintf('x (frame=%d of %d)',1,size(data,4)));
ylabel('y');
zlabel('z');

colorbar;


function changeframe(src,event)

guidata=get(gca,'UserData');

if(isempty(guidata) || ~isfield(guidata,'frame'))
    return;
end

newframe=-1;
switch(event.Key)
    case 'uparrow'
         newframe=min(guidata.frame+1,size(guidata.data,4));
    case 'downarrow'
         newframe=max(guidata.frame-1,1);
end

if(newframe>0 && newframe~=guidata.frame)
    delete(guidata.handles);
    guidata.handles=islicer(guidata.data(:,:,:,newframe));
    xlabel(sprintf('x (frame=%d of %d)',newframe,size(guidata.data,4)));
    guidata.frame=newframe;
    set(gca,'UserData',guidata);
end

function showhelp(source,event)
msgbox(get(source,'TooltipString'),'Help info');
