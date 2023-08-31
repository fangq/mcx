function h = islicer(vol,T, handles, keepxyz, varargin)
%
%    h = islicer(vol);
%       or
%    h = islicer(vol,T);
%
%    author: Anders Brun, anders@cb.uu.se (2009)
%    url: https://www.mathworks.com/matlabcentral/fileexchange/25923-myslicer-make-mouse-interactive-slices-of-a-3-d-volume
%    original version: v1.1 (accessed on 07/25/2017)
%
%    input:
%        vol:  the 3-D or 4-D data being loaded
%        T: a 4x4 transformation matrix; if missing, T is a 4x4 identity
%           matrix
%
%    output:
%        h: the handles to the 3 surfaces plotted along the x/y/z
%           planes
%
%    this file is part of Monte Carlo eXtreme (MCX)
%    License: GPLv3, see http://mcx.sf.net for details
%

if nargin < 2
  T = eye(4);
end

if(nargin < 3)
   handles=[0,0,0];
end

if(nargin < 4)
   keepxyz=0;
end

if(handles(1)==0)
    h1 = slice3i(vol,T,1,round(size(vol,1)/2),handles(1), keepxyz);
else
    h1 = slice3i(vol,T,1,round(min(min(get(handles(1),'xdata')))),handles(1), keepxyz);
end

if(handles(2)==0)
    h2 = slice3i(vol,T,2,round(size(vol,2)/2),handles(2), keepxyz);
else
    h2 = slice3i(vol,T,2,round(min(min(get(handles(2),'ydata')))),handles(2), keepxyz);
end

if(handles(3)==0)
    h3 = slice3i(vol,T,3,round(size(vol,3)/2),handles(3), keepxyz);
else
    h3 = slice3i(vol,T,3,round(min(min(get(handles(3),'zdata')))),handles(3), keepxyz);
end

if(nargout>=1)
    h = [h1,h2,h3];
end

set([h1,h2,h3],'CDataMapping','scaled', varargin{:});

%colormap(jet(64));

view(3);
if(all(handles==0))
    set(camlight,'Visible','on')
end

axis equal;



 