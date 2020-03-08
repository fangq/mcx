function h = islicer(vol,T, varargin)
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

h1 = slice3i(vol,T,1,round(size(vol,1)/2));
h2 = slice3i(vol,T,2,round(size(vol,2)/2));
h3 = slice3i(vol,T,3,round(size(vol,3)/2));

if(nargout>=1)
    h = [h1,h2,h3];
end

set([h1,h2,h3],'CDataMapping','scaled', varargin{:});

%colormap(jet(64));

view(3);
set(camlight,'Visible','on')

axis equal;



 