function h = image3i(im,ij2xyz,handle)
% Display a 2-D image in Matlab xyz 3-D space
% h = image3i(C, IJ2XYZ, handle)
%
% C is an image, encoded with scalars or rgb-values, i.e. the dimensions of
% C are either N x M or N x M x 3. Scalars correspond to indices in the
% current colormap and rgb-values should be within the closed 
% interval [0,1]. Similar to the image command, image3i will treat scalars 
% differently depending on if they are doubles or uint8/uint16. Doubles are
% indexed starting at 1 and uint8/uint16 start at 0. By setting the
% property CDataMapping for h, different behaviours can be obtained, e.g.
% usage of the whole range of the colormap by setting it to 'scaled'.
%
% The image is indexed by two positive integers, i and j, which correspond
% to the first and second index of C. Thus i belongs to [1,N] and j belongs
% to [1,M]. Each coordinate correspond to a pixel center.
%
% IJ2XYZ is a transformation matrix from image coordinates to 3-D world
% coordinates. The transformation is described in homogeneous coordinates,
% i.e. we have added one dimension to each coordinate that is always 1.
%
% If u = [i,j,1]' and r = [x,y,z,1]', the transformation is
%
%     r = IJ2XYZ * u
%
% Thus IJ2XYZ = [ix jx cx;iy jy cy; iz jz cz; 0 0 1], where cx,cy and cz
% encode the translation of the image. The ones and zeroes ensure that
% the result of the transformation is also a homogeneous coordinate. Yes,
% you can skip that last row of IJ2XYZ and image3i will not care about 
% it. :-)
%
% The function returns a handle, h, which is a handle to a surf object. Use
% this handle to e.g. modify the image, make it transparent or delete it.
% The handle can also be used as an optional argument to the image3i 
% command. In this case, the the image patch is recycled, which is an
% efficient way of e.g. changing the image data or move the image to
% another location.
%
% Image3 can mimic Matlabs ordinary image command, e.g.
%     prel = image; C = get(prel,'CData'); delete(prel);
%     colormap(jet(64));
%     image(C);
%     view(0,90);
%     axis ij;
%     axis image;
% is equivalent to:
%     prel = image; C = get(prel,'CData'); delete(prel);
%     colormap(jet(64));
%     image3i(C,[0 1 0;1 0 0; 0 0 0]);
%     view(0,90);
%     axis ij;
%     axis image;
% In the above example it _is_ important to hardcode the axis settings
% to compare image and image3i, otherwise Matlabs does some magic that make 
% them look different.
% 
% Some examples of use:
%
% %% Display an image with (1,1) located at (10,20,5).
% C = rand(8,4)*64;                              % a small random image
% R = [1 0; 0 1; 0 0];
% t = [10-1;20-1;5];
% h = image3i(C,[R t]); axis equal
% view(30,45);
%
%
% %% Reuse the image h from the previous example, move (1,1) to (7,17,2).
% %% This technique is useful in interactive applications.
% h = image3i(C,[R t-3],h); axis equal
% 
%
% %% Display an image centered at (15,10,0), rotated 30 degrees ccw in the 
% %% image plane.
% C = rand(8,4)*64;                              % a small random image
% alpha = 30;                                    % rotation, in ccw degrees
% cx = 15; cy = 10; cz = 0;                      % the center position
% R = [+cos(alpha/180*pi) -sin(alpha/180*pi);    % new x coordinate
%      +sin(alpha/180*pi) +cos(alpha/180*pi);    % new y coordinate
%      0 0];                                     % z = 0.
% t = [cx;cy;cz] - R * [size(C,1)/2+0.5; size(C,2)/2+0.5]; % fix center
% h = image3i(C,[R t]); axis equal
%
% To see and manipulate the properties of the image object, use get and set
%
% get(h)
% set(h,'FaceAlpha',0.25) % Make the image transparent
% set(h,'EdgeColor','black');
% set(h,'LineStyle','-');
% 
% SEE ALSO: imagesc3, surf
%
% Author: Anders Brun, anders@cb.uu.se (2008)

% Hardcoded constants
stepsize = 0.1; % Determines number of tiles in image. It is most efficient
              % to set this to 1, but a higher number might render better
              % graphics when the image is distorted a lot by the
              % perspective mapping of the camera. All texture mapping is
              % based on linear interpolation and does not take texture
              % into account, at least it has been so in Matlab in the
              % past.

if ndims(im) == 2         %Scalar mode
  imsize = size(im);
elseif ndims(im) == 3     %RGB mode
  imsize = size(im);
  imsize = imsize(1:2);
else
  error('Only scalar and RGB images supported')
end


% Create the slice
[uu,vv] = ndgrid(0:stepsize:1, 0:stepsize:1);
% ij2xyz refers to voxel centers. Therefore we need to 
% add half a pixel to each size of the slice.
irange = [0.5 imsize(1)+0.5];
jrange = [0.5 imsize(2)+0.5];
% Create three 2D arrays giving the ijk coordinates for 
% this slice.
iidx = irange(1) + uu*(irange(2)-irange(1));
jidx = jrange(1) + vv*(jrange(2)-jrange(1));

% Map these 2D ijk arrays to xyz
x = ij2xyz(1,1)*iidx + ij2xyz(1,2)*jidx + + ij2xyz(1,3);
y = ij2xyz(2,1)*iidx + ij2xyz(2,2)*jidx + + ij2xyz(2,3);
z = ij2xyz(3,1)*iidx + ij2xyz(3,2)*jidx + + ij2xyz(3,3);


if nargin<3 || handle == 0
  % Make a new surface
  h = surface('XData',x,'YData',y,'ZData',z,...
	      'CData', im,...
	      'FaceColor','texturemap',...
	      'EdgeColor','none',...
	      'LineStyle','none',...
	      'Marker','none',...
	      'MarkerFaceColor','none',...
	      'MarkerEdgeColor','none',...
	      'CDataMapping','direct');
else
  % Reuse old surface
  set(handle,'XData',x,'YData',y,'ZData',z,'CData',im);
  h = handle;
end

% Just to be sure...
set(gcf,'renderer','opengl');
%set(gcf,'renderer','zbuffer');

