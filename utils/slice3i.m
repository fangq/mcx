function h = slice3i(vol, I2X, slicedim, sliceidx, handle)
% Display a slice from a volume in 3-D
% h = slice3(vol, I2X, slicedim, sliceidx, handle) 
%
% Vol is either a scalar or RGB volume, e.g. N x M x K or N x M x K x 3.
% I2X is a transformation matrix from volume coordinates to xyz 3-D world
% coordinates, similar to the transform used by image3i.
%
%     [x y z 1]' = I2X * [i j k 1]'
%
% slicedim  is 1, 2 or 3 and corresponds to i, j and k.
% sliceidx  the index of the slice along slicedim
% handle    an optional handle to a previous slice to reuse
% h         the handle to the created slice.
%
% Example:
%
% load mri;
% T = [1 0 0 0;0 1 0 0;0 0 2.5 0];
% h1 = slice3i(squeeze(D),T,1,64);
% h2 = slice3i(squeeze(D),T,2,64);
% h3 = slice3i(squeeze(D),T,3,14);
% colormap gray(88);
% view(30,45); axis equal;
%
% SEE ALSO: image3i, imagesc3
%
% Author: Anders Brun, anders@cb.uu.se (2009)
%
% Modified by Qianqian Fang, q.fang at neu.edu
%
% Features added: 
%    1) compatible with octave 4.0 or newer
%    2) start 3D rotation by clicking/holding the middle mouse key
%    3) adjust colormap levels by dragging with the right mouse key
%    4) support 4-D data (up/down key to change frames)
% 

try
    h = update_slice(vol, I2X, slicedim, sliceidx, h);
catch
    h = update_slice(vol, I2X, slicedim, sliceidx);
end
    
% set up gui
gui.handle = h;
gui.vol = vol;
gui.I2X = I2X;
gui.slicedim = slicedim;
gui.sliceidx = sliceidx;
set(gui.handle,'ButtonDownFcn',@startmovit);

set(h,'UserData',gui);

function stoprotate(src,hrot)
if strcmp(get(gcf,'SelectionType'), 'extend')
    rotate3d off;
end

function startmovit(src,evnt)

if strcmp(get(gcf,'SelectionType'), 'extend')
    rotate3d on;
    return;
end

% Unpack gui object
gui = get(src,'UserData');
%turn off mouse pointer
%set(gcf,'PointerShapeCData',nan(16,16));
%set(gcf,'Pointer','custom');

thisfig = gcbf();
gui.startray = get(gca,'CurrentPoint');
gui.startidx = gui.sliceidx;
gui.p0=get(gcf,'currentpoint');
gui.level0=size(colormap,1);


% Store gui object
set(src,'UserData',gui);
set(thisfig,'WindowButtonMotionFcn',@movit);
set(thisfig,'WindowButtonUpFcn',@stopmovit);
set(thisfig,'UserData',src);


function movit(src,evnt)
% Unpack gui object
gui = get(get(gcf,'UserData'),'UserData');
% Some safetymeasures
try
if isequal(gui.startray,[])
    return
end
catch
end

if strcmp(get(gcf,'SelectionType'), 'alt')
    if(isempty(gui) || ~isfield(gui,'p0'))
        return;
    end
    delta=get(gcf,'currentpoint')-gui.p0;
    windim=get(gcf,'position');
    delta=delta./windim(3:4);
    currentlevel=round(gui.level0+delta(2)/4*256);
    if(currentlevel>256)
        currentlevel=256;
    elseif(currentlevel<8)
        currentlevel=8;
    end
    colormap(jet(currentlevel));
    return;
end

% Do "smart" positioning of the markers...
nowray = get(gca,'CurrentPoint');

% Project rays on slice-axis
s = gui.I2X(1:3,gui.slicedim);
a = gui.startray(1,:)';
b = gui.startray(2,:)';

alphabeta = pinv([s'*s, -s'*(b-a);(b-a)'*s, -(b-a)'*(b-a)])*[s'*a, (b-a)'*a]';
pstart = alphabeta(1)*s;
alphastart = alphabeta(1);
a = nowray(1,:)';
b = nowray(2,:)';
alphabeta = pinv([s'*s, -s'*(b-a);(b-a)'*s, -(b-a)'*(b-a)])*[s'*a, (b-a)'*a]';
pnow = alphabeta(1)*s;
alphanow = alphabeta(1);
slicediff = alphanow-alphastart;

gui.sliceidx = gui.startidx+slicediff;
gui.sliceidx = min(max(1,gui.sliceidx),size(gui.vol,gui.slicedim));
update_slice(gui.vol, gui.I2X, gui.slicedim, gui.sliceidx, gui.handle);
drawnow;

% Store gui object
set(get(gcf,'UserData'),'UserData',gui);


function stopmovit(src,evnt)

thisfig = gcbf();
%set(gcf,'Pointer','arrow');
set(thisfig,'WindowButtonUpFcn','');
set(thisfig,'WindowButtonMotionFcn','');
gui = get(get(gcf,'UserData'),'UserData');
gui.startray = [];
%gui.sliceidx = gui.nextsliceidx
set(get(gcf,'UserData'),'UserData',gui);
set(gcf,'UserData',[]);
drawnow;

function h = update_slice(vol, I2X, slicedim, sliceidx, handle)

if ndims(vol) == 3         %Scalar mode
elseif ndims(vol) == 4     %RGB mode
    if(size(vol,4)>3)
        gui = get(get(gcf,'UserData'),'UserData');
        if(~isempty(gui) && isfield(gui,'frame'))
            vol=gui.vol(:,:,:,gui.frame);
        else
            vol=gui.vol(:,:,:,1);
        end
    end
else
  error('Only scalar and RGB images supported')
end

% Create the slice
if slicedim == 3 % k
  ij2xyz = I2X(:,[1 2]);
  ij2xyz(:,3) = I2X*[0 0 sliceidx 1]';
  sliceim = squeeze(vol(:,:,round(sliceidx),:));

elseif slicedim == 2 % j
  ij2xyz = I2X(:,[1 3]);
  ij2xyz(:,3) = I2X*[0 sliceidx 0 1]';
  sliceim = squeeze(vol(:,round(sliceidx),:,:));

elseif slicedim == 1 % i
  ij2xyz = I2X(:,[2 3]);
  ij2xyz(:,3) = I2X*[sliceidx 0 0 1]';
  sliceim = squeeze(vol(round(sliceidx),:,:,:));
else
    error('Slicedim should be 1, 2 or 3')
end


if nargin<5 || handle == 0
  h = image3i(sliceim,ij2xyz);
else
  h = image3i(sliceim,ij2xyz,handle);
end

