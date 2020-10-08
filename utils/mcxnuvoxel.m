function [dbratio,uniqidx,bmask,nn,totalarea,iso]=mcxnuvoxel(vol, varargin)
%
% Format:
%    newvol=mcxnuvoxel(vol)
%       or
%    [dbratio,uniqidx,nn,totalarea,iso]=mcxnuvoxel(vol)
%    [dbratio,uniqidx,nn,totalarea,iso]=mcxnuvoxel(vol,'option1',v1,'option2',v2,...)
%
% Preprocessing voxelated labels to find and incorporate curved boundary
% information and improve accuracy
%
% Author: Qianqian Fang <q.fang at neu.edu>
%
% Input:
%    vol: a 3D volume with integer labels, a label 0 is assumed to be
%         background and tissue labels are non-zero positive numbers
%    options (optional): one can add additional 'name', value pairs to
%         specify user options, these options include
%         'debug': [0], if set to 1, print example surfaces for debugging
%         'smoothing': [1], if set to 1, do a Gaussian smoothing for the
%                 labels before extracting isosurface using marching cube
%         'kernelsize': [5], the 3D gaussian kernel size
%         'kernelstd': [2], the 3D gaussian kernel standard-deviation(sigma)
%         'threshold': [0.5], default threshold when extracting isosurfaces
%         'debugpatch': [1], ID of the patch to plot
%         'bmask': [], a pre-computed mask of the same size as vol; a
%                 voxel of non-zero value prevents adding new boundary info
%         'curveonly': [1] if set to 1, remove voxels where patch normals
%                 are align with x/y/z/ axes.
%
% Output:
%    dbratio: a vector of length as uniqidx, the approx. partial volume of 
%         the higher-valued label in a mixed label voxel (between 0-1)
%    uniqidx: the 1-D index of all voxels that are made of mixed labels
%    bmask: an Nx2 array, where N is the length of uniqidx, with the 1st
%         column records the lower-valued label ID, and the 2nd column
%         records the higher-valued label ID in a mixed label voxel
%    nn: nn is a Nx3 array, where N is the length of uniqidx, representing
%         the normalized normal vector in a mixed label voxel, pointing
%         from low to high label numbers
%    totalarea: an Nx1 array, where N is the length of uniqidx, representing
%         the total cross-sectional area of the boundary in each mixed
%         label voxel
%    iso: a struct, iso.vertices denotes the nodes and iso.faces denotes
%         the triangle patches of the combined isosurfaces from all labels.
%
%    if a single output is given, dbratio, bmask, nn, and totalarea are
%    merged into a single 4-D volume, where the indices in the 1st
%    dimension represents
%      1: dbratio, 2-3: bmask, 4-6: nn, 7: totalarea
%
% Example:
%    [xi,yi,zi]=ndgrid(0.5:59.5,0.5:59.5,0.5:59.5);
%    dist=(xi-30).*(xi-30)+(yi-30).*(yi-30)+(zi-30).*(zi-30);
%    vol=zeros(size(dist));
%    vol(dist<10*10)=1;
%    vol(20:40,20:40,1:30)=2;
%    nuvox=mcxnuvoxel(vol);
%
% Dependency:
%    this function depends on the Iso2Mesh toolbox (http://iso2mesh.sf.net)
%
% This function is part of Monte Carlo eXtreme (MCX) URL: http://mcx.space
%
% License: GNU General Public License version 3, please read LICENSE.txt for details
%

%% parse user options
opt=varargin2struct(varargin{:});
dodebug=jsonopt('debug',0,opt);
ksize=jsonopt('kernelsize',3,opt);
kstd=jsonopt('kernelstd',1,opt);
level=jsonopt('threshold',0.5,opt);
debugpatch=jsonopt('debugpatch',1,opt);
bmask=jsonopt('bmask',zeros(size(vol)),opt);
dosmooth=jsonopt('smoothing',1,opt);
curveonly=jsonopt('curveonly',1,opt);

bmask2=zeros(size(vol));

%% read unique labels

labels=sort(unique(vol(:)));
labels(labels==0)=[];

if(length(labels)>255)
    error('MCX currently supports up to 255 labels for this function');
end

%% loop over unique labels in ascending order

iso=struct('vertices',[],'faces',[]);

for i=1:length(labels)
    % convert each label into a binary mask, smooth it, then extract the
    % isosurface using marching cube algorithm (matlab builtin)
    if(dosmooth)
        volsmooth=smooth3(vol==labels(i),'g',ksize,kstd);
    else
        volsmooth=(vol>=labels(i));
    end
    fv0=isosurface(volsmooth,level);
    if(isempty(fv0.vertices))
        continue;
    end

    % get the containing voxel linear id
    c0=meshcentroid(fv0.vertices,fv0.faces);
    voxid=sub2ind(size(vol),floor(c0(:,1))+1,floor(c0(:,2))+1,floor(c0(:,3))+1);
    % identify unique voxels
    uniqidx=unique(voxid);
    bmask2(uniqidx)=labels(i);
    
    % find new boundary voxels that are not covered by previous levelsets
    uniqidx=uniqidx(bmask(uniqidx)==0);
    goodpatchidx=ismember(voxid,uniqidx);
    
    % merge surface patches located inside new boundary voxels
    iso.faces=[iso.faces; size(iso.vertices,1)+fv0.faces(goodpatchidx==1,:)];
    iso.vertices=[iso.vertices; fv0.vertices];
    
    % label those voxels as covered
    bmask(uniqidx)=labels(i);
end

%% handle uniform domains
if(isempty(iso.vertices))
    dbratio=[];
    uniqidx=[];
    nn=[];
    totalarea=[];
    return;
end

%% get the voxel mapping for the final combined isosurface
[iso.vertices,iso.faces]=removeisolatednode(iso.vertices,iso.faces);
c0=meshcentroid(iso.vertices,iso.faces);
voxid=sub2ind(size(vol),floor(c0(:,1))+1,floor(c0(:,2))+1,floor(c0(:,3))+1);
[uniqidx, vox2patch]=unique(voxid);
    
[cc1,cc2]=histc(voxid,uniqidx);
cc=cc1(cc2);

if(dodebug)
    plotmesh(iso.vertices,iso.faces,'facealpha',0.4,'facecolor','b','edgealpha',0.2)
    disp(max(iso.vertices)-min(iso.vertices))
end

%% obtain the low and high labels in all mix-label voxels
bmask=[bmask(uniqidx),bmask2(uniqidx)];
bmask(bmask(:,1)==bmask(:,2),2)=0;

%% computing total area and normal vector for each boundary voxel
areas=elemvolume(iso.vertices,iso.faces);
normals=surfacenorm(iso.vertices,iso.faces);

totalarea=zeros(size(vol));
maxvid=max(voxid);
totalarea(1:maxvid)=accumarray(voxid,areas); % total areas of cross-sections in each boundary voxel
totalarea=totalarea(uniqidx);

%% creating weighted average surface patch normals per boundary voxel
nn=zeros([3 size(vol)]);
nn(1,1:maxvid)=accumarray(voxid,normals(:,1).*areas); % total normal_x of cross-sections in each boundary voxel
nn(2,1:maxvid)=accumarray(voxid,normals(:,2).*areas); % total normal_y of cross-sections in each boundary voxel
nn(3,1:maxvid)=accumarray(voxid,normals(:,3).*areas); % total normal_z of cross-sections in each boundary voxel
nnlen=sqrt(sum(nn.*nn,1));
for i=1:3
    nn(i,voxid)=nn(i,voxid)./nnlen(voxid)';
end
nn=nn(:,uniqidx)';

%% calculating distances between 8 nodes of the enclosing voxel and the
% centroid of the surface patch
p0=[floor(c0(vox2patch,1)) floor(c0(vox2patch,2)) floor(c0(vox2patch,3))];
p0=repmat(p0,1,8)+repmat([0 0 0  1 0 0  0 1 0  0 0 1  1 1 0  1 0 1  0 1 1  1 1 1],size(p0,1),1);

pdiff=reshape(p0-repmat(c0(vox2patch,:),1,8), length(vox2patch), 3, 8);
pdiff=squeeze(sum(pdiff.*repmat(nn,[1,1,8]),2)); % pdiff=dr .* n: unique voxels x3

dbvox=zeros(size(pdiff,1),2);
[s1,s2]=find(pdiff>0);
dbvox(:,1)=accumarray(s1, pdiff(pdiff>0));    % summing positive distances
[s1,s2]=find(pdiff<0);
dbvox(:,2)=-accumarray(s1, pdiff(pdiff<0));   % dbvox: unique voxels x2, summing negative distances

dbratio=dbvox(:,2)./(dbvox(:,1)+dbvox(:,2));  % dbratio: unique voxels

%% remove x/y/z oriented 
if(curveonly)
    [ix,iy]=find(abs(nn)==1);
    boxmask=zeros(size(vol));
    boxmask(uniqidx(ix))=1;
    boxmask=smooth3(boxmask,'b',3);
    boxmask=(boxmask>0);
    ix=find(boxmask(uniqidx)==1);

    nn(ix,:)=[];
    dbratio(ix)=[];
    uniqidx(ix)=[];
    bmask(ix,:)=[];
    totalarea(ix)=[];
end

%% assemble the final volume

if(nargout==1)
    newvol=zeros([7,size(vol)]);
    newvol(1,uniqidx)=dbratio;
    newvol(2:3,uniqidx)=bmask';
    newvol(4:6,uniqidx)=nn';
    newvol(7,uniqidx)=totalarea';
    dbratio=newvol;
end

%% plotting for verification

if(dodebug)
    figure;

    pcidx=find(cc>1);         % find 4-patch that blong to the same voxel (in patch idx)
    pidx0=pcidx(debugpatch);          % pick one such patch group to debug
    idx1=voxid(pidx0);        % find the corresponding voxel id idx1 for the patch pidx0
    patid=find(voxid==idx1);  % these patches are within voxel linear id idx1

    testmask=zeros(size(volsmooth));
    testmask(idx1)=1;
    [no3,fc3]=binsurface(testmask,4);

    figure;
    plotmesh(no3,fc3,'facealpha',0.3,'facecolor','none')
    hold on;
    plotmesh(iso.vertices,iso.faces(patid,:),'facealpha',0.3,'facecolor','b','edgealpha',0.2)
    disp(nn(cc2(pidx0),:))           % normal in voxel id idx1
    plotmesh([c0(pidx0,:); c0(pidx0,:)+nn(cc2(pidx0),:)],'ro-');   % plot centroid of pidx0-th patch to the normal in voxel id idx1
    disp(dbratio(cc2(patid(1))))     % mix ratio of the selected voxel
end

