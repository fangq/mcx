function hs=mcxpreview(cfg, varargin)
%
% Format:
%    hs=mcxpreview(cfg);
%
% Preview the simulation configuration for both MCXLAB and MMCLAB
%
% Author: Qianqian Fang <q.fang at neu.edu>
%
% Input:
%    cfg: a struct, or struct array. Each element of cfg defines 
%         the parameters associated with a simulation. Please run
%         'help mcxlab' or 'help mmclab' to see the details.
%         mcxpreview supports the cfg input for both mcxlab and mmclab.
%
% Output:
%    hs: a struct to the list of handles of the plotted domain elements.
%
% Dependency:
%    this function depends on the Iso2Mesh toolbox (http://iso2mesh.sf.net)
%
% This function is part of Monte Carlo eXtreme (MCX) URL: http://mcx.space
%
% License: GNU General Public License version 3, please read LICENSE.txt for details
%

if(nargin==0)
    error('input field cfg must be defined');
end

if(~isstruct(cfg))
    error('cfg must be a struct or struct array');
end

len=length(cfg);
if(nargout>0)
    hs=cell(len,1);
end

rngstate = rand ('state');

randseed=hex2dec('623F9A9E');
rand('state',randseed);
surfcolors=rand(1024,3);
isholdplot=ishold;

for i=1:len
    if(~isfield(cfg(i),'vol') && ~isfield(cfg(i),'node') && ~isfield(cfg(i),'shapes'))
        error('cfg.vol or cfg.node or cfg.shapes is missing');
    end

    if(i>1)
        figure;
    end
    hold on;

    voxelsize=1;
    if(isfield(cfg(i),'unitinmm'))
        voxelsize=cfg(i).unitinmm;
    end

    offset=1;
    if(isfield(cfg(i),'issrcfrom0') && cfg(i).issrcfrom0==1 || isfield(cfg(i),'node'))
        offset=0;
    end
    
    if(isfield(cfg(i),'vol') && ~isfield(cfg(i),'node'))
        % render mcxlab voxelated domain
        dim=size(cfg(i).vol);
        [bbxno,bbxfc]=latticegrid(0:dim(1):dim(1),0:dim(2):dim(2),0:dim(3):dim(3));
        hbbx=plotmesh((bbxno+offset)*voxelsize,bbxfc,'facecolor','none');

        val=unique(cfg(i).vol(:));
        val(val==0)=[];
        padvol=zeros(dim+2);
        padvol(2:end-1,2:end-1,2:end-1)=cfg(i).vol;
        hseg=[];

        if(length(val)>1)
            hseg=zeros(length(val),1);
            for id=1:length(val)
                [no,fc]=binsurface(padvol==val(id));
                hseg(id)=plotmesh((no-1)*voxelsize,fc,'facealpha',0.3, 'linestyle', 'none', 'facecolor',surfcolors(val(id)+1,:), varargin{:});
            end
        end
    else
        % render mmclab mesh domain
        if(~isfield(cfg(i),'node') || ~isfield(cfg(i),'elem'))
            error('cfg.node or cfg.elem is missing');
        end
        elemtype=ones(size(cfg(i).elem,1),1);
        if(isfield(cfg(i),'elemprop'))
            elemtype=cfg(i).elemprop;
        else
            if(size(cfg(i).elem,2)>4)
               elemtype=cfg(i).elem(:,5);
            end
        end
        etypes=unique(elemtype);
        no=cfg(i).node*voxelsize;
        hseg=zeros(length(etypes),1);
        for id=1:size(etypes,1)
            hseg(id)=plotmesh(no,[],cfg(i).elem(elemtype==etypes(id),:),'facealpha',0.3, 'linestyle', 'none', 'facecolor',surfcolors(etypes(id)+1,:), varargin{:});
        end
    end

    if(isfield(cfg(i),'shapes'))
        if(isfield(cfg(i),'vol'))
            hseg=mcxplotshapes(cfg(i).shapes,size(cfg(i).vol),offset,hseg,varargin{:});
        else
            hseg=mcxplotshapes(cfg(i).shapes,[60,60,60],offset,hseg,varargin{:});
        end
    end

    % rendering source position and direction
    if(~isfield(cfg(i),'srcpos') || ~isfield(cfg(i),'srcdir'))
        error('cfg.srcpos or cfg.srcdir is missing');
    end
    
    cfg(i).srcpos=cfg(i).srcpos;
    srcpos=cfg(i).srcpos*voxelsize;
    hsrc=plotmesh(srcpos,'r*');
    srcvec=cfg(i).srcdir*10*voxelsize;
    headsize=1e2;
    if(isoctavemesh)
        headsize=0.5;
    end
    hdir=quiver3(srcpos(1),srcpos(2), srcpos(3), srcvec(1),srcvec(2),srcvec(3), 'linewidth',3, 'color', 'r', 'MaxHeadSize',headsize,'AutoScaleFactor',1, varargin{:});

    % rendering area-source aperature
    hsrcarea=[]; 
    if(isfield(cfg(i),'srctype'))
        if(strcmp(cfg(i).srctype,'disk') || strcmp(cfg(i).srctype,'gaussian') || strcmp(cfg(i).srctype,'zgaussian'))
            if(~isfield(cfg(i),'srcparam1'))
                error('cfg.srcparam1 is missing');
            end
            [ncyl,fcyl]=meshacylinder(srcpos,srcpos+cfg(i).srcdir(1:3)*1e-5,cfg(i).srcparam1(1)*voxelsize,0,0);
            hsrcarea=plotmesh(ncyl,fcyl{end-1},'facecolor','r','linestyle','none');
        elseif(strcmp(cfg(i).srctype,'planar') || strcmp(cfg(i).srctype,'pattern') || strcmp(cfg(i).srctype,'fourier') || ...
               strcmp(cfg(i).srctype,'fourierx') || strcmp(cfg(i).srctype,'fourierx2d') || strcmp(cfg(i).srctype,'pencilarray'))
            if(~isfield(cfg(i),'srcparam1') || ~isfield(cfg(i),'srcparam2'))
                error('cfg.srcparam2 or cfg.srcparam2 is missing');
            end
            if(strcmp(cfg(i).srctype,'fourierx') || strcmp(cfg(i).srctype,'fourierx2d'))
                vec2=cross(cfg(i).srcdir, cfg(i).srcparam1(1:3)*voxelsize);
            else
                vec2=cfg(i).srcparam2(1:3)*voxelsize;
            end
            nrec=[0 0 0; cfg(i).srcparam1(1:3)*voxelsize; cfg(i).srcparam1(1:3)*voxelsize+vec2; vec2];
            hsrcarea=plotmesh(nrec+repmat(srcpos(:)',[4,1]), {[1 2 3 4 1]});
        elseif(strcmp(cfg(i).srctype,'pattern3d'))
            dim=cfg(i).srcparam1(1:3);
            [bbxno,bbxfc]=latticegrid(0:dim(1):dim(1),0:dim(2):dim(2),0:dim(3):dim(3));
            hbbx=plotmesh(((bbxno+repmat(cfg(i).srcpos(1:3),size(bbxno,1),1))+offset)*voxelsize,bbxfc,'facecolor','y','facealpha',0.3);
        elseif(strcmp(cfg(i).srctype,'slit') || strcmp(cfg(i).srctype,'line'))
            if(~isfield(cfg(i),'srcparam1'))
                error('cfg.srcparam1 is missing');
            end
            hsrcarea=plotmesh([srcpos(1:3); cfg(i).srcparam1(1:3)*voxelsize],[1 2],'linewidth',3,'color','r');
        end
    end
    
    % rendering detectors
    hdet=[];
    if(isfield(cfg(i),'detpos'))
        hdet=zeros(size(cfg(i).detpos,1),1);
        detpos=cfg(i).detpos(:,1:3)*voxelsize;
        if(size(cfg(i).detpos,2)==4)
            detpos(:,4)=cfg(i).detpos(:,4)*voxelsize;
        else
            detpos(:,4)=1;
        end
        for id=1:size(detpos,1)
            [sx,sy,sz]=sphere;
            hdet(id)=surf(sx*detpos(id,4)+(detpos(id,1)), ...
                          sy*detpos(id,4)+(detpos(id,2)), ...
                          sz*detpos(id,4)+(detpos(id,3)), ...
                   'facealpha',0.3,'facecolor','g','linestyle','none');
        end
    end
    
    % combining all handles
    if(nargout>0)
        hs{i}=struct('bbx',hbbx, 'seg', hseg, 'src', hsrc, 'srcarrow', hdir, 'srcarea', hsrcarea, 'det', hdet);
    end
end

if(~isholdplot)
    hold off;
end

rand ('state',rngstate);
