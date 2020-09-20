function hseg=mcxplotshapes(jsonshape,gridsize,offset,hseg,varargin)
%
% Format:
%    mcxplotshapes(jsonshapestr)
%    handles=mcxplotshapes(jsonshapestr)
%    handles=mcxplotshapes(jsonshapestr,gridsize,offset,oldhandles,...)
%
% Create MCX simulation from built-in benchmarks (similar to "mcx --bench")
%
% Author: Qianqian Fang <q.fang at neu.edu>
%
% Input:
%    jsonshapestr: an MCX shape json string with a root object "Shapes"
%    gridsize (optional): this should be set to size(cfg.vol), default is [60 60 60]
%    offset (optional): this should be set to 1-cfg.issrcfrom0, default is 1
%    oldhandles (optional): existing plot handles
%
% Output:
%    handles: an array of all plot object handles
%
% Dependency:
%    this function depends on the Iso2Mesh toolbox (http://iso2mesh.sf.net)
%
% Example:
%    mcxplotshapes('{"Shapes":[{"Grid":{"Tag":1,"Size":[60, 60, 200]}},{"ZLayers":[[1,20,1],[21,32,4],[33,200,3]]}]}')
%    hfig=mcxpreview(mcxcreate('sphshell'));
%    hfig2=mcxpreview(mcxcreate('spherebox'));
%
% This function is part of Monte Carlo eXtreme (MCX) URL: http://mcx.space
%
% License: GNU General Public License version 3, please read LICENSE.txt for details
%

if(nargin<4)
    hseg=[];
end

shapes=loadjson(jsonshape);
if(nargin<3)
    offset=1;
    if(nargin<2)
        gridsize=[60 60 60];
    end
end

isholdplot=ishold;

orig=[0 0 0]+offset;

rngstate = rand ('state');
randseed=hex2dec('623F9A9E');
rand('state',randseed);
surfcolors=rand(1024,3);

hold on;

if(isfield(shapes,'Shapes'))
    for j=1:length(shapes.Shapes)
        shp=shapes.Shapes{j};
        sname=fieldnames(shp);
        tag=1;
        switch(sname{1})
            case {'Grid','Box','Subgrid'}
                if(strcmp(sname{1},'Grid') && ~isempty(hseg))
                    delete(hseg);
                    hseg=[];
                end
                obj=shp.(sname{1});
                if(isfield(obj,'Tag'))
                    tag=obj.Tag;
                end
                gridsize=obj.Size;
                [no,fc]=latticegrid([0 obj.Size(1)],[0 obj.Size(2)],[0 obj.Size(3)]);
                hseg(end+1)=plotmesh(no,fc,'facealpha',0.3, 'linestyle', '-', 'facecolor','none', varargin{:});
            case 'Sphere'
                obj=shp.(sname{1});
                if(isfield(obj,'Tag'))
                    tag=obj.Tag;
                end
                [sx,sy,sz]=sphere;
                hseg(end+1)=surf(sx*obj.R+(obj.O(1)+orig(1)), ...
                     sy*obj.R+(obj.O(2)+orig(2)), sz*obj.R+(obj.O(3)+orig(3)), ...
                     'facealpha',0.3,'facecolor',surfcolors(tag+1,:),'linestyle','none');
            case 'Cylinder'
                obj=shp.(sname{1});
                if(isfield(obj,'Tag'))
                    tag=obj.Tag;
                end
                c0=obj.C0;
                c1=obj.C1;
                len=norm(c0-c1);

                [sx,sy,sz]=cylinder(obj.R);
                sz=sz*len;
                no=rotatevec3d([sx(:),sy(:),sz(:)],c1-c0);
                sx=reshape(no(:,1),2,size(no,1)/2);
                sy=reshape(no(:,2),2,size(no,1)/2);
                sz=reshape(no(:,3),2,size(no,1)/2);
                hseg(end+1)=surf(sx+(c0(1)+orig(1)), ...
                     sy+(c0(2)+orig(2)), sz+(c0(3)+orig(3)), ...
                     'facealpha',0.3,'facecolor',surfcolors(tag+1,:),'linestyle','none');
            case 'Origin'
                orig=shp.(sname{1});
                hseg(end+1)=plot3(orig(1),orig(2),orig(3),'m*');
            case {'XSlabs','YSlabs','ZSlabs'}
                obj=shp.(sname{1});
                if(isfield(obj,'Tag'))
                    tag=obj.Tag;
                end
                for k=1:size(obj.Bounds,1)
                    switch(sname{1})
                        case 'XSlabs'
                            [no,fc]=latticegrid([obj.Bounds(k,1) obj.Bounds(k,2)]+orig(1),[0 gridsize(2)],[0 gridsize(3)]);
                        case 'YSlabs'
                            [no,fc]=latticegrid([0 gridsize(1)],[obj.Bounds(k,1) obj.Bounds(k,2)]+orig(2),[0 gridsize(3)]);
                        case 'ZSlabs'
                            [no,fc]=latticegrid([0 gridsize(1)],[0 gridsize(2)],[obj.Bounds(k,1) obj.Bounds(k,2)]+orig(3));
                    end
                    hseg(end+1)=plotmesh(no,fc,'facealpha',0.3, 'linestyle', 'none', 'facecolor',surfcolors(tag+1,:), varargin{:});
                end
            case {'XLayers','YLayers','ZLayers'}
                obj=shp.(sname{1});
                if(~iscell(obj))
                    obj=num2cell(obj,2);
                end
                for k=1:length(obj)
                    tag=1;
                    if(length(obj{k})>=3)
                        tag=obj{k}(3);
                    end
                    switch(sname{1})
                        case 'XLayers'
                            [no,fc]=latticegrid([obj{k}(1)-1 obj{k}(2)]+orig(1)-1,[0 gridsize(2)],[0 gridsize(3)]);
                        case 'YLayers'
                            [no,fc]=latticegrid([0 gridsize(1)],[obj{k}(1)-1 obj{k}(2)]+orig(2)-1,[0 gridsize(3)]);
                        case 'ZLayers'
                            [no,fc]=latticegrid([0 gridsize(1)],[0 gridsize(2)],[obj{k}(1)-1 obj{k}(2)]+orig(3)-1);
                    end
                    hseg(end+1)=plotmesh(no,fc,'facealpha',0.3, 'linestyle', 'none', 'facecolor',surfcolors(tag+1,:), varargin{:});
                end
            otherwise
                error('unsupported shape constructs');
        end
    end
end

if(~isholdplot)
    hold off;
end

rand ('state',rngstate);
