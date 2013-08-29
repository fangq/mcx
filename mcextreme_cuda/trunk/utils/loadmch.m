function [data, headerstruct, photonseed]=loadmch(fname,format)
%
%    [data, header]=loadmch(fname,format)
%
%    author: Qianqian Fang (fangq <at> nmr.mgh.harvard.edu)
%
%    input:
%        fname: the file name to the output .mch file
%        format:a string to indicate the format used to save
%               the .mch file; if omitted, it is set to 'float'
%
%    output:
%        data:   the output detected photon data array
%                data has header.medium+2 columns, the first column is the 
%                ID of the detector; the 2nd column is the number of 
%                scattering events for a detected photon; the remaining 
%                columns are the partial path lengths (in mm) for each medium type
%        header: file header info, a structure has the following fields
%                [version,medianum,detnum,recordnum,totalphoton,
%                 detectedphoton,savedphoton,lengthunit]
%        photonseed: (optional) if the mch file contains a seed section, this
%                returns the seed data for each detected photon. Each row of 
%                photonseed is a byte array, which can be used to initialize a  
%                seeded simulation. Note that the seed is RNG specific. You must use
%                the an identical RNG to utilize these seeds for a new simulation.
%
%    this file is part of Monte Carlo eXtreme (MCX)
%    License: GPLv3, see http://mcx.sf.net for details
%

if(nargin==1)
   format='float';
end

fid=fopen(fname,'rb');

data=[];
header=[];
photonseed=[];

while(~feof(fid))
	magicheader=fread(fid,4,'char');
	if(strcmp(char(magicheader(:))','MCXH')~=1)
		if(isempty(header))
			fclose(fid);
			error('can not find a MCX history data block');
		end
		break;
	end
	hd=fread(fid,7,'uint'); % version, maxmedia, detnum, colcount, totalphoton, detected, savedphoton
	if(hd(1)~=1) error('version higher than 1 is not supported'); end
        unitmm=fread(fid,1,'float32');
	seedbyte=fread(fid,1,'uint');
	junk=fread(fid,6,'uint');

	dat=fread(fid,hd(7)*hd(4),format);
	dat=reshape(dat,[hd(4),hd(7)])';
	dat(:,3:(2+hd(2)))=dat(:,3:(2+hd(2)))*unitmm;
	data=[data;dat];
        if(seedbyte>0)
            try
              seeds=fread(fid,hd(7)*seedbyte,'uchar');
              seeds=reshape(seeds,[seedbyte,hd(7)])';
              photonseed=[photonseed;seeds];
            catch
              seedbyte=0;
              warning('photon seed section is not found');
            end
        end
	if(isempty(header))
		header=[hd;unitmm]';
	else
		if(any(header([1:4 8])~=[hd([1:4])' unitmm]))
			error('loadmch can only load data generated from a single session');
		else
			header(5:7)=header(5:7)+hd(5:7)';
		end
	end
end

fclose(fid);

if(nargout>=2)
   headerstruct=struct('version',header(1),'medianum',header(2),'detnum',header(3),...
                       'recordnum',header(4),'totalphoton',header(5),...
                       'detectedphoton',header(6),'savedphoton',header(7),...
                       'lengthunit',header(8),'seedbyte',seedbyte);
end
