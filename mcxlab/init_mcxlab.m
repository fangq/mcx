function envpath=init_mcxlab(cudalib)
%
% setenv('LD_LIBARARY_PATH', init_mcxlab);
%    or
% setenv('LD_LIBARARY_PATH', init_mcxlab(cudalib));
%
% initialize the external cuda library in a standard matlab command window
%
% author: Qianqian Fang (fangq<at> nmr.mgh.harvard.edu)
% date: 2011/03/15
%
% input:
%      cudalib: full path to the folder where libcudart.so* (for unix) 
%               or cudart.dll (for windows) can be found;
%               if not given, this script will assume the default 
%               CUDA installation path.
%
% -- this function is part of Monte Carlo eXtreme (http://mcx.sf.net)
%

if(nargin==1)
	cudartpath=cudalib;
else
	comp=computer;
	cudartpath='/usr/local/cuda/lib';
	if(isunix)
		if(~isempty(regexp(comp,'[A-Za-z]+64','ONCE'))) % is a 64 bit machine
			cudartpath='/usr/local/cuda/lib64';
		end
	else
                cudartpath='C:\CUDA\bin';
	end
end

if(~exist(cudartpath,'dir'))
	error('specified directory does not exist');
end

if(isempty(dir(sprintf('%s%s%s*',cudartpath,filesep,'libcudart.'))))
        error(sprintf('libcudart.* file are not found under folder %s',cudartpath));
end

envpath = getenv('LD_LIBRARY_PATH');
if(isempty(envpath))
	envpath=cudartpath;
else
	envpath=[envpath ':' cudartpath];
end
