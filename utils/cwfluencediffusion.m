function varargout=cwfluencediffusion(varargin)
%
%  [Phi r]= cwfluencediffusion(mua, musp, Reff, srcpos,detpos)
%
%  analytical solution to fluence in semi-infinite medium (diffusion model)
%
%  author: Shijie Yan (yan.shiji <at> northeastern.edu)
%
%    input/output: please see details in the help for cwdiffusion
%
%    this file is part of Monte Carlo eXtreme (MCX)
%    License: GPLv3, see http://mcx.sf.net for details
%    see Boas2002, Haskell1994
%
[varargout{1:nargout}]=cwdiffusion(varargin{:});