function Reff = rbgetreff(n_in,n_out)
%
% Reff = rbgetreff(n_in,n_out)
%
% given refractive index of the diffuse medium, calculate the effective
% refractive index, defined as in Haskell 1994 paper.
%
% author: David Boas <dboas at bu.edu>
%
% input:
%     n_in: the refractive index n of the interior of the domain
%     n_out: the refractive index n of the outside space
%
% output:
%     Reff: effective reflection coefficient, see 
%
% license:
%     GPL version 3, see LICENSE_GPLv3.txt files for details 
%
%    original file name calcExtBnd
%    this file was modified from the PMI toolbox
% -- this function is part of Redbird-m toolbox
%

if(nargin==1)
    n_out=1;
end

oc = asin(1/n_in);
ostep = pi / 2000;

o = 0:ostep:oc;

cosop = (1-n_in^2 * sin(o).^2).^0.5;
coso = cos(o);
r_fres = 0.5 * ( (n_in*cosop-n_out*coso)./(n_in*cosop+n_out*coso) ).^2;
r_fres = r_fres + 0.5 * ( (n_in*coso-n_out*cosop)./(n_in*coso+n_out*cosop) ).^2;

r_fres(ceil(oc/ostep):1000) = 1;

o = 0:ostep:ostep*(length(r_fres)-1);
coso = cos(o);

r_phi_int = 2 * sin(o) .* coso .* r_fres;
r_phi = sum(r_phi_int) / 1000 * pi/2;

r_j_int = 3 * sin(o) .* coso.^2 .* r_fres;
r_j = sum(r_j_int) / 1000 * pi/2;

Reff = (r_phi + r_j) / (2 - r_phi + r_j);
