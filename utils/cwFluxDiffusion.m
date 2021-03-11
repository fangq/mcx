function [flux]= cwFluxDiffusion(mua, musp, n, srcpos, detpos)
%
%  [flux] = cwFluxDiffusion(mua, musp, n, srcpos, detpos)
%
%  compute surface flux for a semi-infinite medium
%
%    author: Qianqian Fang (q.fang <at> neu.edu)
%
%    input:
%        mua:   the absorption coefficients in 1/mm
%        musp:  the reduced scattering coefficients in 1/mm
%        n:     the relative refractive index
%        srcpos:array for the source positions (x,y,z)
%        detpos:array for the detector positions (x,y,z)
%
%    output:
%        flux:  the diffuse reflectance for all source/detector pairs
%
%    this file is part of Monte Carlo eXtreme (MCX)
%    License: GPLv3, see http://mcx.sf.net for details
%    see Boas2002, Haskell1994, Kienle1997
%
    Reff = findReff(n);
    D = 1 / (3 * (mua + musp));
    z0 = 1 / (mua + musp);
    zb = (1 + Reff) / (1 - Reff) * 2 * D;
    muEff = sqrt(3 * mua * (mua + musp));
    r1 = getdistance([srcpos(:,1:2) srcpos(:,3) + z0],detpos);
    r2 = getdistance([srcpos(:,1:2) srcpos(:,3) + z0 + 2 * zb],detpos);

    flux = 1 / (4 * pi) * (z0 * (muEff + 1 ./ r1) .* exp(-muEff .* r1) ./ r1.^2 + ...
        (z0 + 2 * zb) * (muEff + 1 ./ r2) .* exp(-muEff .* r2) ./ r2.^2); % Eq. 6 of Kienle1997
end