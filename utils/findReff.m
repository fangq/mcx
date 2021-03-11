function Reff = findReff(nRel)
%
% Compute effective reflection coefficent for refractive-index-mismatched
% boundary (section 5.5.2 of Biomedical Optics: Principles and Imaging[2007])
%
% input:
%   nRel: relative refractive index
%
% output:
%   Reff: effective reflection coefficent
%

    f1 = @(x) 2.*sin(x).*cos(x).*(0.5*((nRel*cos(asin(nRel*sin(x)))-cos(x))./(nRel*cos(asin(nRel*sin(x)))+cos(x))).^2+0.5*((nRel*cos(x)-cos(asin(nRel*sin(x))))./(nRel*cos(x)+cos(asin(nRel*sin(x))))).^2);
    f2 = @(x) 2.*sin(x).*cos(x);
    Rphi = integral(f1,0,asin(1/nRel)) + integral(f2,asin(1/nRel),pi/2); %Eq 5.113

    f1 = @(x) 3.*sin(x).*(cos(x)).^2.*(0.5*((nRel*cos(asin(nRel*sin(x)))-cos(x))./(nRel*cos(asin(nRel*sin(x)))+cos(x))).^2+0.5*((nRel*cos(x)-cos(asin(nRel*sin(x))))./(nRel*cos(x)+cos(asin(nRel*sin(x))))).^2);
    f2 = @(x) 3.*sin(x).*(cos(x)).^2;
    Rj = integral(f1,0,asin(1/nRel)) + integral(f2,asin(1/nRel),pi/2); %Eq 5.114

    Reff = (Rphi + Rj)/(2 - Rphi + Rj); %Eq 5.112
end