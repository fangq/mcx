function dref = cwDrefMC(detp, cfg)
%
% Compute CW diffuse reflectance from MC detected photon profiles
% 
% input:
%   detp: profiles of detected photons
%   cfg:  a struct, or struct array. Each element of cfg defines 
%         the parameters associated with a MC simulation.
%
% output:
%   dref: CW diffuse reflectance at detectors
%
    detWeights = mcxdetweight(detp, cfg.prop);
    detNum = length(unique(detp.detid));
    detWeightSum = zeros(detNum, 1);
    for i = 1 : length(detp.detid)
        detWeightSum(detp.detid(i)) = detWeightSum(detp.detid(i)) + detWeights(i);
    end
    area = pi * cfg.detpos(:,4).^2;
    dref = detWeightSum ./ area / cfg.nphoton; % Eq.12 of photon replay paper[Yao2018]
end