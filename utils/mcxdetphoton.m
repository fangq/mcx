function newdetp = mcxdetphoton(detp, medianum, savedetflag, issaveref, srcnum)
%
% newdetp=mcxdetphoton(detp, medianum, savedetflag)
% newdetp=mcxdetphoton(detp, medianum, savedetflag, issaveref, srcnum)
%
% Separating combined detected photon data into easy-to-read structure based on
% user-specified detected photon output format ("savedetflag")
%
% author: Qianqian Fang (q.fang <at> neu.edu)
%
% input:
%     detp: a 2-D array defining the combined detected photon data, usually
%           detp.data, where detp is the 2nd output from mcxlab
%     medianum: the total number of non-zero tissue types (row number of cfg.prop minus 1)
%     savedetflag: the cfg.savedetflag string, containing letters 'dspmxvwi' denoting different
%           output data fields, please see mcxlab's help
%     issaveref: the cfg.issaveref flag, 1 for saving diffuse reflectance, 0 not to save
%     srcnum: the cfg.srcnum flag, denoting the number of source patterns in the photon-sharing mode
%
% output:
%     newdetp: re-organized detected photon data as a struct; the mapping of the fields are
%              newdetp.detid: the ID(>0) of the detector that captures the photon (1)
%              newdetp.nscat: cummulative scattering event counts in each medium (#medium)
%              newdetp.ppath: cummulative path lengths in each medium, i.e. partial pathlength (#medium)
%                   one need to multiply cfg.unitinmm with ppath to convert it to mm.
%              newdetp.mom: cummulative cos_theta for momentum transfer in each medium (#medium)
%              newdetp.p or .v: exit position and direction, when cfg.issaveexit=1 (3)
%              newdetp.w0: photon initial weight at launch time (3)
%              newdetp.s: exit Stokes parameters for polarized photon (4)
%
% License: GPLv3, see http://mcx.space/ for details
%

c0 = 1;
len = 1;
if (regexp(savedetflag, '[dD]'))
    if (nargin > 3 && issaveref > 1)
        newdetp.w0 = detp(1, :)';
    else
        newdetp.detid = int32(detp(1, :))';
        if (any(newdetp.detid > 65535))
            newdetp.srcid = bitshift(newdetp.detid, -16);
            newdetp.detid = bitand(newdetp.detid, int32(hex2dec('ffff')));
        end
    end
    c0 = 2;
end
len = medianum;
if (regexp(savedetflag, '[sS]'))
    newdetp.nscat = int32(detp(c0:(c0 + len - 1), :))';    % 1st medianum block is num of scattering
    c0 = c0 + len;
end
if (regexp(savedetflag, '[pP]'))
    newdetp.ppath = detp(c0:(c0 + len - 1), :)'; % 2nd medianum block is partial path
    c0 = c0 + len;
end
if (regexp(savedetflag, '[mM]'))
    newdetp.mom = detp(c0:(c0 + len - 1), :)'; % 3rd medianum block is the momentum transfer
    c0 = c0 + len;
end
len = 3;
if (regexp(savedetflag, '[xX]'))
    newdetp.p = detp(c0:(c0 + len - 1), :)';             % columns 7-5 from the right store the exit positions
    c0 = c0 + len;
end
if (regexp(savedetflag, '[vV]'))
    newdetp.v = detp(c0:(c0 + len - 1), :)';        % columns 4-2 from the right store the exit dirs
    c0 = c0 + len;
end
if (regexp(savedetflag, '[wW]'))
    len = 1;
    newdetp.w0 = detp(c0:(c0 + len - 1), :)';  % last column is the initial packet weight
    if (nargin > 4 && srcnum > 1)
        newdetp.w0 = typecast(newdetp.w0, 'uint32');
    end
    c0 = c0 + len;
end
if (regexp(savedetflag, '[iI]'))
    len = 4;
    newdetp.s = detp(c0:(c0 + len - 1), :)';
    c0 = c0 + len;
end
