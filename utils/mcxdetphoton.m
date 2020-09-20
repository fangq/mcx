function newdetp=mcxdetphoton(detp,medianum, savedetflag, issaveref, srcnum)

c0=1;
len=1;
if(regexp(savedetflag,'[dD]'))
    if(nargin>3 && issaveref>1)
        newdetp.w0=detp(1,:)';
    else
        newdetp.detid=int32(detp(1,:))';
    end
    c0=2;
end
len=medianum;
if(regexp(savedetflag,'[sS]'))
    newdetp.nscat=int32(detp(c0:(c0+len-1),:))';    % 1st medianum block is num of scattering
    c0=c0+len;
end
if(regexp(savedetflag,'[pP]'))
    newdetp.ppath=detp(c0:(c0+len-1),:)';% 2nd medianum block is partial path
    c0=c0+len;
end
if(regexp(savedetflag,'[mM]'))
    newdetp.mom=detp(c0:(c0+len-1),:)'; % 3rd medianum block is the momentum transfer
    c0=c0+len;
end
len=3;
if(regexp(savedetflag,'[xX]'))
    newdetp.p=detp(c0:(c0+len-1),:)';             %columns 7-5 from the right store the exit positions
    c0=c0+len;
end
if(regexp(savedetflag,'[vV]'))
    newdetp.v=detp(c0:(c0+len-1),:)';	     %columns 4-2 from the right store the exit dirs
    c0=c0+len;
end
if(regexp(savedetflag,'[wW]'))
    len=1;
    newdetp.w0=detp(c0:(c0+len-1),:)';  % last column is the initial packet weight
    if(nargin>4 && srcnum>1)
        newdetp.w0=typecast(newdetp.w0,'uint32');
    end
    c0=c0+len;
end