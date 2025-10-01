function report = serialcorr(randomseq, maxshift)
%  report=serialcorr(randomseq,maxshift)
%
%  serial correlation function in a random sequence at a range of separations
%
%    author: Qianqian Fang (q.fang <at> neu.edu)
%
%    input:
%        randomseq: a random sequence (1D array)
%        maxshift:  the maximum separation to test with
%
%    output:
%        report: the corr. coeff for the sequence between randomseq
%            and randomseq(i:end) where i<=maxshift
%
%    this file is part of Monte Carlo eXtreme (MCX)
%    License: GPLv3, see https://mcx.space for details
%    see Boas2002, Heskell1996

report = zeros(maxshift, 2);
for i = 1:maxshift
    [r, p] = corrcoef(randomseq(1:end - i), randomseq(i + 1:end));
    report(i, :) = [r(1, 2), p(1, 2)];
    fprintf(1, 'test shift %d (r=%20.16f p=%20.16f)\n', i, r(1, 2), p(1, 2));
end
