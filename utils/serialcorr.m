function report=serialcorr(randomseq,maxshift)

report=zeros(maxshift,2);
for i=1:maxshift
    [r,p]=corrcoef(randomseq(1:end-i),randomseq(i+1:end));
    report(i,:)=[r(1,2),p(1,2)];
    fprintf(1,'test shift %d (r=%20.16f p=%20.16f)\n',i,r(1,2),p(1,2));
end
