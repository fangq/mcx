function separation=getdistance(srcpos,detpos)
srcnum=length(srcpos(:,1));
detnum=length(detpos(:,1));
for s=1:srcnum
	for r=1:detnum
             separation(r,s)=norm(srcpos(s,:)-detpos(r,:),'fro');
	end
end
