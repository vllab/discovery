function BB=cmp_IOU(bbs)
    bbs=single(bbs);
    DD=fullIntersectionMex(bbs);
    B=bbs(:,3).*bbs(:,4);
    N=length(B);
    BB=DD./(repmat(B,1,N)+repmat(B',N,1)-DD);
    for i=1:N
        BB(i,i)=1;
    end
