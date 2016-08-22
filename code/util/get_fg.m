function map=get_fg(H,W,bbs,X)
    map=zeros(H,W);
    for i=1:numel(X)
        ii=X(i);
        v1=bbs(ii,:);
        sy=v1(1);
        sx=v1(2);
        ey=v1(1)+v1(3);
        ex=v1(2)+v1(4);
        map(sx:ex,sy:ey)=map(sx:ex,sy:ey)+1;
    end