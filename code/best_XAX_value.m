function [X,W]=best_XAX_value(SR,TR,sim,T_i,topN)
    if(topN>size(SR,1))
        topN=size(SR,1)/2;
    end
    %topN
    move=SR-TR(T_i,T_i,:);
    S_ij=sum(move.^2,3);
    W_ij=exp(-S_ij);
    for i=1:size(W_ij,1)
        W_ij(i,i)=0;
    end
    %proposal i value
    N=numel(T_i);
    ind=(1:N)'+(T_i-1)*N;
    value=sim(ind);
    value=value<=0;
    B=bsxfun(@plus,value,value');
    B=B>0;
    W_ij(B)=0;
    B=bsxfun(@eq,T_i,T_i');
    W_ij(B)=0;
    X=ones(size(W_ij,1),1);
    X=X/numel(X);
    N1=1/numel(X);
    i=0;
    while(sum(X>=N1)>topN&&i<100)
        N3=X'*W_ij*X;
        X=X.*(W_ij*X/N3);
        i=i+1;
    end
end