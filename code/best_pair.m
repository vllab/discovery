function [X,T_i]=best_pair(source,target,topN,iter_max)
    %source vote on target
    r_desc=source.desc;
    R_relation=source.relation;

    sim=r_desc'*target.desc;
    sim(sim<=0)=0;
    A=sim;
    [~,S_i]=max(A,[],2);
    score=best_XAX_value(R_relation,target.relation,sim,S_i,topN);
    for iter=1:iter_max
        [~,ind]=sort(score,'descend');
        ind=ind(1:topN);
        %update A
        A=updateR(R_relation,target.relation,S_i,sim,ind);
        [~,S_i]=max(A,[],2);
        score=best_XAX_value(R_relation,target.relation,sim,S_i,topN);
    end
    % source score to target score
    X=zeros(size(target.box,1),1);
    T_i=zeros(size(target.box,1),1);
    for i=1:numel(S_i)
        ii=S_i(i);
        if(X(ii)<score(i))
            X(ii)=score(i);
            T_i(ii)=i;
        end
    end
end