function A=updateR(SR,TR,T_i,sim,ind)
    A=zeros(size(sim));
    sub_SR=cell(1,numel(ind));
    sub_TR=cell(1,numel(ind));
    for i=1:numel(ind)
        s_ind=ind(i);
        t_ind=T_i(s_ind);
        sub_TR{i}=squeeze(TR(:,t_ind,:)).*sim(s_ind,t_ind);
        sub_SR{i}=squeeze(SR(:,s_ind,:)).*sim(s_ind,t_ind);
    end
    TR_desc=cell2mat(sub_TR);
    SR_desc=cell2mat(sub_SR);
    A=exp(-tool_dist(SR_desc,TR_desc)./numel(ind)).*sim;
end