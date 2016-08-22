function rois=get_similir(imdb)
    %whitening information
    hog_spec.ny = 8;
    hog_spec.nx = 8;
    hog_spec.nf = 31;
    load('bg11.mat'); % variable "bg" whitening HoG informatiom
    [R, mu_bg] = whiten(bg, hog_spec.nx, hog_spec.ny);
    hog_spec.R = R; 
    hog_spec.mu_bg = mu_bg;

    %edge boxes setting
    edge_model=load('edge/modelBsds');
    edge_model=edge_model.model;
    edge_model.opts.multiscale=0; edge_model.opts.sharpen=2; edge_model.opts.nThreads=4;
    edge_opts = edgeBoxes;
    edge_opts.alpha = .65;     % step size of sliding window search
    edge_opts.beta  = .75;     % nms threshold for object proposals
    edge_opts.minScore = .01;  % min score of boxes to detect
    edge_opts.maxBoxes = 1e4;  % max number of boxes to detect

    selectrate=100; %select at most 100 bounding box
    %feature
    all_hog=cell(1,length(imdb.image_ids));
    all_box=cell(1,length(imdb.image_ids));
    all_score=cell(1,length(imdb.image_ids));
    for ind=1:length(imdb.image_ids)
%         img=imread(database.img{ind});
        img=imread(imdb.image_at(ind));
        bbs=double(edgeBoxes(img,edge_model,edge_opts));
        if(size(bbs,1)>selectrate)
            bbs(selectrate+1:end,:)=[];
        end
        all_score{ind}=bbs(:,5);
        bbs(:,5)=[];
        box=bbs;
        box(:,3:4)=bbs(:,3:4)+bbs(:,1:2);
        %hog feature
        seg.coords = box; %[xmin, ymin, xmax, ymax]
        hh = extract_segfeat_hog(img,seg);
        desc = hh.hist';
        desc = desc - repmat(hog_spec.mu_bg, 1, size(desc, 2));
        desc = hog_spec.R \ (hog_spec.R' \ desc);
        desc = [desc; (-desc' * hog_spec.mu_bg)'];
        all_box{ind}=box;
        all_hog{ind}=desc;
        tic_toc_print('feat : %d/%d\n', ind,length(imdb.image_ids));
    end
    %similiar image
    S=zeros(length(imdb.image_ids));
    for l_ind=1:length(imdb.image_ids)
        for r_ind=l_ind+1:length(imdb.image_ids)
            desc=all_hog{l_ind};
            desc=desc';
            r_desc=all_hog{r_ind};
            D=desc*r_desc;
            a=max(D(:));
            S(l_ind,r_ind)=a;
            S(r_ind,l_ind)=a;
        end
        tic_toc_print('similiar : %d/%d\n', l_ind, length(imdb.image_ids));
    end
    %find positive and negative pair
    [~,list]=sort(S,2,'descend');
    Ntop=10;% find 10 neighbor
    rois=[];
    idx=list(:,1:Ntop);
    for ind=1:length(imdb.image_ids)
        desc1=all_hog{ind};%this image whitening HoG feature
        ll=zeros(Ntop,4);
        rois(ind).boxes=all_box{ind};%bounding box [xmin ymin w h]
        rois(ind).score=all_score{ind};%edge box score
        rois(ind).idx=idx(ind,:);%all neighboring image index
        target=zeros(1,Ntop);
        for i=1:Ntop      %calculate all neighboring image result
            ind2=idx(ind,i);%neighboring image index
            desc2=all_hog{ind2};%neighboring image whitening HoG feature
            D=desc1'*desc2; %calculate similar
            [a,b]=sort(D,2,'descend');
            [~,b1]=max(a(:,1)); %the most similar pair
            bbox=all_box{ind2};
            bbox(:,3:4)=bbox(:,3:4)-bbox(:,1:2);
            DD=cmp_IOU(bbox)-1;%more different bounding box
            DD=DD(b(b1,1),:).*D(b1,:);%more different bounding box and unsimiliar
            [~,c]=sort(DD,'descend');
            ll(i,:)=[b(b1,1) c(1:3)];%one positive and three negative to choose
            %the bounding box we use in neighboring image
            target(i)=b1;%the bounding box we use in this image
        end
        rois(ind).desc=desc1;
        rois(ind).target=target;
        rois(ind).pair=ll;
        rois(ind).image=imdb.image_at(ind);
        tic_toc_print('rois : %d/%d\n', ind,length(imdb.image_ids));
    end
end