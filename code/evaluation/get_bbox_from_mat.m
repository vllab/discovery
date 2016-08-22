function gtbbs=get_bbox_from_mat(gt_img_dir)
    gtdata = dirimg(gt_img_dir,'*.mat');
    ind=0;
    for i=1:gtdata.imnum
        box_list=load(gtdata.img{i});
        box=box_list.bbox_list;
        for j=1:length(box)
            ind=ind+1;
            gtbbs(ind,:)=[box{j} i];
        end
    end
    gtbbs(:,3)=gtbbs(:,3)-gtbbs(:,1);
    gtbbs(:,4)=gtbbs(:,4)-gtbbs(:,2);