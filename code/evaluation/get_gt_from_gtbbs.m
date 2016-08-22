function gt=get_gt_from_gtbbs(gt_img_dir)
    gtbbs=get_bbox_from_mat(gt_img_dir);
    gtbbs(:,3:4)=gtbbs(:,3:4)+gtbbs(:,1:2);
    gt=cell(1,gtbbs(end,5));
    for i=1:size(gtbbs)
        ind=gtbbs(i,5);
        gt{ind}=[gt{ind};gtbbs(i,1:4)];
    end